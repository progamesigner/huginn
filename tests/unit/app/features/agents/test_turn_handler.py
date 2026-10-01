from collections.abc import Sequence
from datetime import UTC, datetime
from typing import override
from unittest.mock import AsyncMock

import pytest

from app.features.agents.config import AgentConfig, DefaultsConfig
from app.features.agents.contracts import (
    NamedTool,
    Tool,
    ToolHarvest,
    ToolSource,
    ToolSourceRegistry,
    TranscriptServiceContract,
)
from app.features.agents.entities import (
    RoutingContext,
    RoutingPolicy,
    RoutingTurn,
    SelectionPolicy,
    TurnBudget,
    TurnPrompt,
    TurnRequest,
    TurnScope,
    resolve_budget,
)
from app.features.agents.exceptions import (
    MalformedToolCallError,
    UnservableModelError,
)
from app.features.agents.router import Router
from app.features.agents.services import TurnHandler, _turn_scope
from lib.llm import (
    CompletionChunk,
    FinishChunk,
    LLMClient,
    LLMContract,
    Message,
    ModelRef,
    ProviderConfig,
    TextChunk,
    TextSegment,
    ToolCallChunk,
)
from lib.temporal import ClockContract, ClockManager
from tests.helpers.clock import FakeClock
from tests.helpers.logging import make_logger
from tests.helpers.replies import RecordingReply

from .test_agent_runner import EchoTool, FakeLLM, SleepTool, StallingLLM

DEFAULTS = DefaultsConfig(model='default-model')


class FakeModels(LLMContract):
    client: LLMClient | None
    refs: list[ModelRef]

    def __init__(self, client: LLMClient | None) -> None:
        self.client = client
        self.refs = []

    @override
    def client_for(self, ref: ModelRef) -> LLMClient | None:
        self.refs.append(ref)
        return self.client

    @override
    def provider_for(self, ref: ModelRef) -> ProviderConfig | None:
        return None


class FakeTranscript:
    prior: tuple[Message, ...]
    recorded: list[tuple[tuple[str, ...], str]]

    def __init__(self, prior: tuple[Message, ...] = ()) -> None:
        self.prior = prior
        self.recorded = []


class FakeStore(TranscriptServiceContract):
    maintain_error: Exception | None
    maintained: list[tuple[str, str, str, ModelRef]]
    opened: list[tuple[str, str]]
    record_error: Exception | None
    transcript: FakeTranscript

    def __init__(
        self,
        transcript: FakeTranscript | None = None,
        *,
        maintain_error: Exception | None = None,
        record_error: Exception | None = None,
    ) -> None:
        self.maintain_error = maintain_error
        self.maintained = []
        self.opened = []
        self.record_error = record_error
        self.transcript = transcript or FakeTranscript()

    @override
    async def history(
        self,
        channel: str,
        chat: str,
    ) -> tuple[Message, ...]:
        self.opened.append((channel, chat))
        return self.transcript.prior

    @override
    async def document_blobs(self, channel: str, chat: str) -> frozenset[str]:
        return frozenset()

    @override
    async def record(
        self,
        channel: str,
        chat: str,
        agent: str,
        prompts: Sequence[TurnPrompt],
        answer: str,
    ) -> None:
        if self.record_error is not None:
            raise self.record_error

        self.transcript.recorded.append(
            (tuple(prompt.text for prompt in prompts), answer),
        )

    @override
    async def maintain(
        self,
        channel: str,
        chat: str,
        agent: str,
        ref: ModelRef,
    ) -> None:
        self.maintained.append((channel, chat, agent, ref))
        if self.maintain_error is not None:
            raise self.maintain_error


class SharedFakeStore(FakeStore):
    @override
    async def record(
        self,
        channel: str,
        chat: str,
        agent: str,
        prompts: Sequence[TurnPrompt],
        answer: str,
    ) -> None:
        await super().record(channel, chat, agent, prompts, answer)
        replay = tuple(
            Message(role='user', segments=(TextSegment(text=prompt.text),))
            for prompt in prompts
            if prompt.text
        )
        if answer:
            replay = (
                *replay,
                Message(role='assistant', segments=(TextSegment(text=answer),)),
            )
        self.transcript.prior = (*self.transcript.prior, *replay)


def agents() -> dict[str, AgentConfig]:
    return {
        'scout': AgentConfig(
            description='scouts',
            instructions='be helpful',
            model='scout-model',
        ),
    }


def answering(text: str = 'answer') -> FakeLLM:
    return FakeLLM([[TextChunk(text=text), FinishChunk(reason='stop')]])


class FakeSource(ToolSource):
    named: tuple[NamedTool, ...]
    unreachable: tuple[str, ...]

    def __init__(
        self,
        *names: str,
        tool: Tool | None = None,
        unreachable: tuple[str, ...] = (),
    ) -> None:
        self.named = tuple((name, tool or EchoTool()) for name in names)
        self.unreachable = unreachable

    @override
    async def collect(self, agent: AgentConfig, scope: TurnScope) -> ToolHarvest:
        return ToolHarvest(tools=self.named, unreachable=self.unreachable)


def offered(llm: FakeLLM) -> list[str]:
    return [
        schema['function']['name']
        for schema in llm.requests[0].tools
        if isinstance(schema, dict) and isinstance(function := schema['function'], dict)
        for name in [function['name']]
        if isinstance(name, str)
    ]


def _registry(source: ToolSource) -> ToolSourceRegistry:
    return ToolSourceRegistry(lambda: (source,))


def handler_with(llm: FakeLLM, source: ToolSource) -> TurnHandler:
    sources = _registry(source)

    return TurnHandler(
        llm=FakeModels(llm),
        router=Router(agents(), FakeModels(llm), DEFAULTS, logger=make_logger()),
        sources=sources,
        store=FakeStore(),
        clock=ClockManager(),
        logger=make_logger(),
    )


def make_handler(
    llm: FakeLLM | None,
    store: FakeStore,
    clock: ClockContract | None = None,
    *,
    default_max_completion_tokens: int | None = None,
) -> TurnHandler:
    return TurnHandler(
        llm=FakeModels(llm),
        router=Router(agents(), FakeModels(llm), DEFAULTS, logger=make_logger()),
        sources=ToolSourceRegistry(),
        store=store,
        clock=clock or ClockManager(),
        default_max_completion_tokens=default_max_completion_tokens,
        logger=make_logger(),
    )


def configured_handler(
    llm: FakeLLM | StallingLLM,
    store: FakeStore,
    source: ToolSource | None = None,
    **overrides: object,
) -> TurnHandler:
    agent = agents()['scout'].model_copy(update=overrides)
    sources = ToolSourceRegistry(lambda: () if source is None else (source,))

    return TurnHandler(
        llm=FakeModels(llm),
        router=Router(
            {'scout': agent},
            FakeModels(llm),
            DEFAULTS,
            logger=make_logger(),
        ),
        sources=sources,
        store=store,
        clock=ClockManager(),
        logger=make_logger(),
    )


def request(*texts: str) -> TurnRequest:
    return TurnRequest(
        channel='support_bot',
        chat='4242',
        policy=RoutingPolicy(agent='scout'),
        prompts=tuple(TurnPrompt(text=text, actor='user:1') for text in texts),
    )


def test_turn_scope_carries_the_last_prompts_identity() -> None:
    turn = request('hello')
    turn = turn.model_copy(
        update={
            'prompts': (
                TurnPrompt(
                    text='hello',
                    actor='user:1',
                    account='account-1',
                    handle='alice',
                ),
            ),
        },
    )

    assert _turn_scope(turn, 'scout', logger=make_logger()) == TurnScope(
        actor='user:1',
        agent='scout',
        channel='support_bot',
        chat='4242',
        account='account-1',
        handle='alice',
    )


def test_turn_scope_withholds_identity_when_accounts_are_mixed() -> None:
    turn = request('first', 'second').model_copy(
        update={
            'prompts': (
                TurnPrompt(
                    text='first',
                    actor='user:1',
                    account='account-1',
                    handle='alice',
                ),
                TurnPrompt(
                    text='second',
                    actor='user:2',
                    account='account-2',
                    handle='bob',
                ),
            ),
        },
    )

    scope = _turn_scope(turn, 'scout', logger=make_logger())

    assert scope.actor == 'user:2'
    assert scope.account is None
    assert scope.handle is None


@pytest.mark.parametrize('linked_last', [True, False])
def test_turn_scope_withholds_identity_when_an_unlinked_actor_joins(
    linked_last: bool,
) -> None:
    linked = TurnPrompt(
        text='mine',
        actor='user:1',
        account='account-1',
        handle='alice',
    )
    unlinked = TurnPrompt(text='theirs', actor='user:2')
    prompts = (unlinked, linked) if linked_last else (linked, unlinked)
    turn = request('first', 'second').model_copy(update={'prompts': prompts})

    scope = _turn_scope(turn, 'scout', logger=make_logger())

    assert scope.account is None
    assert scope.handle is None


async def test_a_turn_streams_the_answer_and_finishes_the_reply() -> None:
    store = FakeStore()
    reply = RecordingReply()

    await make_handler(answering(), store).handle(request('hello'), reply)

    assert reply.fragments == ['answer']
    assert reply.finished
    assert not reply.abandoned


async def test_request_token_limit_overrides_agent_and_default() -> None:
    llm = answering()
    configured = agents()['scout'].model_copy(update={'max_completion_tokens': 8192})
    handler = TurnHandler(
        llm=FakeModels(llm),
        router=Router(
            {'scout': configured},
            FakeModels(llm),
            DEFAULTS,
            logger=make_logger(),
        ),
        sources=ToolSourceRegistry(),
        store=FakeStore(),
        clock=ClockManager(),
        default_max_completion_tokens=16384,
        logger=make_logger(),
    )

    await handler.handle(
        request('hello').model_copy(update={'max_completion_tokens': 4096}),
        RecordingReply(),
    )

    assert llm.requests[0].max_completion_tokens == 4096


async def test_agent_token_limit_overrides_default() -> None:
    llm = answering()
    configured = agents()['scout'].model_copy(update={'max_completion_tokens': 8192})
    handler = TurnHandler(
        llm=FakeModels(llm),
        router=Router(
            {'scout': configured},
            FakeModels(llm),
            DEFAULTS,
            logger=make_logger(),
        ),
        sources=ToolSourceRegistry(),
        store=FakeStore(),
        clock=ClockManager(),
        default_max_completion_tokens=16384,
        logger=make_logger(),
    )

    await handler.handle(request('hello'), RecordingReply())

    assert llm.requests[0].max_completion_tokens == 8192


async def test_default_token_limit_is_used_when_request_and_agent_omit_it() -> None:
    llm = answering()

    await make_handler(
        llm,
        FakeStore(),
        default_max_completion_tokens=16384,
    ).handle(request('hello'), RecordingReply())

    assert llm.requests[0].max_completion_tokens == 16384


async def test_omitted_token_limit_preserves_the_provider_default() -> None:
    llm = answering()

    await make_handler(llm, FakeStore()).handle(request('hello'), RecordingReply())

    assert llm.requests[0].max_completion_tokens is None


async def test_a_turn_tells_the_reply_which_agent_took_it() -> None:
    # NOTE: routing happens before the first token, so a client can name the agent it is
    # talking to while the answer is still streaming
    reply = RecordingReply()

    await make_handler(answering(), FakeStore()).handle(request('hello'), reply)

    assert reply.routed == ['scout']


async def test_a_turn_offers_the_model_every_tool_the_request_left_alone() -> None:
    llm = answering()

    await handler_with(llm, FakeSource('echo', 'recall')).handle(
        request('hello'),
        RecordingReply(),
    )

    assert offered(llm) == ['echo', 'recall']


# NOTE: silence here was the actively harmful part of an outage -- the agent answered as
# though it had consulted a memory server it never reached
async def test_a_turn_tells_the_model_which_servers_it_could_not_reach() -> None:
    llm = answering()

    await handler_with(
        llm,
        FakeSource('echo', unreachable=('muninn', 'skills')),
    ).handle(request('hello'), RecordingReply())

    instructions = llm.requests[0].messages[0]
    assert instructions.role == 'system'
    said = ' '.join(
        segment.text
        for segment in instructions.segments
        if isinstance(segment, TextSegment)
    )
    assert 'muninn, skills' in said
    assert 'be helpful' in said


async def test_a_turn_with_every_server_reachable_says_nothing_extra() -> None:
    llm = answering()

    await handler_with(llm, FakeSource('echo')).handle(
        request('hello'),
        RecordingReply(),
    )

    instructions = llm.requests[0].messages[0]
    assert [
        segment.text
        for segment in instructions.segments
        if isinstance(segment, TextSegment)
    ] == ['be helpful']


async def test_a_turn_never_offers_the_model_a_denied_tool() -> None:
    # NOTE: the api key that opened the turn may take a tool away from the agent, and
    # the only place that can be observed is what the model was offered
    llm = answering()

    await handler_with(llm, FakeSource('echo', 'recall')).handle(
        request('hello').model_copy(update={'denied_tools': frozenset({'recall'})}),
        RecordingReply(),
    )

    assert offered(llm) == ['echo']


async def test_a_turn_opens_the_transcript_of_the_agent_it_routed_to() -> None:
    store = FakeStore()

    await make_handler(answering(), store).handle(request('hello'), RecordingReply())

    assert store.opened == [('support_bot', '4242')]


async def test_a_turn_records_its_prompts_and_answer() -> None:
    store = FakeStore()

    await make_handler(answering(), store).handle(
        request('first', 'second'),
        RecordingReply(),
    )

    assert store.transcript.recorded == [(('first', 'second'), 'answer')]


async def test_a_turn_replays_the_stored_history_into_the_model() -> None:
    prior = (Message(role='user', segments=(TextSegment(text='earlier'),)),)
    store = FakeStore(FakeTranscript(prior))
    llm = answering()

    await make_handler(llm, store).handle(request('hello'), RecordingReply())

    sent = llm.requests[0].messages
    # NOTE: system instructions, then the stored history, then this round's prompts
    assert [message.role for message in sent] == ['system', 'user', 'user']
    assert sent[1] is prior[0]


async def test_a_chat_replays_history_after_routing_to_a_different_agent() -> None:
    store = SharedFakeStore()
    llm = FakeLLM(
        [
            [TextChunk(text='one'), FinishChunk(reason='stop')],
            [TextChunk(text='two'), FinishChunk(reason='stop')],
        ],
    )
    configs = {
        'quick': AgentConfig(description='quick', instructions='answer briefly'),
        'deep': AgentConfig(description='deep', instructions='answer fully'),
    }
    models = FakeModels(llm)
    handler = TurnHandler(
        llm=models,
        router=Router(configs, models, DEFAULTS, logger=make_logger()),
        sources=ToolSourceRegistry(),
        store=store,
        clock=ClockManager(),
        defaults=DEFAULTS,
        logger=make_logger(),
    )

    await handler.handle(
        request('first').model_copy(update={'policy': RoutingPolicy(agent='quick')}),
        RecordingReply(),
    )
    await handler.handle(
        request('second').model_copy(update={'policy': RoutingPolicy(agent='deep')}),
        RecordingReply(),
    )

    second = llm.requests[1].messages
    assert [message.role for message in second] == [
        'system',
        'user',
        'assistant',
        'user',
    ]
    assert isinstance(second[1].segments[0], TextSegment)
    assert second[1].segments[0].text == 'first'


async def test_every_prompt_of_a_coalesced_round_reaches_the_model() -> None:
    llm = answering()

    await make_handler(llm, FakeStore()).handle(
        request('first', 'second'),
        RecordingReply(),
    )

    texts = [
        segment.text
        for message in llm.requests[0].messages
        if message.role == 'user'
        # NOTE: segments[0] is the current-time stamp every user message carries
        for segment in message.segments[1:]
        if isinstance(segment, TextSegment)
    ]
    assert texts == ['first', 'second']


async def test_compaction_runs_after_the_turn_committed() -> None:
    store = FakeStore()

    await make_handler(answering(), store).handle(request('hello'), RecordingReply())

    assert store.maintained == [
        ('support_bot', '4242', 'scout', ModelRef(model='scout-model')),
    ]


async def test_a_turn_asks_the_provider_the_agent_pinned() -> None:
    llm = answering()
    models = FakeModels(llm)
    pinned = agents()['scout'].model_copy(update={'provider': 'bifrost'})
    handler = TurnHandler(
        llm=models,
        router=Router(
            {'scout': pinned},
            FakeModels(llm),
            DEFAULTS,
            logger=make_logger(),
        ),
        sources=ToolSourceRegistry(),
        store=FakeStore(),
        clock=ClockManager(),
        logger=make_logger(),
    )

    await handler.handle(request('hello'), RecordingReply())

    assert models.refs == [ModelRef(model='scout-model', provider='bifrost')]


async def test_an_agent_without_a_model_inherits_the_default_provider() -> None:
    # NOTE: the paired-inheritance rule -- defaults.provider rides along only when
    # defaults.model is what got used
    llm = answering()
    models = FakeModels(llm)
    bare = AgentConfig(description='scouts', instructions='be helpful')
    defaults = DefaultsConfig(model='default-model', provider='openai')
    handler = TurnHandler(
        llm=models,
        router=Router({'scout': bare}, FakeModels(llm), defaults, logger=make_logger()),
        sources=ToolSourceRegistry(),
        store=FakeStore(),
        clock=ClockManager(),
        defaults=defaults,
        logger=make_logger(),
    )

    await handler.handle(request('hello'), RecordingReply())

    assert models.refs == [ModelRef(model='default-model', provider='openai')]


async def test_an_agent_naming_only_a_model_does_not_inherit_the_default_provider() -> (
    None
):
    llm = answering()
    models = FakeModels(llm)
    defaults = DefaultsConfig(model='default-model', provider='openai')
    handler = TurnHandler(
        llm=models,
        router=Router(agents(), FakeModels(llm), defaults, logger=make_logger()),
        sources=ToolSourceRegistry(),
        store=FakeStore(),
        clock=ClockManager(),
        defaults=defaults,
        logger=make_logger(),
    )

    await handler.handle(request('hello'), RecordingReply())

    assert models.refs == [ModelRef(model='scout-model')]


# NOTE: the live failure -- tool assembly or the stream raised, nothing was recorded, and
# the next turn could not tell the round had ever happened
async def test_a_turn_that_raises_still_records_what_it_had_said() -> None:
    store = FakeStore()
    reply = RecordingReply()
    llm = FakeLLM(
        [
            [
                TextChunk(text='let me look'),
                ToolCallChunk(index=0, arguments='{}'),
                FinishChunk(reason='tool_calls'),
            ],
        ],
    )

    with pytest.raises(MalformedToolCallError):
        await make_handler(llm, store).handle(request('hello'), reply)

    assert store.transcript.recorded == [(('hello',), 'let me look')]
    assert not reply.finished


async def test_a_turn_that_raises_before_answering_records_the_prompt_alone() -> None:
    store = FakeStore()

    with pytest.raises(MalformedToolCallError):
        await make_handler(
            FakeLLM([[ToolCallChunk(index=0, arguments='{}')]]),
            store,
        ).handle(request('hello'), RecordingReply())

    assert store.transcript.recorded == [(('hello',), '')]


class BrokenSource(ToolSource):
    @override
    async def collect(self, agent: AgentConfig, scope: TurnScope) -> ToolHarvest:
        raise RuntimeError('the tool server is unreachable')


# NOTE: this is where the live MCP failure landed -- assembling the tool set threw before
# the model was ever asked, and the round left no trace at all
async def test_a_turn_that_cannot_assemble_its_tools_still_records_the_prompt() -> None:
    store = FakeStore()
    handler = TurnHandler(
        llm=FakeModels(answering()),
        router=Router(
            agents(),
            FakeModels(answering()),
            DEFAULTS,
            logger=make_logger(),
        ),
        sources=_registry(BrokenSource()),
        store=store,
        clock=ClockManager(),
        logger=make_logger(),
    )

    with pytest.raises(RuntimeError, match='unreachable'):
        await handler.handle(request('hello'), RecordingReply())

    assert store.transcript.recorded == [(('hello',), '')]


async def test_a_failing_salvage_does_not_replace_the_turns_own_failure() -> None:
    store = FakeStore(record_error=RuntimeError('the database is down'))

    with pytest.raises(MalformedToolCallError):
        await make_handler(
            FakeLLM([[ToolCallChunk(index=0, arguments='{}')]]),
            store,
        ).handle(request('hello'), RecordingReply())


# NOTE: the handler names the ending and the channel words it -- so what is asserted here
# is the signal, and how it reads is TelegramReply's business
async def test_a_timed_out_turn_tells_the_reply_it_ran_out_of_time() -> None:
    reply = RecordingReply()
    store = FakeStore()

    await configured_handler(
        StallingLLM('half an '),
        store,
        turn_timeout=0.05,
    ).handle(request('hello'), reply)

    assert reply.endings == ['timeout']
    assert reply.finished
    # NOTE: the ending is for the reader, never folded into the model's own words
    assert store.transcript.recorded == [(('hello',), 'half an ')]


async def test_a_turn_out_of_iterations_tells_the_reply_it_ran_out_of_steps() -> None:
    def calling() -> list[CompletionChunk]:
        return [
            ToolCallChunk(index=0, id='call_1', name='echo', arguments='{"text": "a"}'),
            FinishChunk(reason='tool_calls'),
        ]

    reply = RecordingReply()

    await configured_handler(
        FakeLLM([calling()]),
        FakeStore(),
        FakeSource('echo'),
        max_iterations=1,
    ).handle(request('hello'), reply)

    assert reply.endings == ['exhausted']
    assert reply.finished


async def test_a_completed_turn_names_no_ending_at_all() -> None:
    reply = RecordingReply()

    await make_handler(answering(), FakeStore()).handle(request('hello'), reply)

    assert reply.endings == []
    assert reply.fragments == ['answer']


async def test_a_failed_compaction_does_not_fail_a_delivered_answer() -> None:
    store = FakeStore(maintain_error=RuntimeError('summariser died'))
    reply = RecordingReply()

    await make_handler(answering(), store).handle(request('hello'), reply)

    assert reply.finished


# NOTE: the live failure this guards against -- the budget expired mid-answer, the runner
# raised, and the rollback took the round's prompts and answer out of the transcript with
# it, so the next turn had no idea the conversation had happened
async def test_an_expired_turn_still_records_what_it_managed_to_say() -> None:
    store = FakeStore()
    reply = RecordingReply()

    await configured_handler(
        StallingLLM('half an '),
        store,
        turn_timeout=0.05,
    ).handle(request('hello'), reply)

    assert store.transcript.recorded == [(('hello',), 'half an ')]
    assert reply.finished
    assert not reply.abandoned


async def test_the_agents_iteration_cap_reaches_the_runner() -> None:
    def calling() -> list[CompletionChunk]:
        return [
            ToolCallChunk(index=0, id='call_1', name='echo', arguments='{"text": "a"}'),
            FinishChunk(reason='tool_calls'),
        ]

    llm = FakeLLM([calling(), calling()])

    await configured_handler(
        llm,
        FakeStore(),
        FakeSource('echo'),
        max_iterations=1,
    ).handle(request('hello'), RecordingReply())

    assert len(llm.requests) == 1


async def test_the_agents_tool_timeout_reaches_the_runner() -> None:
    llm = FakeLLM(
        [
            [
                ToolCallChunk(
                    index=0,
                    id='call_1',
                    name='sleep',
                    arguments='{"seconds": 5.0}',
                ),
                FinishChunk(reason='tool_calls'),
            ],
            [TextChunk(text='answer'), FinishChunk(reason='stop')],
        ],
    )
    reply = RecordingReply()

    await configured_handler(
        llm,
        FakeStore(),
        FakeSource('sleep', tool=SleepTool()),
        tool_timeout=0.05,
    ).handle(request('hello'), reply)

    assert reply.tools == [('sleep', 'started'), ('sleep', 'failed')]


async def test_an_agent_that_configures_nothing_keeps_the_built_in_budget() -> None:
    assert resolve_budget(agents()['scout'], DEFAULTS) == TurnBudget()


async def test_an_agent_budget_overrides_only_what_it_sets() -> None:
    agent = agents()['scout'].model_copy(update={'turn_timeout': 900.0})

    budget = resolve_budget(agent, DEFAULTS)

    assert budget.turn_timeout == 900.0
    assert budget.max_iterations == TurnBudget().max_iterations
    assert budget.tool_timeout == TurnBudget().tool_timeout


async def test_the_defaults_budget_backs_an_agent_that_sets_none() -> None:
    defaults = DEFAULTS.model_copy(update={'max_iterations': 30, 'turn_timeout': 600.0})

    budget = resolve_budget(agents()['scout'], defaults)

    assert budget.max_iterations == 30
    assert budget.turn_timeout == 600.0
    assert budget.tool_timeout == TurnBudget().tool_timeout


async def test_an_agent_budget_overrides_the_defaults_budget() -> None:
    defaults = DEFAULTS.model_copy(update={'max_iterations': 30, 'turn_timeout': 600.0})
    agent = agents()['scout'].model_copy(update={'max_iterations': 5})

    budget = resolve_budget(agent, defaults)

    assert budget.max_iterations == 5
    assert budget.turn_timeout == 600.0


async def test_a_model_without_a_provider_refuses_the_turn() -> None:
    store = FakeStore()

    with pytest.raises(UnservableModelError):
        await make_handler(None, store).handle(request('hello'), RecordingReply())

    assert store.opened == []


async def test_a_round_with_nothing_to_answer_is_not_a_turn() -> None:
    store = FakeStore()

    await make_handler(answering(), store).handle(request(), RecordingReply())

    assert store.opened == []
    assert store.maintained == []


async def test_every_user_message_of_the_round_carries_the_same_header() -> None:
    now = datetime(2026, 3, 4, 12, 30, tzinfo=UTC)
    llm = answering()

    await make_handler(llm, FakeStore(), FakeClock(now)).handle(
        request('first', 'second'),
        RecordingReply(),
    )

    headers = [
        segment.text
        for message in llm.requests[0].messages
        if message.role == 'user'
        for segment in message.segments[:1]
        if isinstance(segment, TextSegment)
    ]
    assert headers == [
        '[2026-03-04T12:30:00+00:00] (support_bot) user:1:\n\n',
        '[2026-03-04T12:30:00+00:00] (support_bot) user:1:\n\n',
    ]


@pytest.mark.parametrize(
    ('fixed', 'limit', 'expected_calls'),
    [(True, 6, 0), (False, 0, 0), (False, 2, 1)],
)
async def test_routing_context_loads_history_only_when_needed(
    fixed: bool,
    limit: int,
    expected_calls: int,
) -> None:
    store = FakeStore()
    store.recent_turns = AsyncMock(
        return_value=(
            (RoutingTurn(agent='a', user=('before',), answer='answer'),),
            'summary',
        ),
    )
    router = Router({}, FakeModels(None), logger=make_logger())
    router.resolve = AsyncMock(side_effect=RuntimeError('stop after routing'))
    handler = TurnHandler(
        FakeModels(None),
        router,
        ToolSourceRegistry(),
        store,
        ClockManager(),
        logger=make_logger(),
    )
    prompts = (TurnPrompt(text='one', actor='u'), TurnPrompt(text='two', actor='u'))
    request = TurnRequest(
        channel='test',
        chat='chat',
        policy=RoutingPolicy(
            agent='a' if fixed else None,
            selection=SelectionPolicy(fallback='a', history_turns=limit),
        ),
        prompts=prompts,
    )
    with pytest.raises(RuntimeError, match='stop after routing'):
        await handler.handle(request, RecordingReply())
    assert store.recent_turns.await_count == expected_calls
    context = router.resolve.call_args.args[2]
    assert isinstance(context, RoutingContext)
    assert context.current == prompts
    assert bool(context.turns) == bool(expected_calls)
