from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from typing import override

import pytest

from app.features.agents.config import AgentConfig
from app.features.agents.contracts import (
    NamedTool,
    ToolSource,
    ToolSourceRegistry,
    TranscriptContract,
    TranscriptStoreContract,
)
from app.features.agents.exceptions import UnservableModelError
from app.features.agents.router import Router
from app.features.agents.services import TurnHandler
from app.features.agents.types import (
    RoutingPolicy,
    TurnPrompt,
    TurnRequest,
)
from lib.llm import (
    FinishChunk,
    LLMClient,
    LLMContract,
    Message,
    ProviderConfig,
    TextChunk,
    TextSegment,
)
from lib.temporal import ClockContract, ClockManager
from tests.helpers.clock import FakeClock
from tests.helpers.replies import RecordingReply

from .test_agent_runner import EchoTool, FakeLLM


class FakeModels(LLMContract):
    client: LLMClient | None

    def __init__(self, client: LLMClient | None) -> None:
        self.client = client

    @override
    def client_for(self, model: str) -> LLMClient | None:
        return self.client

    @override
    def provider_for(self, model: str) -> ProviderConfig | None:
        return None


class FakeTranscript(TranscriptContract):
    prior: tuple[Message, ...]
    recorded: list[tuple[tuple[str, ...], str]]

    def __init__(self, prior: tuple[Message, ...] = ()) -> None:
        self.prior = prior
        self.recorded = []

    @override
    async def history(self) -> tuple[Message, ...]:
        return self.prior

    @override
    async def record(self, prompts: Sequence[TurnPrompt], answer: str) -> None:
        self.recorded.append((tuple(prompt.text for prompt in prompts), answer))


class FakeStore(TranscriptStoreContract):
    maintained: list[tuple[str, str, str, str]]
    opened: list[tuple[str, str, str]]
    transcript: FakeTranscript

    def __init__(
        self,
        transcript: FakeTranscript | None = None,
        *,
        maintain_error: Exception | None = None,
    ) -> None:
        self.maintained = []
        self.maintain_error = maintain_error
        self.opened = []
        self.transcript = transcript or FakeTranscript()

    @override
    def open(
        self,
        channel: str,
        chat: str,
        agent: str,
    ) -> AbstractAsyncContextManager[TranscriptContract]:
        self.opened.append((channel, chat, agent))
        return self._open()

    @override
    async def maintain(
        self,
        channel: str,
        chat: str,
        agent: str,
        model: str,
    ) -> None:
        self.maintained.append((channel, chat, agent, model))
        if self.maintain_error is not None:
            raise self.maintain_error

    @asynccontextmanager
    async def _open(self) -> AsyncIterator[TranscriptContract]:
        yield self.transcript


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

    def __init__(self, *names: str) -> None:
        self.named = tuple((name, EchoTool()) for name in names)

    @override
    async def collect(self, agent: AgentConfig) -> Sequence[NamedTool]:
        return self.named


def offered(llm: FakeLLM) -> list[str]:
    return [
        schema['function']['name']
        for schema in llm.requests[0].tools
        if isinstance(schema, dict) and isinstance(function := schema['function'], dict)
        for name in [function['name']]
        if isinstance(name, str)
    ]


def handler_with(llm: FakeLLM, source: ToolSource) -> TurnHandler:
    sources = ToolSourceRegistry()
    sources.register(lambda: source)

    return TurnHandler(
        llm=FakeModels(llm),
        router=Router(agents(), FakeModels(llm), 'default-model'),
        sources=sources,
        store=FakeStore(),
        clock=ClockManager(),
    )


def make_handler(
    llm: FakeLLM | None,
    store: FakeStore,
    clock: ClockContract | None = None,
) -> TurnHandler:
    return TurnHandler(
        llm=FakeModels(llm),
        router=Router(agents(), FakeModels(llm), 'default-model'),
        sources=ToolSourceRegistry(),
        store=store,
        clock=clock or ClockManager(),
    )


def request(*texts: str) -> TurnRequest:
    return TurnRequest(
        channel='support_bot',
        chat='4242',
        policy=RoutingPolicy(agent='scout'),
        prompts=tuple(TurnPrompt(text=text, actor='user:1') for text in texts),
    )


async def test_a_turn_streams_the_answer_and_finishes_the_reply() -> None:
    store = FakeStore()
    reply = RecordingReply()

    await make_handler(answering(), store).handle(request('hello'), reply)

    assert reply.fragments == ['answer']
    assert reply.finished
    assert not reply.abandoned


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


async def test_a_turn_never_offers_the_model_a_denied_tool() -> None:
    # NOTE: the api key that opened the turn may take a tool away from the agent, and
    # the only place that can be observed is what the model was offered
    llm = answering()

    await handler_with(llm, FakeSource('echo', 'recall')).handle(
        replace(request('hello'), denied_tools=frozenset({'recall'})),
        RecordingReply(),
    )

    assert offered(llm) == ['echo']


async def test_a_turn_opens_the_transcript_of_the_agent_it_routed_to() -> None:
    store = FakeStore()

    await make_handler(answering(), store).handle(request('hello'), RecordingReply())

    assert store.opened == [('support_bot', '4242', 'scout')]


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

    assert store.maintained == [('support_bot', '4242', 'scout', 'scout-model')]


async def test_a_failed_compaction_does_not_fail_a_delivered_answer() -> None:
    store = FakeStore(maintain_error=RuntimeError('summariser died'))
    reply = RecordingReply()

    await make_handler(answering(), store).handle(request('hello'), reply)

    assert reply.finished


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
