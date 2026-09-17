import asyncio
import logging
from base64 import b64encode
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import cast, override
from uuid import uuid4

from structlog.contextvars import bound_contextvars, get_contextvars

from app.contracts.channels import ChannelReplyContract, ToolPhase
from lib.llm import (
    CompletionChunk,
    CompletionRequest,
    FinishChunk,
    ImageSegment,
    LLMClient,
    LLMContract,
    Message,
    ModelRef,
    Segment,
    TextChunk,
    TextSegment,
    ThinkingChunk,
    ToolCallChunk,
    ToolCallSegment,
    ToolResultSegment,
)
from lib.temporal import ClockContract, isoformat

from .config import (
    AgentConfig,
    DefaultsConfig,
    load_instructions,
    resolve_selection,
)
from .contracts import (
    ToolSource,
    ToolSourceRegistry,
    TranscriptStoreContract,
    TurnHandlerContract,
    TurnStream,
)
from .exceptions import MalformedToolCallError, UnservableModelError
from .router import Router
from .types import (
    RoutedAgent,
    Step,
    ToolExecution,
    ToolSet,
    TurnInput,
    TurnOutput,
    TurnPrompt,
    TurnRequest,
    TurnScope,
    resolve_budget,
)

logger = logging.getLogger(__name__)

_UNREACHABLE_NOTE = (
    '\n\nUnavailable for this turn, and not because the agent lacks them: {missing}.'
    ' Any tool they would have offered is absent. Say what you could not reach rather'
    ' than answering as though you had checked it.'
)


def _instructions(agent: AgentConfig, unreachable: Sequence[str]) -> str:
    instructions = load_instructions(agent)
    if not unreachable:
        return instructions

    # NOTE: this moves the static cache breakpoint
    logger.warning('Telling the model it cannot reach %s', ', '.join(unreachable))
    return instructions + _UNREACHABLE_NOTE.format(missing=', '.join(unreachable))


class AnswerRecorder(TurnStream):
    stream: TurnStream
    fragments: list[str]

    def __init__(self, stream: TurnStream) -> None:
        self.stream = stream
        self.fragments = []

    @property
    def answer(self) -> str:
        return ''.join(self.fragments)

    @override
    async def on_routed(self, agent: str) -> None:
        await self.stream.on_routed(agent)

    @override
    async def on_text(self, fragment: str) -> None:
        self.fragments.append(fragment)
        await self.stream.on_text(fragment)

    @override
    async def on_tool(self, name: str, phase: ToolPhase) -> None:
        await self.stream.on_tool(name, phase)


class ToolCallBuilder:
    id: str
    name: str
    arguments: str

    def __init__(self) -> None:
        self.id = ''
        self.name = ''
        self.arguments = ''


class TurnDraft:
    fragments: list[str]
    calls: dict[int, ToolCallBuilder]

    def __init__(self) -> None:
        self.fragments = []
        self.calls = {}

    @property
    def text(self) -> str:
        return ''.join(self.fragments)

    def tool_calls(self) -> tuple[ToolCallSegment, ...]:
        segments = []
        for index, builder in sorted(self.calls.items()):
            if not builder.id:
                raise MalformedToolCallError(
                    f'tool call {builder.name or "<unnamed>"!r} at index {index} '
                    'arrived without a call id',
                )

            segments.append(
                ToolCallSegment(
                    id=builder.id,
                    name=builder.name,
                    arguments=builder.arguments,
                ),
            )

        return tuple(segments)


def _prompt_segments(prompt: TurnPrompt) -> Iterator[Segment]:
    if prompt.text:
        yield TextSegment(text=prompt.text)

    for attachment in prompt.attachments:
        if attachment.kind == 'image':
            yield ImageSegment(
                media_type=attachment.mime,
                data=b64encode(attachment.data).decode('ascii'),
            )
        else:
            yield TextSegment(text=attachment.transcript or attachment.hint)


def _build_messages(turn: TurnInput) -> Iterator[Message]:
    yield Message(
        role='system',
        segments=(TextSegment(text=turn.instructions),),
        cacheable=True,
    )
    yield from turn.prior_turns
    dt = isoformat(turn.now)
    for prompt in turn.prompts:
        if segments := tuple(_prompt_segments(prompt)):
            header = TextSegment(text=f'[{dt}] ({turn.channel}) {prompt.actor}:\n\n')
            yield Message(role='user', segments=(header, *segments))


def _assistant_message(text: str, tool_calls: tuple[ToolCallSegment, ...]) -> Message:
    segments = (TextSegment(text=text), *tool_calls) if text else tool_calls
    return Message(role='assistant', segments=segments)


def _tool_result_messages(results: tuple[ToolResultSegment, ...]) -> Iterator[Message]:
    return (Message(role='tool', segments=(result,)) for result in results)


def _rolling_breakpoint(history: Sequence[Message]) -> tuple[Message, ...]:
    if not history:
        return ()

    return (*history[:-1], history[-1].model_copy(update={'cacheable': True}))


def _turn_id() -> str:
    # NOTE: whoever opens the turn owns its id -- a channel binds one before the reply
    # boundary it needs the id for, so minting here would give that turn a second name
    bound = get_contextvars().get('turn_id')
    return bound if isinstance(bound, str) else uuid4().hex


async def _execute(
    tools: ToolSet,
    call: ToolCallSegment,
    tool_timeout: float,
    clock: ClockContract,
    sink: TurnStream,
) -> tuple[ToolResultSegment, ToolExecution]:
    timer = clock.timer()
    await sink.on_tool(call.name, 'started')
    try:
        async with asyncio.timeout(tool_timeout):
            result = await tools.execute(call)
    except TimeoutError:
        logger.warning('Tool %s timed out after %.1fs', call.name, tool_timeout)
        result = ToolResultSegment(
            tool_call_id=call.id,
            content=f'tool {call.name!r} timed out',
            error=True,
        )

    record = ToolExecution(
        name=call.name,
        duration=timer.elapsed().total_seconds(),
        ok=not result.error,
    )
    logger.info(
        'Tool %s finished in %.3fs (ok=%s)',
        record.name,
        record.duration,
        record.ok,
    )
    await sink.on_tool(record.name, 'finished' if record.ok else 'failed')
    return result, record


async def _execute_calls(
    tools: ToolSet,
    calls: tuple[ToolCallSegment, ...],
    tool_timeout: float,
    clock: ClockContract,
    sink: TurnStream,
) -> tuple[tuple[ToolResultSegment, ...], Step]:
    tasks = (_execute(tools, call, tool_timeout, clock, sink) for call in calls)
    results, records = cast(
        tuple[tuple[ToolResultSegment, ...], tuple[ToolExecution, ...]],
        zip(*await asyncio.gather(*tasks), strict=True),
    )
    return results, Step(tools=records)


async def _consume(
    chunks: AsyncIterator[CompletionChunk],
    sink: TurnStream,
    draft: TurnDraft,
) -> None:
    async for chunk in chunks:
        match chunk:
            case TextChunk(text=fragment):
                draft.fragments.append(fragment)
                await sink.on_text(fragment)
            case ToolCallChunk(index=index, id=id_, name=name, arguments=arguments):
                builder = draft.calls.setdefault(index, ToolCallBuilder())
                if id_ is not None:
                    builder.id = id_
                if name is not None:
                    builder.name = name
                builder.arguments += arguments
            case ThinkingChunk():
                pass
            case FinishChunk():
                pass


def _affordable(deadline: float, minimum: float) -> bool:
    return deadline - asyncio.get_running_loop().time() >= minimum


class AgentRunner:
    llm: LLMClient
    tools: ToolSet
    clock: ClockContract

    def __init__(self, llm: LLMClient, tools: ToolSet, clock: ClockContract) -> None:
        self.llm = llm
        self.tools = tools
        self.clock = clock

    async def run(self, turn: TurnInput, sink: TurnStream) -> TurnOutput:
        with bound_contextvars(turn_id=_turn_id()):
            return await self._process(turn, sink)

    async def _process(self, turn: TurnInput, sink: TurnStream) -> TurnOutput:
        budget = turn.budget
        history = list(_build_messages(turn))
        steps = []
        text = ''
        deadline = asyncio.get_running_loop().time() + budget.turn_timeout

        for iteration in range(budget.max_iterations):
            # NOTE: the first iteration always opens -- a turn that never reached the
            # model has nothing to hand back, however little budget it was given
            if iteration and not _affordable(deadline, budget.min_iteration_budget):
                logger.warning(
                    'Stopping after %d iterations with under %.1fs of'
                    ' turn_timeout=%.1fs left',
                    iteration,
                    budget.min_iteration_budget,
                    budget.turn_timeout,
                )
                return TurnOutput(
                    text=text,
                    ending='timeout',
                    steps=tuple(steps),
                )

            draft = TurnDraft()
            try:
                async with asyncio.timeout_at(deadline):
                    request = CompletionRequest(
                        model=turn.model,
                        messages=_rolling_breakpoint(history),
                        tools=self.tools.schemas(),
                        max_completion_tokens=turn.max_completion_tokens,
                    )
                    await _consume(self.llm.stream(request), sink, draft)
                    tool_calls = draft.tool_calls()

                    if not tool_calls:
                        logger.info(
                            'Turn completed after %d iterations',
                            iteration + 1,
                        )
                        return TurnOutput(text=draft.text, steps=(*steps, Step()))

                    logger.info(
                        'Iteration %d executing %d tool calls',
                        iteration,
                        len(tool_calls),
                    )
                    results, step = await _execute_calls(
                        self.tools,
                        tool_calls,
                        budget.tool_timeout,
                        self.clock,
                        sink,
                    )
            except TimeoutError:
                logger.warning(
                    'Turn exceeded turn_timeout=%.1fs; delivering what it had',
                    budget.turn_timeout,
                )
                return TurnOutput(
                    text=draft.text or text,
                    ending='timeout',
                    steps=tuple(steps),
                )

            text = draft.text
            steps.append(step)
            history.append(_assistant_message(text, tool_calls))
            history.extend(_tool_result_messages(results))

        logger.warning(
            'Turn hit max_iterations=%d without finishing',
            budget.max_iterations,
        )
        return TurnOutput(text=text, ending='exhausted', steps=tuple(steps))


class TurnHandler(TurnHandlerContract):
    clock: ClockContract
    llm: LLMContract
    router: Router
    sources: ToolSourceRegistry
    store: TranscriptStoreContract
    defaults: DefaultsConfig
    default_max_completion_tokens: int | None

    def __init__(
        self,
        llm: LLMContract,
        router: Router,
        sources: ToolSourceRegistry,
        store: TranscriptStoreContract,
        clock: ClockContract,
        defaults: DefaultsConfig | None = None,
        default_max_completion_tokens: int | None = None,
    ) -> None:
        self.clock = clock
        self.defaults = defaults or DefaultsConfig()
        self.default_max_completion_tokens = default_max_completion_tokens
        self.llm = llm
        self.router = router
        self.sources = sources
        self.store = store

    @override
    async def handle(self, request: TurnRequest, reply: ChannelReplyContract) -> None:
        if not request.prompts:
            logger.info('Nothing to answer for chat %s', request.chat)
            return

        agent = await self.router.resolve(
            request.policy,
            request.chat,
            request.prompts[-1].text,
        )
        await reply.on_routed(agent.name)
        ref = resolve_selection(
            agent.config.provider,
            agent.config.model,
            self.defaults,
        )
        client = self.llm.client_for(ref)
        if client is None:
            raise UnservableModelError(
                f'model {ref.model!r} has no configured provider',
            )

        await self._run(request, reply, agent, ref, client)
        # NOTE: after the turn's transaction, and never a reason to fail a delivered
        # answer -- compaction is housekeeping the next turn benefits from
        try:
            await self.store.maintain(request.channel, request.chat, agent.name, ref)
        except Exception:
            logger.exception('Could not compact the conversation for %s', request.chat)

    async def _run(
        self,
        request: TurnRequest,
        reply: ChannelReplyContract,
        agent: RoutedAgent,
        ref: ModelRef,
        client: LLMClient,
    ) -> None:
        history = await self.store.history(request.channel, request.chat, agent.name)
        witness = AnswerRecorder(reply)
        try:
            scope = _turn_scope(request, agent.name)
            tools, unreachable = await build_tool_set(
                self.sources.sources(),
                agent.config,
                scope,
                request.denied_tools,
            )
            runner = AgentRunner(client, tools, self.clock)
            output = await runner.run(
                TurnInput(
                    instructions=_instructions(agent.config, unreachable),
                    prompts=request.prompts,
                    model=ref.model,
                    now=self.clock.now(),
                    channel=request.channel,
                    prior_turns=history,
                    max_completion_tokens=(
                        request.max_completion_tokens
                        or agent.config.max_completion_tokens
                        or self.default_max_completion_tokens
                    ),
                    budget=resolve_budget(agent.config),
                ),
                witness,
            )
        except Exception:
            await self._salvage(request, agent.name, witness.answer)
            raise

        if not output.completed:
            logger.info('Turn ended %s for chat %s', output.ending, request.chat)
            await reply.unfinished(output.ending)

        await reply.finish()
        await self.store.record(
            request.channel,
            request.chat,
            agent.name,
            request.prompts,
            output.text,
        )

    async def _salvage(self, request: TurnRequest, agent: str, answer: str) -> None:
        try:
            await self.store.record(
                request.channel,
                request.chat,
                agent,
                request.prompts,
                answer,
            )
        except Exception:
            logger.exception('Could not record the failed turn for %s', request.chat)


async def build_tool_set(
    sources: Sequence[ToolSource],
    agent: AgentConfig,
    scope: TurnScope,
    denied: frozenset[str] = frozenset(),
) -> tuple[ToolSet, tuple[str, ...]]:
    tools = ToolSet()
    unreachable = []
    for source in sources:
        harvest = await source.collect(agent, scope)
        unreachable.extend(harvest.unreachable)
        for name, tool in harvest.tools:
            if name in denied:
                logger.info('Tool %s is denied for this turn', name)
                continue

            tools.add(name, tool)

    return tools, tuple(unreachable)


def _turn_scope(request: TurnRequest, agent: str) -> TurnScope:
    prompt = request.prompts[-1]
    accounts = {item.account for item in request.prompts}
    account = prompt.account
    handle = prompt.handle
    if len(accounts) > 1:
        logger.info('Withholding account-bound tools: the turn mixes accounts')
        account = None
        handle = None
    return TurnScope(
        actor=prompt.actor,
        agent=agent,
        channel=request.channel,
        chat=request.chat,
        account=account,
        handle=handle,
    )
