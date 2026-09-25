import json

from pydantic import JsonValue

from .types import RoutingContext, TurnPrompt

# NOTE: routing sees text and attachment hints only, never media bytes, tool results
# or reasoning
ROUTING_BUDGET = 8_000
USER_BUDGET = 2_000
ANSWER_BUDGET = 600


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 1)] + '…'


def prompt_text(prompt: TurnPrompt) -> str:
    parts = [prompt.text]
    for attachment in prompt.attachments:
        if attachment.transcript:
            parts.append(attachment.transcript)
        parts.append(f'[{attachment.kind}: {attachment.name or attachment.hint}]')
    return '\n'.join(part for part in parts if part)


def _encoded(state: dict[str, JsonValue]) -> str:
    return json.dumps(state, ensure_ascii=False, separators=(',', ':'))


def routing_state(
    context: RoutingContext,
    budget: int = ROUTING_BUDGET,
) -> dict[str, JsonValue]:
    if budget < 128:
        raise ValueError('routing budget must be at least 128 characters')
    previous = list[JsonValue](
        [
            {
                'handled_by': turn.agent,
                'user': [clip(text, USER_BUDGET) for text in turn.user],
                'assistant': clip(turn.answer, ANSWER_BUDGET),
                'attachments': [clip(hint, 200) for hint in turn.attachments],
            }
            for turn in context.turns
        ],
    )
    current = list[JsonValue](prompt_text(prompt) for prompt in context.current)
    state = dict[str, JsonValue](
        {
            'summary': clip(context.summary, USER_BUDGET) if context.summary else None,
            'previous_turns': previous,
            'current': current,
        },
    )
    while previous and len(_encoded(state)) > budget:
        previous.pop(0)
    if len(_encoded(state)) > budget:
        state['summary'] = None
    if len(_encoded(state)) > budget:
        # NOTE: only an oversized current turn is flattened and truncated, and the
        # budget counts the serialised form, escaping and structure included
        text = '\n'.join(prompt_text(prompt) for prompt in context.current)
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            state['current'] = [clip(text, mid)]
            if len(_encoded(state)) <= budget:
                low = mid
            else:
                high = mid - 1
        state['current'] = [clip(text, low)]
    return state


def render_routing_state(context: RoutingContext, budget: int = ROUTING_BUDGET) -> str:
    # NOTE: a lone message with no history renders exactly as it did before
    if not context.turns and not context.summary and len(context.current) == 1:
        prompt = context.current[0]
        if not prompt.attachments:
            return prompt.text
    return _encoded(routing_state(context, budget))
