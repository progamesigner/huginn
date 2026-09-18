import pytest
from pydantic import BaseModel, ValidationError, create_model

from lib.llm import (
    CompletionChunk,
    CompletionRequest,
    DocumentSegment,
    FinishChunk,
    ImageSegment,
    Message,
    OutputSchema,
    Segment,
    TextChunk,
    TextSegment,
    ThinkingChunk,
    ThinkingSegment,
    ToolCallChunk,
    ToolCallSegment,
    ToolResultSegment,
)


def test_tool_call_segment_holds_raw_json_arguments() -> None:
    segment = ToolCallSegment(
        id='call_1',
        name='search',
        arguments='{"query": "huginn"}',
    )

    assert segment.id == 'call_1'
    assert segment.name == 'search'
    assert segment.arguments == '{"query": "huginn"}'


def test_tool_call_segment_rejects_non_string_id_under_strict_mode() -> None:
    with pytest.raises(ValidationError):
        ToolCallSegment.model_validate({'id': 1, 'name': 'search', 'arguments': '{}'})


def test_tool_call_segment_is_frozen() -> None:
    segment = ToolCallSegment(id='call_1', name='search', arguments='{}')
    field = 'name'

    with pytest.raises(ValidationError):
        setattr(segment, field, 'other')


def test_tool_result_segment_defaults_to_success() -> None:
    segment = ToolResultSegment(tool_call_id='call_1', content='ok')

    assert segment.error is False


def test_thinking_segment_carries_opaque_data_for_roundtrip() -> None:
    data = {'type': 'thinking', 'thinking': 'reasoning...', 'signature': 'abc'}
    segment = ThinkingSegment(data=data)

    assert segment.data == data


def test_segment_union_matches_exhaustively() -> None:
    def describe(segment: Segment) -> str:
        match segment:
            case TextSegment(text=text):
                return f'text:{text}'
            case ImageSegment(media_type=media_type):
                return f'image:{media_type}'
            case DocumentSegment(media_type=media_type):
                return f'document:{media_type}'
            case ToolCallSegment(name=name):
                return f'tool_call:{name}'
            case ToolResultSegment(tool_call_id=tool_call_id):
                return f'tool_result:{tool_call_id}'
            case ThinkingSegment(data=data):
                return f'thinking:{data}'

    assert describe(TextSegment(text='hi')) == 'text:hi'
    assert describe(ToolCallSegment(id='call_1', name='search', arguments='{}')) == (
        'tool_call:search'
    )
    assert describe(ToolResultSegment(tool_call_id='call_1', content='ok')) == (
        'tool_result:call_1'
    )
    assert describe(ThinkingSegment(data='...')) == 'thinking:...'
    assert describe(ImageSegment(media_type='image/jpeg', data='YQ==')) == (
        'image:image/jpeg'
    )
    assert describe(DocumentSegment(media_type='application/pdf', data='YQ==')) == (
        'document:application/pdf'
    )


def test_message_preserves_segment_order() -> None:
    call = ToolCallSegment(id='call_1', name='search', arguments='{}')
    message = Message(role='assistant', segments=(TextSegment(text='looking...'), call))

    assert message.segments == (TextSegment(text='looking...'), call)


def test_message_represents_tool_result_as_single_segment() -> None:
    result = ToolResultSegment(tool_call_id='call_1', content='ok')
    message = Message(role='tool', segments=(result,))

    assert message.segments == (result,)


def test_message_defaults_to_not_cacheable() -> None:
    message = Message(role='user', segments=(TextSegment(text='hi'),))

    assert message.cacheable is False


def test_message_rejects_lax_bool_coercion_under_strict_mode() -> None:
    with pytest.raises(ValidationError):
        Message.model_validate({'role': 'user', 'segments': [], 'cacheable': 1})


def test_completion_chunk_variants_match_exhaustively() -> None:
    def describe(chunk: CompletionChunk) -> str:
        match chunk:
            case TextChunk(text=text):
                return f'text:{text}'
            case ToolCallChunk(index=index, arguments=arguments):
                return f'tool_call:{index}:{arguments}'
            case ThinkingChunk(text=text):
                return f'thinking:{text}'
            case FinishChunk(reason=reason):
                return f'finish:{reason}'

    assert describe(TextChunk(text='hi')) == 'text:hi'
    assert describe(ToolCallChunk(index=0, arguments='{"q')) == 'tool_call:0:{"q'
    assert describe(ThinkingChunk(text='...')) == 'thinking:...'
    assert describe(FinishChunk(reason='stop')) == 'finish:stop'


def test_completion_request_defaults_to_no_tools_and_no_effort() -> None:
    request = CompletionRequest(
        model='gpt-5',
        messages=(Message(role='user', segments=(TextSegment(text='hi'),)),),
    )

    assert request.tools == ()
    assert request.effort == 'none'
    assert request.max_completion_tokens is None
    assert request.output_schema is None


def test_output_schema_rejects_name_outside_openai_charset() -> None:
    with pytest.raises(ValidationError):
        OutputSchema(name='has space', json_schema={'type': 'object'})

    with pytest.raises(ValidationError):
        OutputSchema(name='a' * 65, json_schema={'type': 'object'})


def test_output_schema_from_model_derives_name_from_title() -> None:
    class SummaryOutput(BaseModel):
        summary: str

    output_schema = OutputSchema.from_model(SummaryOutput)

    assert output_schema.name == 'SummaryOutput'
    assert output_schema.json_schema['type'] == 'object'


def test_output_schema_from_model_truncates_long_class_names_to_64_chars() -> None:
    model = create_model('A' * 100, summary=(str, ...))

    output_schema = OutputSchema.from_model(model)

    assert output_schema.name == 'A' * 64


def test_output_schema_from_model_marks_all_properties_required_even_with_defaults() -> (
    None
):
    class LocalSummaryOutput(BaseModel):
        summary: str
        key_facts: tuple[str, ...] = ()

    output_schema = OutputSchema.from_model(LocalSummaryOutput)

    assert output_schema.json_schema['required'] == ['summary', 'key_facts']
