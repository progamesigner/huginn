import asyncio
from abc import ABC, abstractmethod
from urllib.parse import urlsplit

from httpx import HTTPError, Response, TimeoutException
from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from lib.decision.client import DecisionProviderConfig
from lib.decision.contracts import HTTPConnector
from lib.decision.exceptions import (
    DecisionHTTPError,
    DecisionResponseError,
    DecisionTimeoutError,
    DecisionTransportError,
)
from lib.decision.types import DecisionRequest, DecisionResult

_ERROR_TEXT_LIMIT = 200


class WirePayload(BaseModel):
    # NOTE: tolerant of fields the vendor adds, and never quoting what it sent back
    model_config = ConfigDict(extra='ignore', hide_input_in_errors=True)


class HTTPDecisionAdapter(ABC):
    http: HTTPConnector
    provider: DecisionProviderConfig

    def __init__(
        self,
        provider: DecisionProviderConfig,
        http: HTTPConnector,
    ) -> None:
        self.provider = provider
        self.http = http

    async def decide(self, request: DecisionRequest) -> DecisionResult:
        headers = {
            'Authorization': f'Bearer {self.provider.api_key.get_secret_value()}',
        }
        # NOTE: httpx gives a base url a trailing slash, so the endpoint is split
        # rather than posted to as a whole
        endpoint = urlsplit(self.provider.endpoint)
        try:
            async with (
                asyncio.timeout(self.provider.timeout),
                self.http.connect(
                    f'{endpoint.scheme}://{endpoint.netloc}',
                    headers,
                ) as client,
            ):
                response = await client.post(
                    endpoint.path or '/',
                    json=self.encode(request),
                    timeout=self.provider.timeout,
                )
        except (TimeoutException, TimeoutError) as error:
            raise DecisionTimeoutError(
                f'{self.provider.name} did not answer within {self.provider.timeout}s',
            ) from error
        except HTTPError as error:
            raise DecisionTransportError(
                f'{self.provider.name} could not be reached: {type(error).__name__}',
            ) from error

        if response.is_error:
            raise DecisionHTTPError(response.status_code, _error_message(response))

        try:
            payload = response.json()
        except ValueError as error:
            raise DecisionResponseError(
                f'{self.provider.name} answered with a body that is not JSON',
            ) from error

        try:
            result = self.decode(request, payload)
        except ValidationError as error:
            raise DecisionResponseError(
                f'{self.provider.name} answered in an unexpected shape: '
                f'{error.error_count()} problem(s)',
            ) from error

        return checked(request, result)

    @abstractmethod
    def encode(self, request: DecisionRequest) -> dict[str, JsonValue]: ...

    @abstractmethod
    def decode(
        self,
        request: DecisionRequest,
        payload: JsonValue,
    ) -> DecisionResult: ...


def checked(request: DecisionRequest, result: DecisionResult) -> DecisionResult:
    unexpected = set(result.answers).difference(request.questions)
    if unexpected:
        raise DecisionResponseError(
            f'questions {sorted(unexpected)!r} were not requested',
        )

    for name, question in request.questions.items():
        answer = result.answers.get(name)
        if answer is None:
            raise DecisionResponseError(f'question {name!r} was not answered')
        if answer.choice not in question.criteria:
            raise DecisionResponseError(
                f'question {name!r} was answered with an option it does not offer',
            )
        if not set(answer.probabilities).issubset(question.criteria):
            raise DecisionResponseError(
                f'question {name!r} weighed options it does not offer',
            )

    return result


def _error_message(response: Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:_ERROR_TEXT_LIMIT] or response.reason_phrase

    # NOTE: TypeSafe says {"detail": str | {"message"}}; OpenAI and the gateways in
    # front of either say {"error": {"message"}}
    match payload:
        case {'detail': str(message)}:
            return message[:_ERROR_TEXT_LIMIT]
        case {'detail': {'message': str(message)}}:
            return message[:_ERROR_TEXT_LIMIT]
        case {'error': {'message': str(message)}}:
            return message[:_ERROR_TEXT_LIMIT]
        case _:
            return response.reason_phrase
