from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from itertools import repeat

import httpx2

from lib.http import HTTPClientManager

type Handler = Callable[
    [httpx2.Request],
    httpx2.Response | Awaitable[httpx2.Response],
]


@dataclass(frozen=True)
class Call:
    request: httpx2.Request
    response: httpx2.Response | None


class Calls(list[Call]):
    @property
    def last(self) -> Call:
        return self[-1]


@dataclass
class Route:
    method: str
    url: str
    prefix: bool
    calls: Calls = field(default_factory=Calls)
    answers: Iterator[httpx2.Response | Exception] | None = None
    handler: Handler | None = None
    response: httpx2.Response | None = None

    @property
    def called(self) -> bool:
        return bool(self.calls)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def mock(
        self,
        return_value: httpx2.Response | None = None,
        side_effect: Handler | None = None,
        responses: list[httpx2.Response | Exception] | None = None,
        error: Exception | None = None,
    ) -> Route:
        """Answer with a handler, a sequence of answers, one error, or one response."""
        self.response, self.handler, self.answers = return_value, side_effect, None
        if error is not None:
            self.answers = repeat(error)
        elif responses is not None:
            self.answers = iter(responses)
        return self

    def matches(self, request: httpx2.Request) -> bool:
        if request.method != self.method:
            return False
        url = str(request.url)
        if self.prefix:
            return url.startswith(self.url)
        # NOTE: a pattern without a query matches any query
        return url == self.url or (
            '?' not in self.url and url.split('?')[0] == self.url
        )

    async def answer(self, request: httpx2.Request) -> httpx2.Response:
        try:
            response = await self._answer(request)
        except BaseException:
            self.calls.append(Call(request, None))
            raise
        self.calls.append(Call(request, response))
        return response

    async def _answer(self, request: httpx2.Request) -> httpx2.Response:
        if self.handler is not None:
            result = self.handler(request)
            if isinstance(result, httpx2.Response):
                return result
            return await result
        if self.answers is not None:
            answer = next(self.answers, None)
            if answer is None:
                raise AssertionError(
                    f'no answer left for {request.method} {request.url}',
                )
            if isinstance(answer, httpx2.Response):
                return _copy(answer)
            # NOTE: a transport error names the request that raised it
            if isinstance(answer, httpx2.RequestError):
                answer.request = request
            raise answer
        if self.response is not None:
            return _copy(self.response)
        return httpx2.Response(200)


class MockHTTP:
    """Answers requests from registered routes; anything unregistered fails the test."""

    def __init__(self) -> None:
        self.routes: list[Route] = []
        self.transport = httpx2.MockTransport(self._dispatch)

    def manager(self, timeout: float | None = None) -> HTTPClientManager:
        if timeout is None:
            return HTTPClientManager(transport=self.transport)
        return HTTPClientManager(timeout=timeout, transport=self.transport)

    def get(self, url: str = '', *, url__startswith: str = '') -> Route:
        return self.route('GET', url, url__startswith)

    def post(self, url: str = '', *, url__startswith: str = '') -> Route:
        return self.route('POST', url, url__startswith)

    def route(self, method: str, url: str, prefix: str = '') -> Route:
        pattern, starts = (prefix, True) if prefix else (url, False)
        for route in self.routes:
            if (route.method, route.url, route.prefix) == (method, pattern, starts):
                return route
        route = Route(method, pattern, starts)
        self.routes.append(route)
        return route

    async def _dispatch(self, request: httpx2.Request) -> httpx2.Response:
        for route in self.routes:
            if route.matches(request):
                return await route.answer(request)
        raise AssertionError(f'unexpected request: {request.method} {request.url}')


def _copy(response: httpx2.Response) -> httpx2.Response:
    # NOTE: a response given as a stream is the test's to observe, so it is passed on
    if not isinstance(response.stream, httpx2.ByteStream):
        return response
    # NOTE: each request gets a fresh response; a shared one would be read once
    return httpx2.Response(
        response.status_code,
        headers=response.headers,
        content=response.content,
    )
