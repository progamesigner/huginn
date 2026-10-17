from typing import Annotated

from pydantic import ConfigDict, Field, SecretStr

from app.contracts.auth import ApiClient
from app.core.dependencies import Authenticated, Provide
from app.core.exceptions import InvalidRequestError
from app.core.schema import AppRequest, AppResponse
from app.features.identities.contracts import (
    IdentityLinkingServiceContract,
    IdentityManagementServiceContract,
    IdentityResolutionServiceContract,
)
from app.features.identities.entities import LinkedPrincipal
from lib.result import Err
from lib.router import Depends, Router

router = Router()


class ClaimRequest(AppRequest):
    model_config = ConfigDict(extra='forbid', frozen=True)
    code: SecretStr
    handle: str
    display_name: Annotated[str | None, Field(max_length=128)] = None


class LinkRequest(AppRequest):
    model_config = ConfigDict(extra='forbid', frozen=True)
    code: SecretStr


class TargetRequest(AppRequest):
    target: str


class DisplayNameRequest(AppRequest):
    display_name: Annotated[str, Field(min_length=1, max_length=128)]


class IdentityResponse(AppResponse):
    linked: bool
    handle: str | None = None
    account: str | None = None
    display_name: str | None = None


class InviteResponse(AppResponse):
    code: str
    expires_at: str


def asserted(client: ApiClient) -> str:
    if client.asserted_user is None:
        raise InvalidRequestError('X-Huginn-User is required')
    return client.asserted_user


def describe(principal: LinkedPrincipal) -> IdentityResponse:
    return IdentityResponse(
        linked=True,
        handle=principal.handle,
        account=str(principal.user_id),
        display_name=principal.display_name,
    )


@router.get('/v1/identity')
async def identity(
    client: Annotated[ApiClient, Depends(Authenticated())],
    resolver: Annotated[
        IdentityResolutionServiceContract,
        Depends(Provide(IdentityResolutionServiceContract)),
    ],
) -> IdentityResponse:
    principal = await resolver.resolve('http', client.key, asserted(client))
    return (
        describe(principal)
        if isinstance(principal, LinkedPrincipal)
        else IdentityResponse(linked=False)
    )


@router.post('/v1/identity/claim')
async def claim(
    ask: ClaimRequest,
    client: Annotated[ApiClient, Depends(Authenticated())],
    management: Annotated[
        IdentityManagementServiceContract,
        Depends(Provide(IdentityManagementServiceContract)),
    ],
) -> IdentityResponse:
    result = await management.claim(
        'http',
        client.key,
        asserted(client),
        ask.handle,
        credential=ask.code.get_secret_value(),
        display_name=ask.display_name,
    )
    if isinstance(result, Err):
        raise InvalidRequestError(str(result.error))
    return describe(result.value)


@router.post('/v1/identity/link')
async def link(
    ask: LinkRequest,
    client: Annotated[ApiClient, Depends(Authenticated())],
    linker: Annotated[
        IdentityLinkingServiceContract,
        Depends(Provide(IdentityLinkingServiceContract)),
    ],
) -> IdentityResponse:
    result = await linker.link(
        ask.code.get_secret_value(),
        'http',
        client.key,
        asserted(client),
    )
    if isinstance(result, Err):
        raise InvalidRequestError(str(result.error))
    return describe(result.value)


@router.post('/v1/identity/invite')
async def invite(
    ask: TargetRequest,
    client: Annotated[ApiClient, Depends(Authenticated())],
    management: Annotated[
        IdentityManagementServiceContract,
        Depends(Provide(IdentityManagementServiceContract)),
    ],
) -> InviteResponse:
    try:
        issued = await management.invite(
            'http',
            client.key,
            asserted(client),
            ask.target,
        )
    except ValueError as error:
        raise InvalidRequestError(str(error)) from error
    return InviteResponse(
        code=issued.short_code,
        expires_at=issued.expires_at.isoformat(),
    )


@router.patch('/v1/identity')
async def display_name(
    ask: DisplayNameRequest,
    client: Annotated[ApiClient, Depends(Authenticated())],
    management: Annotated[
        IdentityManagementServiceContract,
        Depends(Provide(IdentityManagementServiceContract)),
    ],
    resolver: Annotated[
        IdentityResolutionServiceContract,
        Depends(Provide(IdentityResolutionServiceContract)),
    ],
) -> IdentityResponse:
    external = asserted(client)
    try:
        await management.display_name('http', client.key, external, ask.display_name)
    except ValueError as error:
        raise InvalidRequestError(str(error)) from error
    principal = await resolver.resolve('http', client.key, external)
    if not isinstance(principal, LinkedPrincipal):
        raise InvalidRequestError('identity is no longer linked')
    return describe(principal)
