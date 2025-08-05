from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.archive_request import ArchiveRequest
from ...models.exception import Exception_
from ...types import Response


def _get_kwargs(
    emil_id: str,
    *,
    body: ArchiveRequest,
    ocp_apim_subscription_key: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["Ocp-Apim-Subscription-Key"] = ocp_apim_subscription_key

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/bundle/{emil_id}/download",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(*, client: Union[AuthenticatedClient, Client], response: httpx.Response) -> Optional[Exception_]:
    if response.status_code == 400:
        response_400 = Exception_.from_dict(response.json())

        return response_400
    if response.status_code == 403:
        response_403 = Exception_.from_dict(response.json())

        return response_403
    if response.status_code == 404:
        response_404 = Exception_.from_dict(response.json())

        return response_404
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(*, client: Union[AuthenticatedClient, Client], response: httpx.Response) -> Response[Exception_]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    emil_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: ArchiveRequest,
    ocp_apim_subscription_key: str,
) -> Response[Exception_]:
    """Download an EMIL Product historical archive.

     Download an EMIL Product bundle archive by generating a JSON request with a single 'docIds' element
    containing the document identifier you wish to download.

    Args:
        emil_id (str):
        ocp_apim_subscription_key (str):
        body (ArchiveRequest): Represents a multi-document download request in JSON format.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Exception_]
    """

    kwargs = _get_kwargs(
        emil_id=emil_id,
        body=body,
        ocp_apim_subscription_key=ocp_apim_subscription_key,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    emil_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: ArchiveRequest,
    ocp_apim_subscription_key: str,
) -> Optional[Exception_]:
    """Download an EMIL Product historical archive.

     Download an EMIL Product bundle archive by generating a JSON request with a single 'docIds' element
    containing the document identifier you wish to download.

    Args:
        emil_id (str):
        ocp_apim_subscription_key (str):
        body (ArchiveRequest): Represents a multi-document download request in JSON format.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Exception_
    """

    return sync_detailed(
        emil_id=emil_id,
        client=client,
        body=body,
        ocp_apim_subscription_key=ocp_apim_subscription_key,
    ).parsed


async def asyncio_detailed(
    emil_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: ArchiveRequest,
    ocp_apim_subscription_key: str,
) -> Response[Exception_]:
    """Download an EMIL Product historical archive.

     Download an EMIL Product bundle archive by generating a JSON request with a single 'docIds' element
    containing the document identifier you wish to download.

    Args:
        emil_id (str):
        ocp_apim_subscription_key (str):
        body (ArchiveRequest): Represents a multi-document download request in JSON format.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Exception_]
    """

    kwargs = _get_kwargs(
        emil_id=emil_id,
        body=body,
        ocp_apim_subscription_key=ocp_apim_subscription_key,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    emil_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: ArchiveRequest,
    ocp_apim_subscription_key: str,
) -> Optional[Exception_]:
    """Download an EMIL Product historical archive.

     Download an EMIL Product bundle archive by generating a JSON request with a single 'docIds' element
    containing the document identifier you wish to download.

    Args:
        emil_id (str):
        ocp_apim_subscription_key (str):
        body (ArchiveRequest): Represents a multi-document download request in JSON format.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Exception_
    """

    return (
        await asyncio_detailed(
            emil_id=emil_id,
            client=client,
            body=body,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
