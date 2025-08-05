from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.exception import Exception_
from ...models.product_history import ProductHistory
from ...types import Response


def _get_kwargs(
    emil_id: str,
    *,
    ocp_apim_subscription_key: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["Ocp-Apim-Subscription-Key"] = ocp_apim_subscription_key

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/archive/{emil_id}",
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Exception_, ProductHistory]]:
    if response.status_code == 200:
        response_200 = ProductHistory.from_dict(response.json())

        return response_200
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


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[Exception_, ProductHistory]]:
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
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, ProductHistory]]:
    """Available report archives for a specified EMIL product.

     Available report archives for a specified EMIL product.

    Args:
        emil_id (str):
        ocp_apim_subscription_key (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Exception_, ProductHistory]]
    """

    kwargs = _get_kwargs(
        emil_id=emil_id,
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
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, ProductHistory]]:
    """Available report archives for a specified EMIL product.

     Available report archives for a specified EMIL product.

    Args:
        emil_id (str):
        ocp_apim_subscription_key (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Exception_, ProductHistory]
    """

    return sync_detailed(
        emil_id=emil_id,
        client=client,
        ocp_apim_subscription_key=ocp_apim_subscription_key,
    ).parsed


async def asyncio_detailed(
    emil_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, ProductHistory]]:
    """Available report archives for a specified EMIL product.

     Available report archives for a specified EMIL product.

    Args:
        emil_id (str):
        ocp_apim_subscription_key (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Exception_, ProductHistory]]
    """

    kwargs = _get_kwargs(
        emil_id=emil_id,
        ocp_apim_subscription_key=ocp_apim_subscription_key,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    emil_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, ProductHistory]]:
    """Available report archives for a specified EMIL product.

     Available report archives for a specified EMIL product.

    Args:
        emil_id (str):
        ocp_apim_subscription_key (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Exception_, ProductHistory]
    """

    return (
        await asyncio_detailed(
            emil_id=emil_id,
            client=client,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
