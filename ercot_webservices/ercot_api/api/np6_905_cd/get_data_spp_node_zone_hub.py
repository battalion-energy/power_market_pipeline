from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.exception import Exception_
from ...models.report import Report
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    delivery_hour_from: Union[Unset, int] = UNSET,
    delivery_hour_to: Union[Unset, int] = UNSET,
    delivery_interval_from: Union[Unset, int] = UNSET,
    delivery_interval_to: Union[Unset, int] = UNSET,
    settlement_point: Union[Unset, str] = UNSET,
    settlement_point_type: Union[Unset, str] = UNSET,
    settlement_point_price_from: Union[Unset, float] = UNSET,
    settlement_point_price_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["Ocp-Apim-Subscription-Key"] = ocp_apim_subscription_key

    params: dict[str, Any] = {}

    params["deliveryDateFrom"] = delivery_date_from

    params["deliveryDateTo"] = delivery_date_to

    params["deliveryHourFrom"] = delivery_hour_from

    params["deliveryHourTo"] = delivery_hour_to

    params["deliveryIntervalFrom"] = delivery_interval_from

    params["deliveryIntervalTo"] = delivery_interval_to

    params["settlementPoint"] = settlement_point

    params["settlementPointType"] = settlement_point_type

    params["settlementPointPriceFrom"] = settlement_point_price_from

    params["settlementPointPriceTo"] = settlement_point_price_to

    params["DSTFlag"] = dst_flag

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np6-905-cd/spp_node_zone_hub",
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Exception_, Report]]:
    if response.status_code == 200:
        response_200 = Report.from_dict(response.json())

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
) -> Response[Union[Exception_, Report]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    delivery_hour_from: Union[Unset, int] = UNSET,
    delivery_hour_to: Union[Unset, int] = UNSET,
    delivery_interval_from: Union[Unset, int] = UNSET,
    delivery_interval_to: Union[Unset, int] = UNSET,
    settlement_point: Union[Unset, str] = UNSET,
    settlement_point_type: Union[Unset, str] = UNSET,
    settlement_point_price_from: Union[Unset, float] = UNSET,
    settlement_point_price_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """Settlement Point Prices at Resource Nodes, Hubs and Load Zones

     Settlement Point Prices at Resource Nodes, Hubs and Load Zones

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        delivery_hour_from (Union[Unset, int]):
        delivery_hour_to (Union[Unset, int]):
        delivery_interval_from (Union[Unset, int]):
        delivery_interval_to (Union[Unset, int]):
        settlement_point (Union[Unset, str]):
        settlement_point_type (Union[Unset, str]):
        settlement_point_price_from (Union[Unset, float]):
        settlement_point_price_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
        page (Union[Unset, int]):
        size (Union[Unset, int]):
        sort (Union[Unset, str]):
        dir_ (Union[Unset, str]):
        ocp_apim_subscription_key (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Exception_, Report]]
    """

    kwargs = _get_kwargs(
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        delivery_hour_from=delivery_hour_from,
        delivery_hour_to=delivery_hour_to,
        delivery_interval_from=delivery_interval_from,
        delivery_interval_to=delivery_interval_to,
        settlement_point=settlement_point,
        settlement_point_type=settlement_point_type,
        settlement_point_price_from=settlement_point_price_from,
        settlement_point_price_to=settlement_point_price_to,
        dst_flag=dst_flag,
        page=page,
        size=size,
        sort=sort,
        dir_=dir_,
        ocp_apim_subscription_key=ocp_apim_subscription_key,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    delivery_hour_from: Union[Unset, int] = UNSET,
    delivery_hour_to: Union[Unset, int] = UNSET,
    delivery_interval_from: Union[Unset, int] = UNSET,
    delivery_interval_to: Union[Unset, int] = UNSET,
    settlement_point: Union[Unset, str] = UNSET,
    settlement_point_type: Union[Unset, str] = UNSET,
    settlement_point_price_from: Union[Unset, float] = UNSET,
    settlement_point_price_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """Settlement Point Prices at Resource Nodes, Hubs and Load Zones

     Settlement Point Prices at Resource Nodes, Hubs and Load Zones

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        delivery_hour_from (Union[Unset, int]):
        delivery_hour_to (Union[Unset, int]):
        delivery_interval_from (Union[Unset, int]):
        delivery_interval_to (Union[Unset, int]):
        settlement_point (Union[Unset, str]):
        settlement_point_type (Union[Unset, str]):
        settlement_point_price_from (Union[Unset, float]):
        settlement_point_price_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
        page (Union[Unset, int]):
        size (Union[Unset, int]):
        sort (Union[Unset, str]):
        dir_ (Union[Unset, str]):
        ocp_apim_subscription_key (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Exception_, Report]
    """

    return sync_detailed(
        client=client,
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        delivery_hour_from=delivery_hour_from,
        delivery_hour_to=delivery_hour_to,
        delivery_interval_from=delivery_interval_from,
        delivery_interval_to=delivery_interval_to,
        settlement_point=settlement_point,
        settlement_point_type=settlement_point_type,
        settlement_point_price_from=settlement_point_price_from,
        settlement_point_price_to=settlement_point_price_to,
        dst_flag=dst_flag,
        page=page,
        size=size,
        sort=sort,
        dir_=dir_,
        ocp_apim_subscription_key=ocp_apim_subscription_key,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    delivery_hour_from: Union[Unset, int] = UNSET,
    delivery_hour_to: Union[Unset, int] = UNSET,
    delivery_interval_from: Union[Unset, int] = UNSET,
    delivery_interval_to: Union[Unset, int] = UNSET,
    settlement_point: Union[Unset, str] = UNSET,
    settlement_point_type: Union[Unset, str] = UNSET,
    settlement_point_price_from: Union[Unset, float] = UNSET,
    settlement_point_price_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """Settlement Point Prices at Resource Nodes, Hubs and Load Zones

     Settlement Point Prices at Resource Nodes, Hubs and Load Zones

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        delivery_hour_from (Union[Unset, int]):
        delivery_hour_to (Union[Unset, int]):
        delivery_interval_from (Union[Unset, int]):
        delivery_interval_to (Union[Unset, int]):
        settlement_point (Union[Unset, str]):
        settlement_point_type (Union[Unset, str]):
        settlement_point_price_from (Union[Unset, float]):
        settlement_point_price_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
        page (Union[Unset, int]):
        size (Union[Unset, int]):
        sort (Union[Unset, str]):
        dir_ (Union[Unset, str]):
        ocp_apim_subscription_key (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Exception_, Report]]
    """

    kwargs = _get_kwargs(
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        delivery_hour_from=delivery_hour_from,
        delivery_hour_to=delivery_hour_to,
        delivery_interval_from=delivery_interval_from,
        delivery_interval_to=delivery_interval_to,
        settlement_point=settlement_point,
        settlement_point_type=settlement_point_type,
        settlement_point_price_from=settlement_point_price_from,
        settlement_point_price_to=settlement_point_price_to,
        dst_flag=dst_flag,
        page=page,
        size=size,
        sort=sort,
        dir_=dir_,
        ocp_apim_subscription_key=ocp_apim_subscription_key,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    delivery_hour_from: Union[Unset, int] = UNSET,
    delivery_hour_to: Union[Unset, int] = UNSET,
    delivery_interval_from: Union[Unset, int] = UNSET,
    delivery_interval_to: Union[Unset, int] = UNSET,
    settlement_point: Union[Unset, str] = UNSET,
    settlement_point_type: Union[Unset, str] = UNSET,
    settlement_point_price_from: Union[Unset, float] = UNSET,
    settlement_point_price_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """Settlement Point Prices at Resource Nodes, Hubs and Load Zones

     Settlement Point Prices at Resource Nodes, Hubs and Load Zones

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        delivery_hour_from (Union[Unset, int]):
        delivery_hour_to (Union[Unset, int]):
        delivery_interval_from (Union[Unset, int]):
        delivery_interval_to (Union[Unset, int]):
        settlement_point (Union[Unset, str]):
        settlement_point_type (Union[Unset, str]):
        settlement_point_price_from (Union[Unset, float]):
        settlement_point_price_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
        page (Union[Unset, int]):
        size (Union[Unset, int]):
        sort (Union[Unset, str]):
        dir_ (Union[Unset, str]):
        ocp_apim_subscription_key (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Exception_, Report]
    """

    return (
        await asyncio_detailed(
            client=client,
            delivery_date_from=delivery_date_from,
            delivery_date_to=delivery_date_to,
            delivery_hour_from=delivery_hour_from,
            delivery_hour_to=delivery_hour_to,
            delivery_interval_from=delivery_interval_from,
            delivery_interval_to=delivery_interval_to,
            settlement_point=settlement_point,
            settlement_point_type=settlement_point_type,
            settlement_point_price_from=settlement_point_price_from,
            settlement_point_price_to=settlement_point_price_to,
            dst_flag=dst_flag,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
