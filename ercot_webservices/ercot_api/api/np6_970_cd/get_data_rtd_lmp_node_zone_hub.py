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
    rtd_timestamp_from: Union[Unset, str] = UNSET,
    rtd_timestamp_to: Union[Unset, str] = UNSET,
    repeat_hour_flag: Union[Unset, bool] = UNSET,
    interval_id_from: Union[Unset, int] = UNSET,
    interval_id_to: Union[Unset, int] = UNSET,
    interval_ending_from: Union[Unset, str] = UNSET,
    interval_ending_to: Union[Unset, str] = UNSET,
    interval_repeat_hour_flag: Union[Unset, bool] = UNSET,
    settlement_point: Union[Unset, str] = UNSET,
    settlement_point_type: Union[Unset, str] = UNSET,
    lmp_from: Union[Unset, float] = UNSET,
    lmp_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["Ocp-Apim-Subscription-Key"] = ocp_apim_subscription_key

    params: dict[str, Any] = {}

    params["RTDTimestampFrom"] = rtd_timestamp_from

    params["RTDTimestampTo"] = rtd_timestamp_to

    params["repeatHourFlag"] = repeat_hour_flag

    params["intervalIdFrom"] = interval_id_from

    params["intervalIdTo"] = interval_id_to

    params["intervalEndingFrom"] = interval_ending_from

    params["intervalEndingTo"] = interval_ending_to

    params["intervalRepeatHourFlag"] = interval_repeat_hour_flag

    params["settlementPoint"] = settlement_point

    params["settlementPointType"] = settlement_point_type

    params["LMPFrom"] = lmp_from

    params["LMPTo"] = lmp_to

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np6-970-cd/rtd_lmp_node_zone_hub",
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
    rtd_timestamp_from: Union[Unset, str] = UNSET,
    rtd_timestamp_to: Union[Unset, str] = UNSET,
    repeat_hour_flag: Union[Unset, bool] = UNSET,
    interval_id_from: Union[Unset, int] = UNSET,
    interval_id_to: Union[Unset, int] = UNSET,
    interval_ending_from: Union[Unset, str] = UNSET,
    interval_ending_to: Union[Unset, str] = UNSET,
    interval_repeat_hour_flag: Union[Unset, bool] = UNSET,
    settlement_point: Union[Unset, str] = UNSET,
    settlement_point_type: Union[Unset, str] = UNSET,
    lmp_from: Union[Unset, float] = UNSET,
    lmp_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """RTD Indicative LMPs by Resource Nodes, Load Zones and Hubs

     RTD Indicative LMPs by Resource Nodes, Load Zones and Hubs

    Args:
        rtd_timestamp_from (Union[Unset, str]):
        rtd_timestamp_to (Union[Unset, str]):
        repeat_hour_flag (Union[Unset, bool]):
        interval_id_from (Union[Unset, int]):
        interval_id_to (Union[Unset, int]):
        interval_ending_from (Union[Unset, str]):
        interval_ending_to (Union[Unset, str]):
        interval_repeat_hour_flag (Union[Unset, bool]):
        settlement_point (Union[Unset, str]):
        settlement_point_type (Union[Unset, str]):
        lmp_from (Union[Unset, float]):
        lmp_to (Union[Unset, float]):
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
        rtd_timestamp_from=rtd_timestamp_from,
        rtd_timestamp_to=rtd_timestamp_to,
        repeat_hour_flag=repeat_hour_flag,
        interval_id_from=interval_id_from,
        interval_id_to=interval_id_to,
        interval_ending_from=interval_ending_from,
        interval_ending_to=interval_ending_to,
        interval_repeat_hour_flag=interval_repeat_hour_flag,
        settlement_point=settlement_point,
        settlement_point_type=settlement_point_type,
        lmp_from=lmp_from,
        lmp_to=lmp_to,
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
    rtd_timestamp_from: Union[Unset, str] = UNSET,
    rtd_timestamp_to: Union[Unset, str] = UNSET,
    repeat_hour_flag: Union[Unset, bool] = UNSET,
    interval_id_from: Union[Unset, int] = UNSET,
    interval_id_to: Union[Unset, int] = UNSET,
    interval_ending_from: Union[Unset, str] = UNSET,
    interval_ending_to: Union[Unset, str] = UNSET,
    interval_repeat_hour_flag: Union[Unset, bool] = UNSET,
    settlement_point: Union[Unset, str] = UNSET,
    settlement_point_type: Union[Unset, str] = UNSET,
    lmp_from: Union[Unset, float] = UNSET,
    lmp_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """RTD Indicative LMPs by Resource Nodes, Load Zones and Hubs

     RTD Indicative LMPs by Resource Nodes, Load Zones and Hubs

    Args:
        rtd_timestamp_from (Union[Unset, str]):
        rtd_timestamp_to (Union[Unset, str]):
        repeat_hour_flag (Union[Unset, bool]):
        interval_id_from (Union[Unset, int]):
        interval_id_to (Union[Unset, int]):
        interval_ending_from (Union[Unset, str]):
        interval_ending_to (Union[Unset, str]):
        interval_repeat_hour_flag (Union[Unset, bool]):
        settlement_point (Union[Unset, str]):
        settlement_point_type (Union[Unset, str]):
        lmp_from (Union[Unset, float]):
        lmp_to (Union[Unset, float]):
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
        rtd_timestamp_from=rtd_timestamp_from,
        rtd_timestamp_to=rtd_timestamp_to,
        repeat_hour_flag=repeat_hour_flag,
        interval_id_from=interval_id_from,
        interval_id_to=interval_id_to,
        interval_ending_from=interval_ending_from,
        interval_ending_to=interval_ending_to,
        interval_repeat_hour_flag=interval_repeat_hour_flag,
        settlement_point=settlement_point,
        settlement_point_type=settlement_point_type,
        lmp_from=lmp_from,
        lmp_to=lmp_to,
        page=page,
        size=size,
        sort=sort,
        dir_=dir_,
        ocp_apim_subscription_key=ocp_apim_subscription_key,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    rtd_timestamp_from: Union[Unset, str] = UNSET,
    rtd_timestamp_to: Union[Unset, str] = UNSET,
    repeat_hour_flag: Union[Unset, bool] = UNSET,
    interval_id_from: Union[Unset, int] = UNSET,
    interval_id_to: Union[Unset, int] = UNSET,
    interval_ending_from: Union[Unset, str] = UNSET,
    interval_ending_to: Union[Unset, str] = UNSET,
    interval_repeat_hour_flag: Union[Unset, bool] = UNSET,
    settlement_point: Union[Unset, str] = UNSET,
    settlement_point_type: Union[Unset, str] = UNSET,
    lmp_from: Union[Unset, float] = UNSET,
    lmp_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """RTD Indicative LMPs by Resource Nodes, Load Zones and Hubs

     RTD Indicative LMPs by Resource Nodes, Load Zones and Hubs

    Args:
        rtd_timestamp_from (Union[Unset, str]):
        rtd_timestamp_to (Union[Unset, str]):
        repeat_hour_flag (Union[Unset, bool]):
        interval_id_from (Union[Unset, int]):
        interval_id_to (Union[Unset, int]):
        interval_ending_from (Union[Unset, str]):
        interval_ending_to (Union[Unset, str]):
        interval_repeat_hour_flag (Union[Unset, bool]):
        settlement_point (Union[Unset, str]):
        settlement_point_type (Union[Unset, str]):
        lmp_from (Union[Unset, float]):
        lmp_to (Union[Unset, float]):
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
        rtd_timestamp_from=rtd_timestamp_from,
        rtd_timestamp_to=rtd_timestamp_to,
        repeat_hour_flag=repeat_hour_flag,
        interval_id_from=interval_id_from,
        interval_id_to=interval_id_to,
        interval_ending_from=interval_ending_from,
        interval_ending_to=interval_ending_to,
        interval_repeat_hour_flag=interval_repeat_hour_flag,
        settlement_point=settlement_point,
        settlement_point_type=settlement_point_type,
        lmp_from=lmp_from,
        lmp_to=lmp_to,
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
    rtd_timestamp_from: Union[Unset, str] = UNSET,
    rtd_timestamp_to: Union[Unset, str] = UNSET,
    repeat_hour_flag: Union[Unset, bool] = UNSET,
    interval_id_from: Union[Unset, int] = UNSET,
    interval_id_to: Union[Unset, int] = UNSET,
    interval_ending_from: Union[Unset, str] = UNSET,
    interval_ending_to: Union[Unset, str] = UNSET,
    interval_repeat_hour_flag: Union[Unset, bool] = UNSET,
    settlement_point: Union[Unset, str] = UNSET,
    settlement_point_type: Union[Unset, str] = UNSET,
    lmp_from: Union[Unset, float] = UNSET,
    lmp_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """RTD Indicative LMPs by Resource Nodes, Load Zones and Hubs

     RTD Indicative LMPs by Resource Nodes, Load Zones and Hubs

    Args:
        rtd_timestamp_from (Union[Unset, str]):
        rtd_timestamp_to (Union[Unset, str]):
        repeat_hour_flag (Union[Unset, bool]):
        interval_id_from (Union[Unset, int]):
        interval_id_to (Union[Unset, int]):
        interval_ending_from (Union[Unset, str]):
        interval_ending_to (Union[Unset, str]):
        interval_repeat_hour_flag (Union[Unset, bool]):
        settlement_point (Union[Unset, str]):
        settlement_point_type (Union[Unset, str]):
        lmp_from (Union[Unset, float]):
        lmp_to (Union[Unset, float]):
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
            rtd_timestamp_from=rtd_timestamp_from,
            rtd_timestamp_to=rtd_timestamp_to,
            repeat_hour_flag=repeat_hour_flag,
            interval_id_from=interval_id_from,
            interval_id_to=interval_id_to,
            interval_ending_from=interval_ending_from,
            interval_ending_to=interval_ending_to,
            interval_repeat_hour_flag=interval_repeat_hour_flag,
            settlement_point=settlement_point,
            settlement_point_type=settlement_point_type,
            lmp_from=lmp_from,
            lmp_to=lmp_to,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
