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
    operating_day_from: Union[Unset, str] = UNSET,
    operating_day_to: Union[Unset, str] = UNSET,
    hour_ending: Union[Unset, str] = UNSET,
    north_from: Union[Unset, float] = UNSET,
    north_to: Union[Unset, float] = UNSET,
    south_from: Union[Unset, float] = UNSET,
    south_to: Union[Unset, float] = UNSET,
    west_from: Union[Unset, float] = UNSET,
    west_to: Union[Unset, float] = UNSET,
    houston_from: Union[Unset, float] = UNSET,
    houston_to: Union[Unset, float] = UNSET,
    total_from: Union[Unset, float] = UNSET,
    total_to: Union[Unset, float] = UNSET,
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

    params["operatingDayFrom"] = operating_day_from

    params["operatingDayTo"] = operating_day_to

    params["hourEnding"] = hour_ending

    params["northFrom"] = north_from

    params["northTo"] = north_to

    params["southFrom"] = south_from

    params["southTo"] = south_to

    params["westFrom"] = west_from

    params["westTo"] = west_to

    params["houstonFrom"] = houston_from

    params["houstonTo"] = houston_to

    params["totalFrom"] = total_from

    params["totalTo"] = total_to

    params["DSTFlag"] = dst_flag

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np6-346-cd/act_sys_load_by_fzn",
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
    operating_day_from: Union[Unset, str] = UNSET,
    operating_day_to: Union[Unset, str] = UNSET,
    hour_ending: Union[Unset, str] = UNSET,
    north_from: Union[Unset, float] = UNSET,
    north_to: Union[Unset, float] = UNSET,
    south_from: Union[Unset, float] = UNSET,
    south_to: Union[Unset, float] = UNSET,
    west_from: Union[Unset, float] = UNSET,
    west_to: Union[Unset, float] = UNSET,
    houston_from: Union[Unset, float] = UNSET,
    houston_to: Union[Unset, float] = UNSET,
    total_from: Union[Unset, float] = UNSET,
    total_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """Actual System Load by Forecast Zone

     Actual System Load by Forecast Zone

    Args:
        operating_day_from (Union[Unset, str]):
        operating_day_to (Union[Unset, str]):
        hour_ending (Union[Unset, str]):
        north_from (Union[Unset, float]):
        north_to (Union[Unset, float]):
        south_from (Union[Unset, float]):
        south_to (Union[Unset, float]):
        west_from (Union[Unset, float]):
        west_to (Union[Unset, float]):
        houston_from (Union[Unset, float]):
        houston_to (Union[Unset, float]):
        total_from (Union[Unset, float]):
        total_to (Union[Unset, float]):
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
        operating_day_from=operating_day_from,
        operating_day_to=operating_day_to,
        hour_ending=hour_ending,
        north_from=north_from,
        north_to=north_to,
        south_from=south_from,
        south_to=south_to,
        west_from=west_from,
        west_to=west_to,
        houston_from=houston_from,
        houston_to=houston_to,
        total_from=total_from,
        total_to=total_to,
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
    operating_day_from: Union[Unset, str] = UNSET,
    operating_day_to: Union[Unset, str] = UNSET,
    hour_ending: Union[Unset, str] = UNSET,
    north_from: Union[Unset, float] = UNSET,
    north_to: Union[Unset, float] = UNSET,
    south_from: Union[Unset, float] = UNSET,
    south_to: Union[Unset, float] = UNSET,
    west_from: Union[Unset, float] = UNSET,
    west_to: Union[Unset, float] = UNSET,
    houston_from: Union[Unset, float] = UNSET,
    houston_to: Union[Unset, float] = UNSET,
    total_from: Union[Unset, float] = UNSET,
    total_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """Actual System Load by Forecast Zone

     Actual System Load by Forecast Zone

    Args:
        operating_day_from (Union[Unset, str]):
        operating_day_to (Union[Unset, str]):
        hour_ending (Union[Unset, str]):
        north_from (Union[Unset, float]):
        north_to (Union[Unset, float]):
        south_from (Union[Unset, float]):
        south_to (Union[Unset, float]):
        west_from (Union[Unset, float]):
        west_to (Union[Unset, float]):
        houston_from (Union[Unset, float]):
        houston_to (Union[Unset, float]):
        total_from (Union[Unset, float]):
        total_to (Union[Unset, float]):
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
        operating_day_from=operating_day_from,
        operating_day_to=operating_day_to,
        hour_ending=hour_ending,
        north_from=north_from,
        north_to=north_to,
        south_from=south_from,
        south_to=south_to,
        west_from=west_from,
        west_to=west_to,
        houston_from=houston_from,
        houston_to=houston_to,
        total_from=total_from,
        total_to=total_to,
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
    operating_day_from: Union[Unset, str] = UNSET,
    operating_day_to: Union[Unset, str] = UNSET,
    hour_ending: Union[Unset, str] = UNSET,
    north_from: Union[Unset, float] = UNSET,
    north_to: Union[Unset, float] = UNSET,
    south_from: Union[Unset, float] = UNSET,
    south_to: Union[Unset, float] = UNSET,
    west_from: Union[Unset, float] = UNSET,
    west_to: Union[Unset, float] = UNSET,
    houston_from: Union[Unset, float] = UNSET,
    houston_to: Union[Unset, float] = UNSET,
    total_from: Union[Unset, float] = UNSET,
    total_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """Actual System Load by Forecast Zone

     Actual System Load by Forecast Zone

    Args:
        operating_day_from (Union[Unset, str]):
        operating_day_to (Union[Unset, str]):
        hour_ending (Union[Unset, str]):
        north_from (Union[Unset, float]):
        north_to (Union[Unset, float]):
        south_from (Union[Unset, float]):
        south_to (Union[Unset, float]):
        west_from (Union[Unset, float]):
        west_to (Union[Unset, float]):
        houston_from (Union[Unset, float]):
        houston_to (Union[Unset, float]):
        total_from (Union[Unset, float]):
        total_to (Union[Unset, float]):
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
        operating_day_from=operating_day_from,
        operating_day_to=operating_day_to,
        hour_ending=hour_ending,
        north_from=north_from,
        north_to=north_to,
        south_from=south_from,
        south_to=south_to,
        west_from=west_from,
        west_to=west_to,
        houston_from=houston_from,
        houston_to=houston_to,
        total_from=total_from,
        total_to=total_to,
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
    operating_day_from: Union[Unset, str] = UNSET,
    operating_day_to: Union[Unset, str] = UNSET,
    hour_ending: Union[Unset, str] = UNSET,
    north_from: Union[Unset, float] = UNSET,
    north_to: Union[Unset, float] = UNSET,
    south_from: Union[Unset, float] = UNSET,
    south_to: Union[Unset, float] = UNSET,
    west_from: Union[Unset, float] = UNSET,
    west_to: Union[Unset, float] = UNSET,
    houston_from: Union[Unset, float] = UNSET,
    houston_to: Union[Unset, float] = UNSET,
    total_from: Union[Unset, float] = UNSET,
    total_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """Actual System Load by Forecast Zone

     Actual System Load by Forecast Zone

    Args:
        operating_day_from (Union[Unset, str]):
        operating_day_to (Union[Unset, str]):
        hour_ending (Union[Unset, str]):
        north_from (Union[Unset, float]):
        north_to (Union[Unset, float]):
        south_from (Union[Unset, float]):
        south_to (Union[Unset, float]):
        west_from (Union[Unset, float]):
        west_to (Union[Unset, float]):
        houston_from (Union[Unset, float]):
        houston_to (Union[Unset, float]):
        total_from (Union[Unset, float]):
        total_to (Union[Unset, float]):
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
            operating_day_from=operating_day_from,
            operating_day_to=operating_day_to,
            hour_ending=hour_ending,
            north_from=north_from,
            north_to=north_to,
            south_from=south_from,
            south_to=south_to,
            west_from=west_from,
            west_to=west_to,
            houston_from=houston_from,
            houston_to=houston_to,
            total_from=total_from,
            total_to=total_to,
            dst_flag=dst_flag,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
