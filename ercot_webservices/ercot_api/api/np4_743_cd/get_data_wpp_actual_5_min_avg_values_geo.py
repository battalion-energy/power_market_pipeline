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
    hsl_system_wide_from: Union[Unset, float] = UNSET,
    hsl_system_wide_to: Union[Unset, float] = UNSET,
    interval_ending_from: Union[Unset, str] = UNSET,
    interval_ending_to: Union[Unset, str] = UNSET,
    gen_system_wide_from: Union[Unset, float] = UNSET,
    gen_system_wide_to: Union[Unset, float] = UNSET,
    panhandle_from: Union[Unset, float] = UNSET,
    panhandle_to: Union[Unset, float] = UNSET,
    coast_from: Union[Unset, float] = UNSET,
    coast_to: Union[Unset, float] = UNSET,
    south_from: Union[Unset, float] = UNSET,
    south_to: Union[Unset, float] = UNSET,
    west_from: Union[Unset, float] = UNSET,
    west_to: Union[Unset, float] = UNSET,
    north_from: Union[Unset, float] = UNSET,
    north_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["Ocp-Apim-Subscription-Key"] = ocp_apim_subscription_key

    params: dict[str, Any] = {}

    params["HSLSystemWideFrom"] = hsl_system_wide_from

    params["HSLSystemWideTo"] = hsl_system_wide_to

    params["intervalEndingFrom"] = interval_ending_from

    params["intervalEndingTo"] = interval_ending_to

    params["genSystemWideFrom"] = gen_system_wide_from

    params["genSystemWideTo"] = gen_system_wide_to

    params["panhandleFrom"] = panhandle_from

    params["panhandleTo"] = panhandle_to

    params["coastFrom"] = coast_from

    params["coastTo"] = coast_to

    params["southFrom"] = south_from

    params["southTo"] = south_to

    params["westFrom"] = west_from

    params["westTo"] = west_to

    params["northFrom"] = north_from

    params["northTo"] = north_to

    params["DSTFlag"] = dst_flag

    params["postedDatetimeFrom"] = posted_datetime_from

    params["postedDatetimeTo"] = posted_datetime_to

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np4-743-cd/wpp_actual_5min_avg_values_geo",
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
    hsl_system_wide_from: Union[Unset, float] = UNSET,
    hsl_system_wide_to: Union[Unset, float] = UNSET,
    interval_ending_from: Union[Unset, str] = UNSET,
    interval_ending_to: Union[Unset, str] = UNSET,
    gen_system_wide_from: Union[Unset, float] = UNSET,
    gen_system_wide_to: Union[Unset, float] = UNSET,
    panhandle_from: Union[Unset, float] = UNSET,
    panhandle_to: Union[Unset, float] = UNSET,
    coast_from: Union[Unset, float] = UNSET,
    coast_to: Union[Unset, float] = UNSET,
    south_from: Union[Unset, float] = UNSET,
    south_to: Union[Unset, float] = UNSET,
    west_from: Union[Unset, float] = UNSET,
    west_to: Union[Unset, float] = UNSET,
    north_from: Union[Unset, float] = UNSET,
    north_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """Wind Power Production - Actual 5-Minute Averaged Values by Geographical Region

     Wind Power Production - Actual 5-Minute Averaged Values by Geographical Region

    Args:
        hsl_system_wide_from (Union[Unset, float]):
        hsl_system_wide_to (Union[Unset, float]):
        interval_ending_from (Union[Unset, str]):
        interval_ending_to (Union[Unset, str]):
        gen_system_wide_from (Union[Unset, float]):
        gen_system_wide_to (Union[Unset, float]):
        panhandle_from (Union[Unset, float]):
        panhandle_to (Union[Unset, float]):
        coast_from (Union[Unset, float]):
        coast_to (Union[Unset, float]):
        south_from (Union[Unset, float]):
        south_to (Union[Unset, float]):
        west_from (Union[Unset, float]):
        west_to (Union[Unset, float]):
        north_from (Union[Unset, float]):
        north_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
        posted_datetime_from (Union[Unset, str]):
        posted_datetime_to (Union[Unset, str]):
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
        hsl_system_wide_from=hsl_system_wide_from,
        hsl_system_wide_to=hsl_system_wide_to,
        interval_ending_from=interval_ending_from,
        interval_ending_to=interval_ending_to,
        gen_system_wide_from=gen_system_wide_from,
        gen_system_wide_to=gen_system_wide_to,
        panhandle_from=panhandle_from,
        panhandle_to=panhandle_to,
        coast_from=coast_from,
        coast_to=coast_to,
        south_from=south_from,
        south_to=south_to,
        west_from=west_from,
        west_to=west_to,
        north_from=north_from,
        north_to=north_to,
        dst_flag=dst_flag,
        posted_datetime_from=posted_datetime_from,
        posted_datetime_to=posted_datetime_to,
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
    hsl_system_wide_from: Union[Unset, float] = UNSET,
    hsl_system_wide_to: Union[Unset, float] = UNSET,
    interval_ending_from: Union[Unset, str] = UNSET,
    interval_ending_to: Union[Unset, str] = UNSET,
    gen_system_wide_from: Union[Unset, float] = UNSET,
    gen_system_wide_to: Union[Unset, float] = UNSET,
    panhandle_from: Union[Unset, float] = UNSET,
    panhandle_to: Union[Unset, float] = UNSET,
    coast_from: Union[Unset, float] = UNSET,
    coast_to: Union[Unset, float] = UNSET,
    south_from: Union[Unset, float] = UNSET,
    south_to: Union[Unset, float] = UNSET,
    west_from: Union[Unset, float] = UNSET,
    west_to: Union[Unset, float] = UNSET,
    north_from: Union[Unset, float] = UNSET,
    north_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """Wind Power Production - Actual 5-Minute Averaged Values by Geographical Region

     Wind Power Production - Actual 5-Minute Averaged Values by Geographical Region

    Args:
        hsl_system_wide_from (Union[Unset, float]):
        hsl_system_wide_to (Union[Unset, float]):
        interval_ending_from (Union[Unset, str]):
        interval_ending_to (Union[Unset, str]):
        gen_system_wide_from (Union[Unset, float]):
        gen_system_wide_to (Union[Unset, float]):
        panhandle_from (Union[Unset, float]):
        panhandle_to (Union[Unset, float]):
        coast_from (Union[Unset, float]):
        coast_to (Union[Unset, float]):
        south_from (Union[Unset, float]):
        south_to (Union[Unset, float]):
        west_from (Union[Unset, float]):
        west_to (Union[Unset, float]):
        north_from (Union[Unset, float]):
        north_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
        posted_datetime_from (Union[Unset, str]):
        posted_datetime_to (Union[Unset, str]):
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
        hsl_system_wide_from=hsl_system_wide_from,
        hsl_system_wide_to=hsl_system_wide_to,
        interval_ending_from=interval_ending_from,
        interval_ending_to=interval_ending_to,
        gen_system_wide_from=gen_system_wide_from,
        gen_system_wide_to=gen_system_wide_to,
        panhandle_from=panhandle_from,
        panhandle_to=panhandle_to,
        coast_from=coast_from,
        coast_to=coast_to,
        south_from=south_from,
        south_to=south_to,
        west_from=west_from,
        west_to=west_to,
        north_from=north_from,
        north_to=north_to,
        dst_flag=dst_flag,
        posted_datetime_from=posted_datetime_from,
        posted_datetime_to=posted_datetime_to,
        page=page,
        size=size,
        sort=sort,
        dir_=dir_,
        ocp_apim_subscription_key=ocp_apim_subscription_key,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    hsl_system_wide_from: Union[Unset, float] = UNSET,
    hsl_system_wide_to: Union[Unset, float] = UNSET,
    interval_ending_from: Union[Unset, str] = UNSET,
    interval_ending_to: Union[Unset, str] = UNSET,
    gen_system_wide_from: Union[Unset, float] = UNSET,
    gen_system_wide_to: Union[Unset, float] = UNSET,
    panhandle_from: Union[Unset, float] = UNSET,
    panhandle_to: Union[Unset, float] = UNSET,
    coast_from: Union[Unset, float] = UNSET,
    coast_to: Union[Unset, float] = UNSET,
    south_from: Union[Unset, float] = UNSET,
    south_to: Union[Unset, float] = UNSET,
    west_from: Union[Unset, float] = UNSET,
    west_to: Union[Unset, float] = UNSET,
    north_from: Union[Unset, float] = UNSET,
    north_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """Wind Power Production - Actual 5-Minute Averaged Values by Geographical Region

     Wind Power Production - Actual 5-Minute Averaged Values by Geographical Region

    Args:
        hsl_system_wide_from (Union[Unset, float]):
        hsl_system_wide_to (Union[Unset, float]):
        interval_ending_from (Union[Unset, str]):
        interval_ending_to (Union[Unset, str]):
        gen_system_wide_from (Union[Unset, float]):
        gen_system_wide_to (Union[Unset, float]):
        panhandle_from (Union[Unset, float]):
        panhandle_to (Union[Unset, float]):
        coast_from (Union[Unset, float]):
        coast_to (Union[Unset, float]):
        south_from (Union[Unset, float]):
        south_to (Union[Unset, float]):
        west_from (Union[Unset, float]):
        west_to (Union[Unset, float]):
        north_from (Union[Unset, float]):
        north_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
        posted_datetime_from (Union[Unset, str]):
        posted_datetime_to (Union[Unset, str]):
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
        hsl_system_wide_from=hsl_system_wide_from,
        hsl_system_wide_to=hsl_system_wide_to,
        interval_ending_from=interval_ending_from,
        interval_ending_to=interval_ending_to,
        gen_system_wide_from=gen_system_wide_from,
        gen_system_wide_to=gen_system_wide_to,
        panhandle_from=panhandle_from,
        panhandle_to=panhandle_to,
        coast_from=coast_from,
        coast_to=coast_to,
        south_from=south_from,
        south_to=south_to,
        west_from=west_from,
        west_to=west_to,
        north_from=north_from,
        north_to=north_to,
        dst_flag=dst_flag,
        posted_datetime_from=posted_datetime_from,
        posted_datetime_to=posted_datetime_to,
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
    hsl_system_wide_from: Union[Unset, float] = UNSET,
    hsl_system_wide_to: Union[Unset, float] = UNSET,
    interval_ending_from: Union[Unset, str] = UNSET,
    interval_ending_to: Union[Unset, str] = UNSET,
    gen_system_wide_from: Union[Unset, float] = UNSET,
    gen_system_wide_to: Union[Unset, float] = UNSET,
    panhandle_from: Union[Unset, float] = UNSET,
    panhandle_to: Union[Unset, float] = UNSET,
    coast_from: Union[Unset, float] = UNSET,
    coast_to: Union[Unset, float] = UNSET,
    south_from: Union[Unset, float] = UNSET,
    south_to: Union[Unset, float] = UNSET,
    west_from: Union[Unset, float] = UNSET,
    west_to: Union[Unset, float] = UNSET,
    north_from: Union[Unset, float] = UNSET,
    north_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """Wind Power Production - Actual 5-Minute Averaged Values by Geographical Region

     Wind Power Production - Actual 5-Minute Averaged Values by Geographical Region

    Args:
        hsl_system_wide_from (Union[Unset, float]):
        hsl_system_wide_to (Union[Unset, float]):
        interval_ending_from (Union[Unset, str]):
        interval_ending_to (Union[Unset, str]):
        gen_system_wide_from (Union[Unset, float]):
        gen_system_wide_to (Union[Unset, float]):
        panhandle_from (Union[Unset, float]):
        panhandle_to (Union[Unset, float]):
        coast_from (Union[Unset, float]):
        coast_to (Union[Unset, float]):
        south_from (Union[Unset, float]):
        south_to (Union[Unset, float]):
        west_from (Union[Unset, float]):
        west_to (Union[Unset, float]):
        north_from (Union[Unset, float]):
        north_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
        posted_datetime_from (Union[Unset, str]):
        posted_datetime_to (Union[Unset, str]):
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
            hsl_system_wide_from=hsl_system_wide_from,
            hsl_system_wide_to=hsl_system_wide_to,
            interval_ending_from=interval_ending_from,
            interval_ending_to=interval_ending_to,
            gen_system_wide_from=gen_system_wide_from,
            gen_system_wide_to=gen_system_wide_to,
            panhandle_from=panhandle_from,
            panhandle_to=panhandle_to,
            coast_from=coast_from,
            coast_to=coast_to,
            south_from=south_from,
            south_to=south_to,
            west_from=west_from,
            west_to=west_to,
            north_from=north_from,
            north_to=north_to,
            dst_flag=dst_flag,
            posted_datetime_from=posted_datetime_from,
            posted_datetime_to=posted_datetime_to,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
