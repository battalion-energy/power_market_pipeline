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
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    interval_ending_from: Union[Unset, str] = UNSET,
    interval_ending_to: Union[Unset, str] = UNSET,
    gen_system_wide_from: Union[Unset, float] = UNSET,
    gen_system_wide_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    hsl_system_wide_from: Union[Unset, float] = UNSET,
    hsl_system_wide_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["Ocp-Apim-Subscription-Key"] = ocp_apim_subscription_key

    params: dict[str, Any] = {}

    params["postedDatetimeFrom"] = posted_datetime_from

    params["postedDatetimeTo"] = posted_datetime_to

    params["intervalEndingFrom"] = interval_ending_from

    params["intervalEndingTo"] = interval_ending_to

    params["genSystemWideFrom"] = gen_system_wide_from

    params["genSystemWideTo"] = gen_system_wide_to

    params["DSTFlag"] = dst_flag

    params["HSLSystemWideFrom"] = hsl_system_wide_from

    params["HSLSystemWideTo"] = hsl_system_wide_to

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np4-738-cd/spp_actual_5min_avg_values",
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
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    interval_ending_from: Union[Unset, str] = UNSET,
    interval_ending_to: Union[Unset, str] = UNSET,
    gen_system_wide_from: Union[Unset, float] = UNSET,
    gen_system_wide_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    hsl_system_wide_from: Union[Unset, float] = UNSET,
    hsl_system_wide_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """Solar Power Production - Actual 5-Minute Averaged Values

     Solar Power Production - Actual 5-Minute Averaged Values

    Args:
        posted_datetime_from (Union[Unset, str]):
        posted_datetime_to (Union[Unset, str]):
        interval_ending_from (Union[Unset, str]):
        interval_ending_to (Union[Unset, str]):
        gen_system_wide_from (Union[Unset, float]):
        gen_system_wide_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
        hsl_system_wide_from (Union[Unset, float]):
        hsl_system_wide_to (Union[Unset, float]):
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
        posted_datetime_from=posted_datetime_from,
        posted_datetime_to=posted_datetime_to,
        interval_ending_from=interval_ending_from,
        interval_ending_to=interval_ending_to,
        gen_system_wide_from=gen_system_wide_from,
        gen_system_wide_to=gen_system_wide_to,
        dst_flag=dst_flag,
        hsl_system_wide_from=hsl_system_wide_from,
        hsl_system_wide_to=hsl_system_wide_to,
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
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    interval_ending_from: Union[Unset, str] = UNSET,
    interval_ending_to: Union[Unset, str] = UNSET,
    gen_system_wide_from: Union[Unset, float] = UNSET,
    gen_system_wide_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    hsl_system_wide_from: Union[Unset, float] = UNSET,
    hsl_system_wide_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """Solar Power Production - Actual 5-Minute Averaged Values

     Solar Power Production - Actual 5-Minute Averaged Values

    Args:
        posted_datetime_from (Union[Unset, str]):
        posted_datetime_to (Union[Unset, str]):
        interval_ending_from (Union[Unset, str]):
        interval_ending_to (Union[Unset, str]):
        gen_system_wide_from (Union[Unset, float]):
        gen_system_wide_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
        hsl_system_wide_from (Union[Unset, float]):
        hsl_system_wide_to (Union[Unset, float]):
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
        posted_datetime_from=posted_datetime_from,
        posted_datetime_to=posted_datetime_to,
        interval_ending_from=interval_ending_from,
        interval_ending_to=interval_ending_to,
        gen_system_wide_from=gen_system_wide_from,
        gen_system_wide_to=gen_system_wide_to,
        dst_flag=dst_flag,
        hsl_system_wide_from=hsl_system_wide_from,
        hsl_system_wide_to=hsl_system_wide_to,
        page=page,
        size=size,
        sort=sort,
        dir_=dir_,
        ocp_apim_subscription_key=ocp_apim_subscription_key,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    interval_ending_from: Union[Unset, str] = UNSET,
    interval_ending_to: Union[Unset, str] = UNSET,
    gen_system_wide_from: Union[Unset, float] = UNSET,
    gen_system_wide_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    hsl_system_wide_from: Union[Unset, float] = UNSET,
    hsl_system_wide_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """Solar Power Production - Actual 5-Minute Averaged Values

     Solar Power Production - Actual 5-Minute Averaged Values

    Args:
        posted_datetime_from (Union[Unset, str]):
        posted_datetime_to (Union[Unset, str]):
        interval_ending_from (Union[Unset, str]):
        interval_ending_to (Union[Unset, str]):
        gen_system_wide_from (Union[Unset, float]):
        gen_system_wide_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
        hsl_system_wide_from (Union[Unset, float]):
        hsl_system_wide_to (Union[Unset, float]):
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
        posted_datetime_from=posted_datetime_from,
        posted_datetime_to=posted_datetime_to,
        interval_ending_from=interval_ending_from,
        interval_ending_to=interval_ending_to,
        gen_system_wide_from=gen_system_wide_from,
        gen_system_wide_to=gen_system_wide_to,
        dst_flag=dst_flag,
        hsl_system_wide_from=hsl_system_wide_from,
        hsl_system_wide_to=hsl_system_wide_to,
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
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    interval_ending_from: Union[Unset, str] = UNSET,
    interval_ending_to: Union[Unset, str] = UNSET,
    gen_system_wide_from: Union[Unset, float] = UNSET,
    gen_system_wide_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    hsl_system_wide_from: Union[Unset, float] = UNSET,
    hsl_system_wide_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """Solar Power Production - Actual 5-Minute Averaged Values

     Solar Power Production - Actual 5-Minute Averaged Values

    Args:
        posted_datetime_from (Union[Unset, str]):
        posted_datetime_to (Union[Unset, str]):
        interval_ending_from (Union[Unset, str]):
        interval_ending_to (Union[Unset, str]):
        gen_system_wide_from (Union[Unset, float]):
        gen_system_wide_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
        hsl_system_wide_from (Union[Unset, float]):
        hsl_system_wide_to (Union[Unset, float]):
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
            posted_datetime_from=posted_datetime_from,
            posted_datetime_to=posted_datetime_to,
            interval_ending_from=interval_ending_from,
            interval_ending_to=interval_ending_to,
            gen_system_wide_from=gen_system_wide_from,
            gen_system_wide_to=gen_system_wide_to,
            dst_flag=dst_flag,
            hsl_system_wide_from=hsl_system_wide_from,
            hsl_system_wide_to=hsl_system_wide_to,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
