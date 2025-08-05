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
    stppf_system_wide_from: Union[Unset, float] = UNSET,
    stppf_system_wide_to: Union[Unset, float] = UNSET,
    pvgrpp_system_wide_from: Union[Unset, float] = UNSET,
    pvgrpp_system_wide_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hsl_system_wide_from: Union[Unset, float] = UNSET,
    hsl_system_wide_to: Union[Unset, float] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    gen_system_wide_from: Union[Unset, float] = UNSET,
    gen_system_wide_to: Union[Unset, float] = UNSET,
    cophsl_system_wide_from: Union[Unset, float] = UNSET,
    cophsl_system_wide_to: Union[Unset, float] = UNSET,
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

    params["STPPFSystemWideFrom"] = stppf_system_wide_from

    params["STPPFSystemWideTo"] = stppf_system_wide_to

    params["PVGRPPSystemWideFrom"] = pvgrpp_system_wide_from

    params["PVGRPPSystemWideTo"] = pvgrpp_system_wide_to

    params["DSTFlag"] = dst_flag

    params["deliveryDateFrom"] = delivery_date_from

    params["deliveryDateTo"] = delivery_date_to

    params["HSLSystemWideFrom"] = hsl_system_wide_from

    params["HSLSystemWideTo"] = hsl_system_wide_to

    params["hourEndingFrom"] = hour_ending_from

    params["hourEndingTo"] = hour_ending_to

    params["genSystemWideFrom"] = gen_system_wide_from

    params["genSystemWideTo"] = gen_system_wide_to

    params["COPHSLSystemWideFrom"] = cophsl_system_wide_from

    params["COPHSLSystemWideTo"] = cophsl_system_wide_to

    params["postedDatetimeFrom"] = posted_datetime_from

    params["postedDatetimeTo"] = posted_datetime_to

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np4-737-cd/spp_hrly_avrg_actl_fcast",
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
    stppf_system_wide_from: Union[Unset, float] = UNSET,
    stppf_system_wide_to: Union[Unset, float] = UNSET,
    pvgrpp_system_wide_from: Union[Unset, float] = UNSET,
    pvgrpp_system_wide_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hsl_system_wide_from: Union[Unset, float] = UNSET,
    hsl_system_wide_to: Union[Unset, float] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    gen_system_wide_from: Union[Unset, float] = UNSET,
    gen_system_wide_to: Union[Unset, float] = UNSET,
    cophsl_system_wide_from: Union[Unset, float] = UNSET,
    cophsl_system_wide_to: Union[Unset, float] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """Solar Power Production - Hourly Averaged Actual and Forecasted Values

     Solar Power Production - Hourly Averaged Actual and Forecasted Values

    Args:
        stppf_system_wide_from (Union[Unset, float]):
        stppf_system_wide_to (Union[Unset, float]):
        pvgrpp_system_wide_from (Union[Unset, float]):
        pvgrpp_system_wide_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hsl_system_wide_from (Union[Unset, float]):
        hsl_system_wide_to (Union[Unset, float]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        gen_system_wide_from (Union[Unset, float]):
        gen_system_wide_to (Union[Unset, float]):
        cophsl_system_wide_from (Union[Unset, float]):
        cophsl_system_wide_to (Union[Unset, float]):
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
        stppf_system_wide_from=stppf_system_wide_from,
        stppf_system_wide_to=stppf_system_wide_to,
        pvgrpp_system_wide_from=pvgrpp_system_wide_from,
        pvgrpp_system_wide_to=pvgrpp_system_wide_to,
        dst_flag=dst_flag,
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        hsl_system_wide_from=hsl_system_wide_from,
        hsl_system_wide_to=hsl_system_wide_to,
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        gen_system_wide_from=gen_system_wide_from,
        gen_system_wide_to=gen_system_wide_to,
        cophsl_system_wide_from=cophsl_system_wide_from,
        cophsl_system_wide_to=cophsl_system_wide_to,
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
    stppf_system_wide_from: Union[Unset, float] = UNSET,
    stppf_system_wide_to: Union[Unset, float] = UNSET,
    pvgrpp_system_wide_from: Union[Unset, float] = UNSET,
    pvgrpp_system_wide_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hsl_system_wide_from: Union[Unset, float] = UNSET,
    hsl_system_wide_to: Union[Unset, float] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    gen_system_wide_from: Union[Unset, float] = UNSET,
    gen_system_wide_to: Union[Unset, float] = UNSET,
    cophsl_system_wide_from: Union[Unset, float] = UNSET,
    cophsl_system_wide_to: Union[Unset, float] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """Solar Power Production - Hourly Averaged Actual and Forecasted Values

     Solar Power Production - Hourly Averaged Actual and Forecasted Values

    Args:
        stppf_system_wide_from (Union[Unset, float]):
        stppf_system_wide_to (Union[Unset, float]):
        pvgrpp_system_wide_from (Union[Unset, float]):
        pvgrpp_system_wide_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hsl_system_wide_from (Union[Unset, float]):
        hsl_system_wide_to (Union[Unset, float]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        gen_system_wide_from (Union[Unset, float]):
        gen_system_wide_to (Union[Unset, float]):
        cophsl_system_wide_from (Union[Unset, float]):
        cophsl_system_wide_to (Union[Unset, float]):
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
        stppf_system_wide_from=stppf_system_wide_from,
        stppf_system_wide_to=stppf_system_wide_to,
        pvgrpp_system_wide_from=pvgrpp_system_wide_from,
        pvgrpp_system_wide_to=pvgrpp_system_wide_to,
        dst_flag=dst_flag,
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        hsl_system_wide_from=hsl_system_wide_from,
        hsl_system_wide_to=hsl_system_wide_to,
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        gen_system_wide_from=gen_system_wide_from,
        gen_system_wide_to=gen_system_wide_to,
        cophsl_system_wide_from=cophsl_system_wide_from,
        cophsl_system_wide_to=cophsl_system_wide_to,
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
    stppf_system_wide_from: Union[Unset, float] = UNSET,
    stppf_system_wide_to: Union[Unset, float] = UNSET,
    pvgrpp_system_wide_from: Union[Unset, float] = UNSET,
    pvgrpp_system_wide_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hsl_system_wide_from: Union[Unset, float] = UNSET,
    hsl_system_wide_to: Union[Unset, float] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    gen_system_wide_from: Union[Unset, float] = UNSET,
    gen_system_wide_to: Union[Unset, float] = UNSET,
    cophsl_system_wide_from: Union[Unset, float] = UNSET,
    cophsl_system_wide_to: Union[Unset, float] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """Solar Power Production - Hourly Averaged Actual and Forecasted Values

     Solar Power Production - Hourly Averaged Actual and Forecasted Values

    Args:
        stppf_system_wide_from (Union[Unset, float]):
        stppf_system_wide_to (Union[Unset, float]):
        pvgrpp_system_wide_from (Union[Unset, float]):
        pvgrpp_system_wide_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hsl_system_wide_from (Union[Unset, float]):
        hsl_system_wide_to (Union[Unset, float]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        gen_system_wide_from (Union[Unset, float]):
        gen_system_wide_to (Union[Unset, float]):
        cophsl_system_wide_from (Union[Unset, float]):
        cophsl_system_wide_to (Union[Unset, float]):
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
        stppf_system_wide_from=stppf_system_wide_from,
        stppf_system_wide_to=stppf_system_wide_to,
        pvgrpp_system_wide_from=pvgrpp_system_wide_from,
        pvgrpp_system_wide_to=pvgrpp_system_wide_to,
        dst_flag=dst_flag,
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        hsl_system_wide_from=hsl_system_wide_from,
        hsl_system_wide_to=hsl_system_wide_to,
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        gen_system_wide_from=gen_system_wide_from,
        gen_system_wide_to=gen_system_wide_to,
        cophsl_system_wide_from=cophsl_system_wide_from,
        cophsl_system_wide_to=cophsl_system_wide_to,
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
    stppf_system_wide_from: Union[Unset, float] = UNSET,
    stppf_system_wide_to: Union[Unset, float] = UNSET,
    pvgrpp_system_wide_from: Union[Unset, float] = UNSET,
    pvgrpp_system_wide_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hsl_system_wide_from: Union[Unset, float] = UNSET,
    hsl_system_wide_to: Union[Unset, float] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    gen_system_wide_from: Union[Unset, float] = UNSET,
    gen_system_wide_to: Union[Unset, float] = UNSET,
    cophsl_system_wide_from: Union[Unset, float] = UNSET,
    cophsl_system_wide_to: Union[Unset, float] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """Solar Power Production - Hourly Averaged Actual and Forecasted Values

     Solar Power Production - Hourly Averaged Actual and Forecasted Values

    Args:
        stppf_system_wide_from (Union[Unset, float]):
        stppf_system_wide_to (Union[Unset, float]):
        pvgrpp_system_wide_from (Union[Unset, float]):
        pvgrpp_system_wide_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hsl_system_wide_from (Union[Unset, float]):
        hsl_system_wide_to (Union[Unset, float]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        gen_system_wide_from (Union[Unset, float]):
        gen_system_wide_to (Union[Unset, float]):
        cophsl_system_wide_from (Union[Unset, float]):
        cophsl_system_wide_to (Union[Unset, float]):
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
            stppf_system_wide_from=stppf_system_wide_from,
            stppf_system_wide_to=stppf_system_wide_to,
            pvgrpp_system_wide_from=pvgrpp_system_wide_from,
            pvgrpp_system_wide_to=pvgrpp_system_wide_to,
            dst_flag=dst_flag,
            delivery_date_from=delivery_date_from,
            delivery_date_to=delivery_date_to,
            hsl_system_wide_from=hsl_system_wide_from,
            hsl_system_wide_to=hsl_system_wide_to,
            hour_ending_from=hour_ending_from,
            hour_ending_to=hour_ending_to,
            gen_system_wide_from=gen_system_wide_from,
            gen_system_wide_to=gen_system_wide_to,
            cophsl_system_wide_from=cophsl_system_wide_from,
            cophsl_system_wide_to=cophsl_system_wide_to,
            posted_datetime_from=posted_datetime_from,
            posted_datetime_to=posted_datetime_to,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
