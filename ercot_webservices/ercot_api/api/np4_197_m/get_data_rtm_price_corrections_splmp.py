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
    settlement_point_name: Union[Unset, str] = UNSET,
    lmp_original_from: Union[Unset, float] = UNSET,
    lmp_original_to: Union[Unset, float] = UNSET,
    lmp_corrected_from: Union[Unset, float] = UNSET,
    lmp_corrected_to: Union[Unset, float] = UNSET,
    price_correction_time_from: Union[Unset, str] = UNSET,
    price_correction_time_to: Union[Unset, str] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    sced_timestamp_from: Union[Unset, str] = UNSET,
    sced_timestamp_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["Ocp-Apim-Subscription-Key"] = ocp_apim_subscription_key

    params: dict[str, Any] = {}

    params["settlementPointName"] = settlement_point_name

    params["LMPOriginalFrom"] = lmp_original_from

    params["LMPOriginalTo"] = lmp_original_to

    params["LMPCorrectedFrom"] = lmp_corrected_from

    params["LMPCorrectedTo"] = lmp_corrected_to

    params["priceCorrectionTimeFrom"] = price_correction_time_from

    params["priceCorrectionTimeTo"] = price_correction_time_to

    params["DSTFlag"] = dst_flag

    params["SCEDTimestampFrom"] = sced_timestamp_from

    params["SCEDTimestampTo"] = sced_timestamp_to

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np4-197-m/rtm_price_corrections_splmp",
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
    settlement_point_name: Union[Unset, str] = UNSET,
    lmp_original_from: Union[Unset, float] = UNSET,
    lmp_original_to: Union[Unset, float] = UNSET,
    lmp_corrected_from: Union[Unset, float] = UNSET,
    lmp_corrected_to: Union[Unset, float] = UNSET,
    price_correction_time_from: Union[Unset, str] = UNSET,
    price_correction_time_to: Union[Unset, str] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    sced_timestamp_from: Union[Unset, str] = UNSET,
    sced_timestamp_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """RTM Price Corrections SP LMP

     RTM Price Corrections SP LMP

    Args:
        settlement_point_name (Union[Unset, str]):
        lmp_original_from (Union[Unset, float]):
        lmp_original_to (Union[Unset, float]):
        lmp_corrected_from (Union[Unset, float]):
        lmp_corrected_to (Union[Unset, float]):
        price_correction_time_from (Union[Unset, str]):
        price_correction_time_to (Union[Unset, str]):
        dst_flag (Union[Unset, bool]):
        sced_timestamp_from (Union[Unset, str]):
        sced_timestamp_to (Union[Unset, str]):
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
        settlement_point_name=settlement_point_name,
        lmp_original_from=lmp_original_from,
        lmp_original_to=lmp_original_to,
        lmp_corrected_from=lmp_corrected_from,
        lmp_corrected_to=lmp_corrected_to,
        price_correction_time_from=price_correction_time_from,
        price_correction_time_to=price_correction_time_to,
        dst_flag=dst_flag,
        sced_timestamp_from=sced_timestamp_from,
        sced_timestamp_to=sced_timestamp_to,
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
    settlement_point_name: Union[Unset, str] = UNSET,
    lmp_original_from: Union[Unset, float] = UNSET,
    lmp_original_to: Union[Unset, float] = UNSET,
    lmp_corrected_from: Union[Unset, float] = UNSET,
    lmp_corrected_to: Union[Unset, float] = UNSET,
    price_correction_time_from: Union[Unset, str] = UNSET,
    price_correction_time_to: Union[Unset, str] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    sced_timestamp_from: Union[Unset, str] = UNSET,
    sced_timestamp_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """RTM Price Corrections SP LMP

     RTM Price Corrections SP LMP

    Args:
        settlement_point_name (Union[Unset, str]):
        lmp_original_from (Union[Unset, float]):
        lmp_original_to (Union[Unset, float]):
        lmp_corrected_from (Union[Unset, float]):
        lmp_corrected_to (Union[Unset, float]):
        price_correction_time_from (Union[Unset, str]):
        price_correction_time_to (Union[Unset, str]):
        dst_flag (Union[Unset, bool]):
        sced_timestamp_from (Union[Unset, str]):
        sced_timestamp_to (Union[Unset, str]):
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
        settlement_point_name=settlement_point_name,
        lmp_original_from=lmp_original_from,
        lmp_original_to=lmp_original_to,
        lmp_corrected_from=lmp_corrected_from,
        lmp_corrected_to=lmp_corrected_to,
        price_correction_time_from=price_correction_time_from,
        price_correction_time_to=price_correction_time_to,
        dst_flag=dst_flag,
        sced_timestamp_from=sced_timestamp_from,
        sced_timestamp_to=sced_timestamp_to,
        page=page,
        size=size,
        sort=sort,
        dir_=dir_,
        ocp_apim_subscription_key=ocp_apim_subscription_key,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    settlement_point_name: Union[Unset, str] = UNSET,
    lmp_original_from: Union[Unset, float] = UNSET,
    lmp_original_to: Union[Unset, float] = UNSET,
    lmp_corrected_from: Union[Unset, float] = UNSET,
    lmp_corrected_to: Union[Unset, float] = UNSET,
    price_correction_time_from: Union[Unset, str] = UNSET,
    price_correction_time_to: Union[Unset, str] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    sced_timestamp_from: Union[Unset, str] = UNSET,
    sced_timestamp_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """RTM Price Corrections SP LMP

     RTM Price Corrections SP LMP

    Args:
        settlement_point_name (Union[Unset, str]):
        lmp_original_from (Union[Unset, float]):
        lmp_original_to (Union[Unset, float]):
        lmp_corrected_from (Union[Unset, float]):
        lmp_corrected_to (Union[Unset, float]):
        price_correction_time_from (Union[Unset, str]):
        price_correction_time_to (Union[Unset, str]):
        dst_flag (Union[Unset, bool]):
        sced_timestamp_from (Union[Unset, str]):
        sced_timestamp_to (Union[Unset, str]):
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
        settlement_point_name=settlement_point_name,
        lmp_original_from=lmp_original_from,
        lmp_original_to=lmp_original_to,
        lmp_corrected_from=lmp_corrected_from,
        lmp_corrected_to=lmp_corrected_to,
        price_correction_time_from=price_correction_time_from,
        price_correction_time_to=price_correction_time_to,
        dst_flag=dst_flag,
        sced_timestamp_from=sced_timestamp_from,
        sced_timestamp_to=sced_timestamp_to,
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
    settlement_point_name: Union[Unset, str] = UNSET,
    lmp_original_from: Union[Unset, float] = UNSET,
    lmp_original_to: Union[Unset, float] = UNSET,
    lmp_corrected_from: Union[Unset, float] = UNSET,
    lmp_corrected_to: Union[Unset, float] = UNSET,
    price_correction_time_from: Union[Unset, str] = UNSET,
    price_correction_time_to: Union[Unset, str] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    sced_timestamp_from: Union[Unset, str] = UNSET,
    sced_timestamp_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """RTM Price Corrections SP LMP

     RTM Price Corrections SP LMP

    Args:
        settlement_point_name (Union[Unset, str]):
        lmp_original_from (Union[Unset, float]):
        lmp_original_to (Union[Unset, float]):
        lmp_corrected_from (Union[Unset, float]):
        lmp_corrected_to (Union[Unset, float]):
        price_correction_time_from (Union[Unset, str]):
        price_correction_time_to (Union[Unset, str]):
        dst_flag (Union[Unset, bool]):
        sced_timestamp_from (Union[Unset, str]):
        sced_timestamp_to (Union[Unset, str]):
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
            settlement_point_name=settlement_point_name,
            lmp_original_from=lmp_original_from,
            lmp_original_to=lmp_original_to,
            lmp_corrected_from=lmp_corrected_from,
            lmp_corrected_to=lmp_corrected_to,
            price_correction_time_from=price_correction_time_from,
            price_correction_time_to=price_correction_time_to,
            dst_flag=dst_flag,
            sced_timestamp_from=sced_timestamp_from,
            sced_timestamp_to=sced_timestamp_to,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
