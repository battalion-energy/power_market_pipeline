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
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    settlement_point_source: Union[Unset, str] = UNSET,
    settlement_point_sink: Union[Unset, str] = UNSET,
    mw_from: Union[Unset, float] = UNSET,
    mw_to: Union[Unset, float] = UNSET,
    price_from: Union[Unset, float] = UNSET,
    price_to: Union[Unset, float] = UNSET,
    offer_id: Union[Unset, str] = UNSET,
    multi_hour_block: Union[Unset, bool] = UNSET,
    crr_id: Union[Unset, str] = UNSET,
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

    params["hourEndingFrom"] = hour_ending_from

    params["hourEndingTo"] = hour_ending_to

    params["qseName"] = qse_name

    params["settlementPointSource"] = settlement_point_source

    params["settlementPointSink"] = settlement_point_sink

    params["MWFrom"] = mw_from

    params["MWTo"] = mw_to

    params["priceFrom"] = price_from

    params["priceTo"] = price_to

    params["offerId"] = offer_id

    params["multiHourBlock"] = multi_hour_block

    params["CRRId"] = crr_id

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np3-966-er/60_dam_ptp_obl_opt",
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
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    settlement_point_source: Union[Unset, str] = UNSET,
    settlement_point_sink: Union[Unset, str] = UNSET,
    mw_from: Union[Unset, float] = UNSET,
    mw_to: Union[Unset, float] = UNSET,
    price_from: Union[Unset, float] = UNSET,
    price_to: Union[Unset, float] = UNSET,
    offer_id: Union[Unset, str] = UNSET,
    multi_hour_block: Union[Unset, bool] = UNSET,
    crr_id: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """60-Day DAM PTP Obligation Option

     60-Day DAM PTP Obligation Option

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        qse_name (Union[Unset, str]):
        settlement_point_source (Union[Unset, str]):
        settlement_point_sink (Union[Unset, str]):
        mw_from (Union[Unset, float]):
        mw_to (Union[Unset, float]):
        price_from (Union[Unset, float]):
        price_to (Union[Unset, float]):
        offer_id (Union[Unset, str]):
        multi_hour_block (Union[Unset, bool]):
        crr_id (Union[Unset, str]):
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
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        qse_name=qse_name,
        settlement_point_source=settlement_point_source,
        settlement_point_sink=settlement_point_sink,
        mw_from=mw_from,
        mw_to=mw_to,
        price_from=price_from,
        price_to=price_to,
        offer_id=offer_id,
        multi_hour_block=multi_hour_block,
        crr_id=crr_id,
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
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    settlement_point_source: Union[Unset, str] = UNSET,
    settlement_point_sink: Union[Unset, str] = UNSET,
    mw_from: Union[Unset, float] = UNSET,
    mw_to: Union[Unset, float] = UNSET,
    price_from: Union[Unset, float] = UNSET,
    price_to: Union[Unset, float] = UNSET,
    offer_id: Union[Unset, str] = UNSET,
    multi_hour_block: Union[Unset, bool] = UNSET,
    crr_id: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """60-Day DAM PTP Obligation Option

     60-Day DAM PTP Obligation Option

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        qse_name (Union[Unset, str]):
        settlement_point_source (Union[Unset, str]):
        settlement_point_sink (Union[Unset, str]):
        mw_from (Union[Unset, float]):
        mw_to (Union[Unset, float]):
        price_from (Union[Unset, float]):
        price_to (Union[Unset, float]):
        offer_id (Union[Unset, str]):
        multi_hour_block (Union[Unset, bool]):
        crr_id (Union[Unset, str]):
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
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        qse_name=qse_name,
        settlement_point_source=settlement_point_source,
        settlement_point_sink=settlement_point_sink,
        mw_from=mw_from,
        mw_to=mw_to,
        price_from=price_from,
        price_to=price_to,
        offer_id=offer_id,
        multi_hour_block=multi_hour_block,
        crr_id=crr_id,
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
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    settlement_point_source: Union[Unset, str] = UNSET,
    settlement_point_sink: Union[Unset, str] = UNSET,
    mw_from: Union[Unset, float] = UNSET,
    mw_to: Union[Unset, float] = UNSET,
    price_from: Union[Unset, float] = UNSET,
    price_to: Union[Unset, float] = UNSET,
    offer_id: Union[Unset, str] = UNSET,
    multi_hour_block: Union[Unset, bool] = UNSET,
    crr_id: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """60-Day DAM PTP Obligation Option

     60-Day DAM PTP Obligation Option

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        qse_name (Union[Unset, str]):
        settlement_point_source (Union[Unset, str]):
        settlement_point_sink (Union[Unset, str]):
        mw_from (Union[Unset, float]):
        mw_to (Union[Unset, float]):
        price_from (Union[Unset, float]):
        price_to (Union[Unset, float]):
        offer_id (Union[Unset, str]):
        multi_hour_block (Union[Unset, bool]):
        crr_id (Union[Unset, str]):
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
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        qse_name=qse_name,
        settlement_point_source=settlement_point_source,
        settlement_point_sink=settlement_point_sink,
        mw_from=mw_from,
        mw_to=mw_to,
        price_from=price_from,
        price_to=price_to,
        offer_id=offer_id,
        multi_hour_block=multi_hour_block,
        crr_id=crr_id,
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
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    settlement_point_source: Union[Unset, str] = UNSET,
    settlement_point_sink: Union[Unset, str] = UNSET,
    mw_from: Union[Unset, float] = UNSET,
    mw_to: Union[Unset, float] = UNSET,
    price_from: Union[Unset, float] = UNSET,
    price_to: Union[Unset, float] = UNSET,
    offer_id: Union[Unset, str] = UNSET,
    multi_hour_block: Union[Unset, bool] = UNSET,
    crr_id: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """60-Day DAM PTP Obligation Option

     60-Day DAM PTP Obligation Option

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        qse_name (Union[Unset, str]):
        settlement_point_source (Union[Unset, str]):
        settlement_point_sink (Union[Unset, str]):
        mw_from (Union[Unset, float]):
        mw_to (Union[Unset, float]):
        price_from (Union[Unset, float]):
        price_to (Union[Unset, float]):
        offer_id (Union[Unset, str]):
        multi_hour_block (Union[Unset, bool]):
        crr_id (Union[Unset, str]):
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
            hour_ending_from=hour_ending_from,
            hour_ending_to=hour_ending_to,
            qse_name=qse_name,
            settlement_point_source=settlement_point_source,
            settlement_point_sink=settlement_point_sink,
            mw_from=mw_from,
            mw_to=mw_to,
            price_from=price_from,
            price_to=price_to,
            offer_id=offer_id,
            multi_hour_block=multi_hour_block,
            crr_id=crr_id,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
