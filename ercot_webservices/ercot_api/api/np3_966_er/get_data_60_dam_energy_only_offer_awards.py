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
    settlement_point_name: Union[Unset, str] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    energy_only_offer_award_in_mw_from: Union[Unset, float] = UNSET,
    energy_only_offer_award_in_mw_to: Union[Unset, float] = UNSET,
    settlement_point_price_from: Union[Unset, float] = UNSET,
    settlement_point_price_to: Union[Unset, float] = UNSET,
    offer_id: Union[Unset, str] = UNSET,
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

    params["settlementPointName"] = settlement_point_name

    params["qseName"] = qse_name

    params["energyOnlyOfferAwardInMWFrom"] = energy_only_offer_award_in_mw_from

    params["energyOnlyOfferAwardInMWTo"] = energy_only_offer_award_in_mw_to

    params["settlementPointPriceFrom"] = settlement_point_price_from

    params["settlementPointPriceTo"] = settlement_point_price_to

    params["offerId"] = offer_id

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np3-966-er/60_dam_energy_only_offer_awards",
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
    settlement_point_name: Union[Unset, str] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    energy_only_offer_award_in_mw_from: Union[Unset, float] = UNSET,
    energy_only_offer_award_in_mw_to: Union[Unset, float] = UNSET,
    settlement_point_price_from: Union[Unset, float] = UNSET,
    settlement_point_price_to: Union[Unset, float] = UNSET,
    offer_id: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """60-Day DAM Energy Offer Only Awards

     60-Day DAM Energy Offer Only Awards

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        settlement_point_name (Union[Unset, str]):
        qse_name (Union[Unset, str]):
        energy_only_offer_award_in_mw_from (Union[Unset, float]):
        energy_only_offer_award_in_mw_to (Union[Unset, float]):
        settlement_point_price_from (Union[Unset, float]):
        settlement_point_price_to (Union[Unset, float]):
        offer_id (Union[Unset, str]):
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
        settlement_point_name=settlement_point_name,
        qse_name=qse_name,
        energy_only_offer_award_in_mw_from=energy_only_offer_award_in_mw_from,
        energy_only_offer_award_in_mw_to=energy_only_offer_award_in_mw_to,
        settlement_point_price_from=settlement_point_price_from,
        settlement_point_price_to=settlement_point_price_to,
        offer_id=offer_id,
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
    settlement_point_name: Union[Unset, str] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    energy_only_offer_award_in_mw_from: Union[Unset, float] = UNSET,
    energy_only_offer_award_in_mw_to: Union[Unset, float] = UNSET,
    settlement_point_price_from: Union[Unset, float] = UNSET,
    settlement_point_price_to: Union[Unset, float] = UNSET,
    offer_id: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """60-Day DAM Energy Offer Only Awards

     60-Day DAM Energy Offer Only Awards

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        settlement_point_name (Union[Unset, str]):
        qse_name (Union[Unset, str]):
        energy_only_offer_award_in_mw_from (Union[Unset, float]):
        energy_only_offer_award_in_mw_to (Union[Unset, float]):
        settlement_point_price_from (Union[Unset, float]):
        settlement_point_price_to (Union[Unset, float]):
        offer_id (Union[Unset, str]):
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
        settlement_point_name=settlement_point_name,
        qse_name=qse_name,
        energy_only_offer_award_in_mw_from=energy_only_offer_award_in_mw_from,
        energy_only_offer_award_in_mw_to=energy_only_offer_award_in_mw_to,
        settlement_point_price_from=settlement_point_price_from,
        settlement_point_price_to=settlement_point_price_to,
        offer_id=offer_id,
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
    settlement_point_name: Union[Unset, str] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    energy_only_offer_award_in_mw_from: Union[Unset, float] = UNSET,
    energy_only_offer_award_in_mw_to: Union[Unset, float] = UNSET,
    settlement_point_price_from: Union[Unset, float] = UNSET,
    settlement_point_price_to: Union[Unset, float] = UNSET,
    offer_id: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """60-Day DAM Energy Offer Only Awards

     60-Day DAM Energy Offer Only Awards

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        settlement_point_name (Union[Unset, str]):
        qse_name (Union[Unset, str]):
        energy_only_offer_award_in_mw_from (Union[Unset, float]):
        energy_only_offer_award_in_mw_to (Union[Unset, float]):
        settlement_point_price_from (Union[Unset, float]):
        settlement_point_price_to (Union[Unset, float]):
        offer_id (Union[Unset, str]):
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
        settlement_point_name=settlement_point_name,
        qse_name=qse_name,
        energy_only_offer_award_in_mw_from=energy_only_offer_award_in_mw_from,
        energy_only_offer_award_in_mw_to=energy_only_offer_award_in_mw_to,
        settlement_point_price_from=settlement_point_price_from,
        settlement_point_price_to=settlement_point_price_to,
        offer_id=offer_id,
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
    settlement_point_name: Union[Unset, str] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    energy_only_offer_award_in_mw_from: Union[Unset, float] = UNSET,
    energy_only_offer_award_in_mw_to: Union[Unset, float] = UNSET,
    settlement_point_price_from: Union[Unset, float] = UNSET,
    settlement_point_price_to: Union[Unset, float] = UNSET,
    offer_id: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """60-Day DAM Energy Offer Only Awards

     60-Day DAM Energy Offer Only Awards

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        settlement_point_name (Union[Unset, str]):
        qse_name (Union[Unset, str]):
        energy_only_offer_award_in_mw_from (Union[Unset, float]):
        energy_only_offer_award_in_mw_to (Union[Unset, float]):
        settlement_point_price_from (Union[Unset, float]):
        settlement_point_price_to (Union[Unset, float]):
        offer_id (Union[Unset, str]):
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
            settlement_point_name=settlement_point_name,
            qse_name=qse_name,
            energy_only_offer_award_in_mw_from=energy_only_offer_award_in_mw_from,
            energy_only_offer_award_in_mw_to=energy_only_offer_award_in_mw_to,
            settlement_point_price_from=settlement_point_price_from,
            settlement_point_price_to=settlement_point_price_to,
            offer_id=offer_id,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
