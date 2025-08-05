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
    energy_only_bid_mw_1_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_1_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_1_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_1_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_2_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_2_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_2_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_2_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_3_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_3_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_3_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_3_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_4_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_4_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_4_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_4_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_5_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_5_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_5_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_5_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_6_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_6_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_6_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_6_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_7_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_7_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_7_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_7_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_8_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_8_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_8_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_8_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_9_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_9_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_9_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_9_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_10_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_10_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_10_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_10_to: Union[Unset, float] = UNSET,
    bid_id_from: Union[Unset, float] = UNSET,
    bid_id_to: Union[Unset, float] = UNSET,
    multi_hour_block: Union[Unset, bool] = UNSET,
    block_curve: Union[Unset, bool] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    settlement_point_name: Union[Unset, str] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["Ocp-Apim-Subscription-Key"] = ocp_apim_subscription_key

    params: dict[str, Any] = {}

    params["energyOnlyBidMw1From"] = energy_only_bid_mw_1_from

    params["energyOnlyBidMw1To"] = energy_only_bid_mw_1_to

    params["energyOnlyBidPrice1From"] = energy_only_bid_price_1_from

    params["energyOnlyBidPrice1To"] = energy_only_bid_price_1_to

    params["energyOnlyBidMw2From"] = energy_only_bid_mw_2_from

    params["energyOnlyBidMw2To"] = energy_only_bid_mw_2_to

    params["energyOnlyBidPrice2From"] = energy_only_bid_price_2_from

    params["energyOnlyBidPrice2To"] = energy_only_bid_price_2_to

    params["energyOnlyBidMw3From"] = energy_only_bid_mw_3_from

    params["energyOnlyBidMw3To"] = energy_only_bid_mw_3_to

    params["energyOnlyBidPrice3From"] = energy_only_bid_price_3_from

    params["energyOnlyBidPrice3To"] = energy_only_bid_price_3_to

    params["energyOnlyBidMw4From"] = energy_only_bid_mw_4_from

    params["energyOnlyBidMw4To"] = energy_only_bid_mw_4_to

    params["energyOnlyBidPrice4From"] = energy_only_bid_price_4_from

    params["energyOnlyBidPrice4To"] = energy_only_bid_price_4_to

    params["energyOnlyBidMw5From"] = energy_only_bid_mw_5_from

    params["energyOnlyBidMw5To"] = energy_only_bid_mw_5_to

    params["energyOnlyBidPrice5From"] = energy_only_bid_price_5_from

    params["energyOnlyBidPrice5To"] = energy_only_bid_price_5_to

    params["energyOnlyBidMw6From"] = energy_only_bid_mw_6_from

    params["energyOnlyBidMw6To"] = energy_only_bid_mw_6_to

    params["energyOnlyBidPrice6From"] = energy_only_bid_price_6_from

    params["energyOnlyBidPrice6To"] = energy_only_bid_price_6_to

    params["energyOnlyBidMw7From"] = energy_only_bid_mw_7_from

    params["energyOnlyBidMw7To"] = energy_only_bid_mw_7_to

    params["energyOnlyBidPrice7From"] = energy_only_bid_price_7_from

    params["energyOnlyBidPrice7To"] = energy_only_bid_price_7_to

    params["energyOnlyBidMw8From"] = energy_only_bid_mw_8_from

    params["energyOnlyBidMw8To"] = energy_only_bid_mw_8_to

    params["energyOnlyBidPrice8From"] = energy_only_bid_price_8_from

    params["energyOnlyBidPrice8To"] = energy_only_bid_price_8_to

    params["energyOnlyBidMw9From"] = energy_only_bid_mw_9_from

    params["energyOnlyBidMw9To"] = energy_only_bid_mw_9_to

    params["energyOnlyBidPrice9From"] = energy_only_bid_price_9_from

    params["energyOnlyBidPrice9To"] = energy_only_bid_price_9_to

    params["energyOnlyBidMw10From"] = energy_only_bid_mw_10_from

    params["energyOnlyBidMw10To"] = energy_only_bid_mw_10_to

    params["energyOnlyBidPrice10From"] = energy_only_bid_price_10_from

    params["energyOnlyBidPrice10To"] = energy_only_bid_price_10_to

    params["bidIdFrom"] = bid_id_from

    params["bidIdTo"] = bid_id_to

    params["multiHourBlock"] = multi_hour_block

    params["blockCurve"] = block_curve

    params["deliveryDateFrom"] = delivery_date_from

    params["deliveryDateTo"] = delivery_date_to

    params["hourEndingFrom"] = hour_ending_from

    params["hourEndingTo"] = hour_ending_to

    params["settlementPointName"] = settlement_point_name

    params["qseName"] = qse_name

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np3-966-er/60_dam_energy_bids",
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
    energy_only_bid_mw_1_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_1_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_1_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_1_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_2_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_2_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_2_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_2_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_3_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_3_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_3_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_3_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_4_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_4_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_4_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_4_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_5_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_5_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_5_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_5_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_6_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_6_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_6_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_6_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_7_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_7_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_7_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_7_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_8_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_8_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_8_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_8_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_9_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_9_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_9_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_9_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_10_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_10_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_10_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_10_to: Union[Unset, float] = UNSET,
    bid_id_from: Union[Unset, float] = UNSET,
    bid_id_to: Union[Unset, float] = UNSET,
    multi_hour_block: Union[Unset, bool] = UNSET,
    block_curve: Union[Unset, bool] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    settlement_point_name: Union[Unset, str] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """60-Day DAM Energy Bids

     60-Day DAM Energy Bids

    Args:
        energy_only_bid_mw_1_from (Union[Unset, float]):
        energy_only_bid_mw_1_to (Union[Unset, float]):
        energy_only_bid_price_1_from (Union[Unset, float]):
        energy_only_bid_price_1_to (Union[Unset, float]):
        energy_only_bid_mw_2_from (Union[Unset, float]):
        energy_only_bid_mw_2_to (Union[Unset, float]):
        energy_only_bid_price_2_from (Union[Unset, float]):
        energy_only_bid_price_2_to (Union[Unset, float]):
        energy_only_bid_mw_3_from (Union[Unset, float]):
        energy_only_bid_mw_3_to (Union[Unset, float]):
        energy_only_bid_price_3_from (Union[Unset, float]):
        energy_only_bid_price_3_to (Union[Unset, float]):
        energy_only_bid_mw_4_from (Union[Unset, float]):
        energy_only_bid_mw_4_to (Union[Unset, float]):
        energy_only_bid_price_4_from (Union[Unset, float]):
        energy_only_bid_price_4_to (Union[Unset, float]):
        energy_only_bid_mw_5_from (Union[Unset, float]):
        energy_only_bid_mw_5_to (Union[Unset, float]):
        energy_only_bid_price_5_from (Union[Unset, float]):
        energy_only_bid_price_5_to (Union[Unset, float]):
        energy_only_bid_mw_6_from (Union[Unset, float]):
        energy_only_bid_mw_6_to (Union[Unset, float]):
        energy_only_bid_price_6_from (Union[Unset, float]):
        energy_only_bid_price_6_to (Union[Unset, float]):
        energy_only_bid_mw_7_from (Union[Unset, float]):
        energy_only_bid_mw_7_to (Union[Unset, float]):
        energy_only_bid_price_7_from (Union[Unset, float]):
        energy_only_bid_price_7_to (Union[Unset, float]):
        energy_only_bid_mw_8_from (Union[Unset, float]):
        energy_only_bid_mw_8_to (Union[Unset, float]):
        energy_only_bid_price_8_from (Union[Unset, float]):
        energy_only_bid_price_8_to (Union[Unset, float]):
        energy_only_bid_mw_9_from (Union[Unset, float]):
        energy_only_bid_mw_9_to (Union[Unset, float]):
        energy_only_bid_price_9_from (Union[Unset, float]):
        energy_only_bid_price_9_to (Union[Unset, float]):
        energy_only_bid_mw_10_from (Union[Unset, float]):
        energy_only_bid_mw_10_to (Union[Unset, float]):
        energy_only_bid_price_10_from (Union[Unset, float]):
        energy_only_bid_price_10_to (Union[Unset, float]):
        bid_id_from (Union[Unset, float]):
        bid_id_to (Union[Unset, float]):
        multi_hour_block (Union[Unset, bool]):
        block_curve (Union[Unset, bool]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        settlement_point_name (Union[Unset, str]):
        qse_name (Union[Unset, str]):
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
        energy_only_bid_mw_1_from=energy_only_bid_mw_1_from,
        energy_only_bid_mw_1_to=energy_only_bid_mw_1_to,
        energy_only_bid_price_1_from=energy_only_bid_price_1_from,
        energy_only_bid_price_1_to=energy_only_bid_price_1_to,
        energy_only_bid_mw_2_from=energy_only_bid_mw_2_from,
        energy_only_bid_mw_2_to=energy_only_bid_mw_2_to,
        energy_only_bid_price_2_from=energy_only_bid_price_2_from,
        energy_only_bid_price_2_to=energy_only_bid_price_2_to,
        energy_only_bid_mw_3_from=energy_only_bid_mw_3_from,
        energy_only_bid_mw_3_to=energy_only_bid_mw_3_to,
        energy_only_bid_price_3_from=energy_only_bid_price_3_from,
        energy_only_bid_price_3_to=energy_only_bid_price_3_to,
        energy_only_bid_mw_4_from=energy_only_bid_mw_4_from,
        energy_only_bid_mw_4_to=energy_only_bid_mw_4_to,
        energy_only_bid_price_4_from=energy_only_bid_price_4_from,
        energy_only_bid_price_4_to=energy_only_bid_price_4_to,
        energy_only_bid_mw_5_from=energy_only_bid_mw_5_from,
        energy_only_bid_mw_5_to=energy_only_bid_mw_5_to,
        energy_only_bid_price_5_from=energy_only_bid_price_5_from,
        energy_only_bid_price_5_to=energy_only_bid_price_5_to,
        energy_only_bid_mw_6_from=energy_only_bid_mw_6_from,
        energy_only_bid_mw_6_to=energy_only_bid_mw_6_to,
        energy_only_bid_price_6_from=energy_only_bid_price_6_from,
        energy_only_bid_price_6_to=energy_only_bid_price_6_to,
        energy_only_bid_mw_7_from=energy_only_bid_mw_7_from,
        energy_only_bid_mw_7_to=energy_only_bid_mw_7_to,
        energy_only_bid_price_7_from=energy_only_bid_price_7_from,
        energy_only_bid_price_7_to=energy_only_bid_price_7_to,
        energy_only_bid_mw_8_from=energy_only_bid_mw_8_from,
        energy_only_bid_mw_8_to=energy_only_bid_mw_8_to,
        energy_only_bid_price_8_from=energy_only_bid_price_8_from,
        energy_only_bid_price_8_to=energy_only_bid_price_8_to,
        energy_only_bid_mw_9_from=energy_only_bid_mw_9_from,
        energy_only_bid_mw_9_to=energy_only_bid_mw_9_to,
        energy_only_bid_price_9_from=energy_only_bid_price_9_from,
        energy_only_bid_price_9_to=energy_only_bid_price_9_to,
        energy_only_bid_mw_10_from=energy_only_bid_mw_10_from,
        energy_only_bid_mw_10_to=energy_only_bid_mw_10_to,
        energy_only_bid_price_10_from=energy_only_bid_price_10_from,
        energy_only_bid_price_10_to=energy_only_bid_price_10_to,
        bid_id_from=bid_id_from,
        bid_id_to=bid_id_to,
        multi_hour_block=multi_hour_block,
        block_curve=block_curve,
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        settlement_point_name=settlement_point_name,
        qse_name=qse_name,
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
    energy_only_bid_mw_1_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_1_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_1_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_1_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_2_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_2_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_2_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_2_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_3_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_3_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_3_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_3_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_4_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_4_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_4_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_4_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_5_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_5_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_5_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_5_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_6_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_6_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_6_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_6_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_7_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_7_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_7_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_7_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_8_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_8_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_8_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_8_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_9_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_9_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_9_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_9_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_10_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_10_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_10_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_10_to: Union[Unset, float] = UNSET,
    bid_id_from: Union[Unset, float] = UNSET,
    bid_id_to: Union[Unset, float] = UNSET,
    multi_hour_block: Union[Unset, bool] = UNSET,
    block_curve: Union[Unset, bool] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    settlement_point_name: Union[Unset, str] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """60-Day DAM Energy Bids

     60-Day DAM Energy Bids

    Args:
        energy_only_bid_mw_1_from (Union[Unset, float]):
        energy_only_bid_mw_1_to (Union[Unset, float]):
        energy_only_bid_price_1_from (Union[Unset, float]):
        energy_only_bid_price_1_to (Union[Unset, float]):
        energy_only_bid_mw_2_from (Union[Unset, float]):
        energy_only_bid_mw_2_to (Union[Unset, float]):
        energy_only_bid_price_2_from (Union[Unset, float]):
        energy_only_bid_price_2_to (Union[Unset, float]):
        energy_only_bid_mw_3_from (Union[Unset, float]):
        energy_only_bid_mw_3_to (Union[Unset, float]):
        energy_only_bid_price_3_from (Union[Unset, float]):
        energy_only_bid_price_3_to (Union[Unset, float]):
        energy_only_bid_mw_4_from (Union[Unset, float]):
        energy_only_bid_mw_4_to (Union[Unset, float]):
        energy_only_bid_price_4_from (Union[Unset, float]):
        energy_only_bid_price_4_to (Union[Unset, float]):
        energy_only_bid_mw_5_from (Union[Unset, float]):
        energy_only_bid_mw_5_to (Union[Unset, float]):
        energy_only_bid_price_5_from (Union[Unset, float]):
        energy_only_bid_price_5_to (Union[Unset, float]):
        energy_only_bid_mw_6_from (Union[Unset, float]):
        energy_only_bid_mw_6_to (Union[Unset, float]):
        energy_only_bid_price_6_from (Union[Unset, float]):
        energy_only_bid_price_6_to (Union[Unset, float]):
        energy_only_bid_mw_7_from (Union[Unset, float]):
        energy_only_bid_mw_7_to (Union[Unset, float]):
        energy_only_bid_price_7_from (Union[Unset, float]):
        energy_only_bid_price_7_to (Union[Unset, float]):
        energy_only_bid_mw_8_from (Union[Unset, float]):
        energy_only_bid_mw_8_to (Union[Unset, float]):
        energy_only_bid_price_8_from (Union[Unset, float]):
        energy_only_bid_price_8_to (Union[Unset, float]):
        energy_only_bid_mw_9_from (Union[Unset, float]):
        energy_only_bid_mw_9_to (Union[Unset, float]):
        energy_only_bid_price_9_from (Union[Unset, float]):
        energy_only_bid_price_9_to (Union[Unset, float]):
        energy_only_bid_mw_10_from (Union[Unset, float]):
        energy_only_bid_mw_10_to (Union[Unset, float]):
        energy_only_bid_price_10_from (Union[Unset, float]):
        energy_only_bid_price_10_to (Union[Unset, float]):
        bid_id_from (Union[Unset, float]):
        bid_id_to (Union[Unset, float]):
        multi_hour_block (Union[Unset, bool]):
        block_curve (Union[Unset, bool]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        settlement_point_name (Union[Unset, str]):
        qse_name (Union[Unset, str]):
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
        energy_only_bid_mw_1_from=energy_only_bid_mw_1_from,
        energy_only_bid_mw_1_to=energy_only_bid_mw_1_to,
        energy_only_bid_price_1_from=energy_only_bid_price_1_from,
        energy_only_bid_price_1_to=energy_only_bid_price_1_to,
        energy_only_bid_mw_2_from=energy_only_bid_mw_2_from,
        energy_only_bid_mw_2_to=energy_only_bid_mw_2_to,
        energy_only_bid_price_2_from=energy_only_bid_price_2_from,
        energy_only_bid_price_2_to=energy_only_bid_price_2_to,
        energy_only_bid_mw_3_from=energy_only_bid_mw_3_from,
        energy_only_bid_mw_3_to=energy_only_bid_mw_3_to,
        energy_only_bid_price_3_from=energy_only_bid_price_3_from,
        energy_only_bid_price_3_to=energy_only_bid_price_3_to,
        energy_only_bid_mw_4_from=energy_only_bid_mw_4_from,
        energy_only_bid_mw_4_to=energy_only_bid_mw_4_to,
        energy_only_bid_price_4_from=energy_only_bid_price_4_from,
        energy_only_bid_price_4_to=energy_only_bid_price_4_to,
        energy_only_bid_mw_5_from=energy_only_bid_mw_5_from,
        energy_only_bid_mw_5_to=energy_only_bid_mw_5_to,
        energy_only_bid_price_5_from=energy_only_bid_price_5_from,
        energy_only_bid_price_5_to=energy_only_bid_price_5_to,
        energy_only_bid_mw_6_from=energy_only_bid_mw_6_from,
        energy_only_bid_mw_6_to=energy_only_bid_mw_6_to,
        energy_only_bid_price_6_from=energy_only_bid_price_6_from,
        energy_only_bid_price_6_to=energy_only_bid_price_6_to,
        energy_only_bid_mw_7_from=energy_only_bid_mw_7_from,
        energy_only_bid_mw_7_to=energy_only_bid_mw_7_to,
        energy_only_bid_price_7_from=energy_only_bid_price_7_from,
        energy_only_bid_price_7_to=energy_only_bid_price_7_to,
        energy_only_bid_mw_8_from=energy_only_bid_mw_8_from,
        energy_only_bid_mw_8_to=energy_only_bid_mw_8_to,
        energy_only_bid_price_8_from=energy_only_bid_price_8_from,
        energy_only_bid_price_8_to=energy_only_bid_price_8_to,
        energy_only_bid_mw_9_from=energy_only_bid_mw_9_from,
        energy_only_bid_mw_9_to=energy_only_bid_mw_9_to,
        energy_only_bid_price_9_from=energy_only_bid_price_9_from,
        energy_only_bid_price_9_to=energy_only_bid_price_9_to,
        energy_only_bid_mw_10_from=energy_only_bid_mw_10_from,
        energy_only_bid_mw_10_to=energy_only_bid_mw_10_to,
        energy_only_bid_price_10_from=energy_only_bid_price_10_from,
        energy_only_bid_price_10_to=energy_only_bid_price_10_to,
        bid_id_from=bid_id_from,
        bid_id_to=bid_id_to,
        multi_hour_block=multi_hour_block,
        block_curve=block_curve,
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        settlement_point_name=settlement_point_name,
        qse_name=qse_name,
        page=page,
        size=size,
        sort=sort,
        dir_=dir_,
        ocp_apim_subscription_key=ocp_apim_subscription_key,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    energy_only_bid_mw_1_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_1_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_1_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_1_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_2_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_2_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_2_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_2_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_3_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_3_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_3_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_3_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_4_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_4_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_4_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_4_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_5_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_5_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_5_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_5_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_6_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_6_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_6_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_6_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_7_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_7_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_7_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_7_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_8_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_8_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_8_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_8_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_9_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_9_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_9_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_9_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_10_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_10_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_10_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_10_to: Union[Unset, float] = UNSET,
    bid_id_from: Union[Unset, float] = UNSET,
    bid_id_to: Union[Unset, float] = UNSET,
    multi_hour_block: Union[Unset, bool] = UNSET,
    block_curve: Union[Unset, bool] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    settlement_point_name: Union[Unset, str] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """60-Day DAM Energy Bids

     60-Day DAM Energy Bids

    Args:
        energy_only_bid_mw_1_from (Union[Unset, float]):
        energy_only_bid_mw_1_to (Union[Unset, float]):
        energy_only_bid_price_1_from (Union[Unset, float]):
        energy_only_bid_price_1_to (Union[Unset, float]):
        energy_only_bid_mw_2_from (Union[Unset, float]):
        energy_only_bid_mw_2_to (Union[Unset, float]):
        energy_only_bid_price_2_from (Union[Unset, float]):
        energy_only_bid_price_2_to (Union[Unset, float]):
        energy_only_bid_mw_3_from (Union[Unset, float]):
        energy_only_bid_mw_3_to (Union[Unset, float]):
        energy_only_bid_price_3_from (Union[Unset, float]):
        energy_only_bid_price_3_to (Union[Unset, float]):
        energy_only_bid_mw_4_from (Union[Unset, float]):
        energy_only_bid_mw_4_to (Union[Unset, float]):
        energy_only_bid_price_4_from (Union[Unset, float]):
        energy_only_bid_price_4_to (Union[Unset, float]):
        energy_only_bid_mw_5_from (Union[Unset, float]):
        energy_only_bid_mw_5_to (Union[Unset, float]):
        energy_only_bid_price_5_from (Union[Unset, float]):
        energy_only_bid_price_5_to (Union[Unset, float]):
        energy_only_bid_mw_6_from (Union[Unset, float]):
        energy_only_bid_mw_6_to (Union[Unset, float]):
        energy_only_bid_price_6_from (Union[Unset, float]):
        energy_only_bid_price_6_to (Union[Unset, float]):
        energy_only_bid_mw_7_from (Union[Unset, float]):
        energy_only_bid_mw_7_to (Union[Unset, float]):
        energy_only_bid_price_7_from (Union[Unset, float]):
        energy_only_bid_price_7_to (Union[Unset, float]):
        energy_only_bid_mw_8_from (Union[Unset, float]):
        energy_only_bid_mw_8_to (Union[Unset, float]):
        energy_only_bid_price_8_from (Union[Unset, float]):
        energy_only_bid_price_8_to (Union[Unset, float]):
        energy_only_bid_mw_9_from (Union[Unset, float]):
        energy_only_bid_mw_9_to (Union[Unset, float]):
        energy_only_bid_price_9_from (Union[Unset, float]):
        energy_only_bid_price_9_to (Union[Unset, float]):
        energy_only_bid_mw_10_from (Union[Unset, float]):
        energy_only_bid_mw_10_to (Union[Unset, float]):
        energy_only_bid_price_10_from (Union[Unset, float]):
        energy_only_bid_price_10_to (Union[Unset, float]):
        bid_id_from (Union[Unset, float]):
        bid_id_to (Union[Unset, float]):
        multi_hour_block (Union[Unset, bool]):
        block_curve (Union[Unset, bool]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        settlement_point_name (Union[Unset, str]):
        qse_name (Union[Unset, str]):
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
        energy_only_bid_mw_1_from=energy_only_bid_mw_1_from,
        energy_only_bid_mw_1_to=energy_only_bid_mw_1_to,
        energy_only_bid_price_1_from=energy_only_bid_price_1_from,
        energy_only_bid_price_1_to=energy_only_bid_price_1_to,
        energy_only_bid_mw_2_from=energy_only_bid_mw_2_from,
        energy_only_bid_mw_2_to=energy_only_bid_mw_2_to,
        energy_only_bid_price_2_from=energy_only_bid_price_2_from,
        energy_only_bid_price_2_to=energy_only_bid_price_2_to,
        energy_only_bid_mw_3_from=energy_only_bid_mw_3_from,
        energy_only_bid_mw_3_to=energy_only_bid_mw_3_to,
        energy_only_bid_price_3_from=energy_only_bid_price_3_from,
        energy_only_bid_price_3_to=energy_only_bid_price_3_to,
        energy_only_bid_mw_4_from=energy_only_bid_mw_4_from,
        energy_only_bid_mw_4_to=energy_only_bid_mw_4_to,
        energy_only_bid_price_4_from=energy_only_bid_price_4_from,
        energy_only_bid_price_4_to=energy_only_bid_price_4_to,
        energy_only_bid_mw_5_from=energy_only_bid_mw_5_from,
        energy_only_bid_mw_5_to=energy_only_bid_mw_5_to,
        energy_only_bid_price_5_from=energy_only_bid_price_5_from,
        energy_only_bid_price_5_to=energy_only_bid_price_5_to,
        energy_only_bid_mw_6_from=energy_only_bid_mw_6_from,
        energy_only_bid_mw_6_to=energy_only_bid_mw_6_to,
        energy_only_bid_price_6_from=energy_only_bid_price_6_from,
        energy_only_bid_price_6_to=energy_only_bid_price_6_to,
        energy_only_bid_mw_7_from=energy_only_bid_mw_7_from,
        energy_only_bid_mw_7_to=energy_only_bid_mw_7_to,
        energy_only_bid_price_7_from=energy_only_bid_price_7_from,
        energy_only_bid_price_7_to=energy_only_bid_price_7_to,
        energy_only_bid_mw_8_from=energy_only_bid_mw_8_from,
        energy_only_bid_mw_8_to=energy_only_bid_mw_8_to,
        energy_only_bid_price_8_from=energy_only_bid_price_8_from,
        energy_only_bid_price_8_to=energy_only_bid_price_8_to,
        energy_only_bid_mw_9_from=energy_only_bid_mw_9_from,
        energy_only_bid_mw_9_to=energy_only_bid_mw_9_to,
        energy_only_bid_price_9_from=energy_only_bid_price_9_from,
        energy_only_bid_price_9_to=energy_only_bid_price_9_to,
        energy_only_bid_mw_10_from=energy_only_bid_mw_10_from,
        energy_only_bid_mw_10_to=energy_only_bid_mw_10_to,
        energy_only_bid_price_10_from=energy_only_bid_price_10_from,
        energy_only_bid_price_10_to=energy_only_bid_price_10_to,
        bid_id_from=bid_id_from,
        bid_id_to=bid_id_to,
        multi_hour_block=multi_hour_block,
        block_curve=block_curve,
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        settlement_point_name=settlement_point_name,
        qse_name=qse_name,
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
    energy_only_bid_mw_1_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_1_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_1_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_1_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_2_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_2_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_2_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_2_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_3_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_3_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_3_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_3_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_4_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_4_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_4_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_4_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_5_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_5_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_5_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_5_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_6_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_6_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_6_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_6_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_7_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_7_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_7_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_7_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_8_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_8_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_8_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_8_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_9_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_9_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_9_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_9_to: Union[Unset, float] = UNSET,
    energy_only_bid_mw_10_from: Union[Unset, float] = UNSET,
    energy_only_bid_mw_10_to: Union[Unset, float] = UNSET,
    energy_only_bid_price_10_from: Union[Unset, float] = UNSET,
    energy_only_bid_price_10_to: Union[Unset, float] = UNSET,
    bid_id_from: Union[Unset, float] = UNSET,
    bid_id_to: Union[Unset, float] = UNSET,
    multi_hour_block: Union[Unset, bool] = UNSET,
    block_curve: Union[Unset, bool] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    settlement_point_name: Union[Unset, str] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """60-Day DAM Energy Bids

     60-Day DAM Energy Bids

    Args:
        energy_only_bid_mw_1_from (Union[Unset, float]):
        energy_only_bid_mw_1_to (Union[Unset, float]):
        energy_only_bid_price_1_from (Union[Unset, float]):
        energy_only_bid_price_1_to (Union[Unset, float]):
        energy_only_bid_mw_2_from (Union[Unset, float]):
        energy_only_bid_mw_2_to (Union[Unset, float]):
        energy_only_bid_price_2_from (Union[Unset, float]):
        energy_only_bid_price_2_to (Union[Unset, float]):
        energy_only_bid_mw_3_from (Union[Unset, float]):
        energy_only_bid_mw_3_to (Union[Unset, float]):
        energy_only_bid_price_3_from (Union[Unset, float]):
        energy_only_bid_price_3_to (Union[Unset, float]):
        energy_only_bid_mw_4_from (Union[Unset, float]):
        energy_only_bid_mw_4_to (Union[Unset, float]):
        energy_only_bid_price_4_from (Union[Unset, float]):
        energy_only_bid_price_4_to (Union[Unset, float]):
        energy_only_bid_mw_5_from (Union[Unset, float]):
        energy_only_bid_mw_5_to (Union[Unset, float]):
        energy_only_bid_price_5_from (Union[Unset, float]):
        energy_only_bid_price_5_to (Union[Unset, float]):
        energy_only_bid_mw_6_from (Union[Unset, float]):
        energy_only_bid_mw_6_to (Union[Unset, float]):
        energy_only_bid_price_6_from (Union[Unset, float]):
        energy_only_bid_price_6_to (Union[Unset, float]):
        energy_only_bid_mw_7_from (Union[Unset, float]):
        energy_only_bid_mw_7_to (Union[Unset, float]):
        energy_only_bid_price_7_from (Union[Unset, float]):
        energy_only_bid_price_7_to (Union[Unset, float]):
        energy_only_bid_mw_8_from (Union[Unset, float]):
        energy_only_bid_mw_8_to (Union[Unset, float]):
        energy_only_bid_price_8_from (Union[Unset, float]):
        energy_only_bid_price_8_to (Union[Unset, float]):
        energy_only_bid_mw_9_from (Union[Unset, float]):
        energy_only_bid_mw_9_to (Union[Unset, float]):
        energy_only_bid_price_9_from (Union[Unset, float]):
        energy_only_bid_price_9_to (Union[Unset, float]):
        energy_only_bid_mw_10_from (Union[Unset, float]):
        energy_only_bid_mw_10_to (Union[Unset, float]):
        energy_only_bid_price_10_from (Union[Unset, float]):
        energy_only_bid_price_10_to (Union[Unset, float]):
        bid_id_from (Union[Unset, float]):
        bid_id_to (Union[Unset, float]):
        multi_hour_block (Union[Unset, bool]):
        block_curve (Union[Unset, bool]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        settlement_point_name (Union[Unset, str]):
        qse_name (Union[Unset, str]):
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
            energy_only_bid_mw_1_from=energy_only_bid_mw_1_from,
            energy_only_bid_mw_1_to=energy_only_bid_mw_1_to,
            energy_only_bid_price_1_from=energy_only_bid_price_1_from,
            energy_only_bid_price_1_to=energy_only_bid_price_1_to,
            energy_only_bid_mw_2_from=energy_only_bid_mw_2_from,
            energy_only_bid_mw_2_to=energy_only_bid_mw_2_to,
            energy_only_bid_price_2_from=energy_only_bid_price_2_from,
            energy_only_bid_price_2_to=energy_only_bid_price_2_to,
            energy_only_bid_mw_3_from=energy_only_bid_mw_3_from,
            energy_only_bid_mw_3_to=energy_only_bid_mw_3_to,
            energy_only_bid_price_3_from=energy_only_bid_price_3_from,
            energy_only_bid_price_3_to=energy_only_bid_price_3_to,
            energy_only_bid_mw_4_from=energy_only_bid_mw_4_from,
            energy_only_bid_mw_4_to=energy_only_bid_mw_4_to,
            energy_only_bid_price_4_from=energy_only_bid_price_4_from,
            energy_only_bid_price_4_to=energy_only_bid_price_4_to,
            energy_only_bid_mw_5_from=energy_only_bid_mw_5_from,
            energy_only_bid_mw_5_to=energy_only_bid_mw_5_to,
            energy_only_bid_price_5_from=energy_only_bid_price_5_from,
            energy_only_bid_price_5_to=energy_only_bid_price_5_to,
            energy_only_bid_mw_6_from=energy_only_bid_mw_6_from,
            energy_only_bid_mw_6_to=energy_only_bid_mw_6_to,
            energy_only_bid_price_6_from=energy_only_bid_price_6_from,
            energy_only_bid_price_6_to=energy_only_bid_price_6_to,
            energy_only_bid_mw_7_from=energy_only_bid_mw_7_from,
            energy_only_bid_mw_7_to=energy_only_bid_mw_7_to,
            energy_only_bid_price_7_from=energy_only_bid_price_7_from,
            energy_only_bid_price_7_to=energy_only_bid_price_7_to,
            energy_only_bid_mw_8_from=energy_only_bid_mw_8_from,
            energy_only_bid_mw_8_to=energy_only_bid_mw_8_to,
            energy_only_bid_price_8_from=energy_only_bid_price_8_from,
            energy_only_bid_price_8_to=energy_only_bid_price_8_to,
            energy_only_bid_mw_9_from=energy_only_bid_mw_9_from,
            energy_only_bid_mw_9_to=energy_only_bid_mw_9_to,
            energy_only_bid_price_9_from=energy_only_bid_price_9_from,
            energy_only_bid_price_9_to=energy_only_bid_price_9_to,
            energy_only_bid_mw_10_from=energy_only_bid_mw_10_from,
            energy_only_bid_mw_10_to=energy_only_bid_mw_10_to,
            energy_only_bid_price_10_from=energy_only_bid_price_10_from,
            energy_only_bid_price_10_to=energy_only_bid_price_10_to,
            bid_id_from=bid_id_from,
            bid_id_to=bid_id_to,
            multi_hour_block=multi_hour_block,
            block_curve=block_curve,
            delivery_date_from=delivery_date_from,
            delivery_date_to=delivery_date_to,
            hour_ending_from=hour_ending_from,
            hour_ending_to=hour_ending_to,
            settlement_point_name=settlement_point_name,
            qse_name=qse_name,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
