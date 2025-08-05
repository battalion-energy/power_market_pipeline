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
    price_2rrsffr_from: Union[Unset, float] = UNSET,
    price_2rrsffr_to: Union[Unset, float] = UNSET,
    price_2rrsufr_from: Union[Unset, float] = UNSET,
    price_2rrsufr_to: Union[Unset, float] = UNSET,
    price_2_online_nspin_from: Union[Unset, float] = UNSET,
    price_2_online_nspin_to: Union[Unset, float] = UNSET,
    price_2regup_from: Union[Unset, float] = UNSET,
    price_2regup_to: Union[Unset, float] = UNSET,
    price_2regdn_from: Union[Unset, float] = UNSET,
    price_2regdn_to: Union[Unset, float] = UNSET,
    price_2_offline_nspin_from: Union[Unset, float] = UNSET,
    price_2_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw2_from: Union[Unset, float] = UNSET,
    quantity_mw2_to: Union[Unset, float] = UNSET,
    block_indicator_3: Union[Unset, bool] = UNSET,
    price_3rrspfr_from: Union[Unset, float] = UNSET,
    price_3rrspfr_to: Union[Unset, float] = UNSET,
    price_3rrsffr_from: Union[Unset, float] = UNSET,
    price_3rrsffr_to: Union[Unset, float] = UNSET,
    price_3rrsufr_from: Union[Unset, float] = UNSET,
    price_3rrsufr_to: Union[Unset, float] = UNSET,
    price_3_online_nspin_from: Union[Unset, float] = UNSET,
    price_3_online_nspin_to: Union[Unset, float] = UNSET,
    price_3regup_from: Union[Unset, float] = UNSET,
    price_3regup_to: Union[Unset, float] = UNSET,
    price_3regdn_from: Union[Unset, float] = UNSET,
    price_3regdn_to: Union[Unset, float] = UNSET,
    price_3_offline_nspin_from: Union[Unset, float] = UNSET,
    price_3_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw3_from: Union[Unset, float] = UNSET,
    quantity_mw3_to: Union[Unset, float] = UNSET,
    block_indicator_4: Union[Unset, bool] = UNSET,
    price_4rrspfr_from: Union[Unset, float] = UNSET,
    price_4rrspfr_to: Union[Unset, float] = UNSET,
    price_4rrsffr_from: Union[Unset, float] = UNSET,
    price_4rrsffr_to: Union[Unset, float] = UNSET,
    price_4rrsufr_from: Union[Unset, float] = UNSET,
    price_4rrsufr_to: Union[Unset, float] = UNSET,
    price_4_online_nspin_from: Union[Unset, float] = UNSET,
    price_4_online_nspin_to: Union[Unset, float] = UNSET,
    price_4regup_from: Union[Unset, float] = UNSET,
    price_4regup_to: Union[Unset, float] = UNSET,
    price_4regdn_from: Union[Unset, float] = UNSET,
    price_4regdn_to: Union[Unset, float] = UNSET,
    price_4_offline_nspin_from: Union[Unset, float] = UNSET,
    price_4_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw4_from: Union[Unset, float] = UNSET,
    quantity_mw4_to: Union[Unset, float] = UNSET,
    block_indicator_5: Union[Unset, bool] = UNSET,
    price_5rrspfr_from: Union[Unset, float] = UNSET,
    price_5rrspfr_to: Union[Unset, float] = UNSET,
    price_5rrsffr_from: Union[Unset, float] = UNSET,
    price_5rrsffr_to: Union[Unset, float] = UNSET,
    price_5rrsufr_from: Union[Unset, float] = UNSET,
    price_5rrsufr_to: Union[Unset, float] = UNSET,
    price_5_online_nspin_from: Union[Unset, float] = UNSET,
    price_5_online_nspin_to: Union[Unset, float] = UNSET,
    price_5regup_from: Union[Unset, float] = UNSET,
    price_5regup_to: Union[Unset, float] = UNSET,
    price_5regdn_from: Union[Unset, float] = UNSET,
    price_5regdn_to: Union[Unset, float] = UNSET,
    price_5_offline_nspin_from: Union[Unset, float] = UNSET,
    price_5_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw5_from: Union[Unset, float] = UNSET,
    quantity_mw5_to: Union[Unset, float] = UNSET,
    price_1ecrs_from: Union[Unset, float] = UNSET,
    price_1ecrs_to: Union[Unset, float] = UNSET,
    price_1offec_from: Union[Unset, float] = UNSET,
    price_1offec_to: Union[Unset, float] = UNSET,
    price_2ecrs_from: Union[Unset, float] = UNSET,
    price_2ecrs_to: Union[Unset, float] = UNSET,
    price_2offec_from: Union[Unset, float] = UNSET,
    price_2offec_to: Union[Unset, float] = UNSET,
    price_3ecrs_from: Union[Unset, float] = UNSET,
    price_3ecrs_to: Union[Unset, float] = UNSET,
    price_3offec_from: Union[Unset, float] = UNSET,
    price_3offec_to: Union[Unset, float] = UNSET,
    price_4ecrs_from: Union[Unset, float] = UNSET,
    price_4ecrs_to: Union[Unset, float] = UNSET,
    price_4offec_from: Union[Unset, float] = UNSET,
    price_4offec_to: Union[Unset, float] = UNSET,
    price_5ecrs_from: Union[Unset, float] = UNSET,
    price_5ecrs_to: Union[Unset, float] = UNSET,
    price_5offec_from: Union[Unset, float] = UNSET,
    price_5offec_to: Union[Unset, float] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    dme_name: Union[Unset, str] = UNSET,
    resource_name: Union[Unset, str] = UNSET,
    multi_hour_block: Union[Unset, bool] = UNSET,
    block_indicator_1: Union[Unset, bool] = UNSET,
    price_1rrspfr_from: Union[Unset, float] = UNSET,
    price_1rrspfr_to: Union[Unset, float] = UNSET,
    price_1rrsffr_from: Union[Unset, float] = UNSET,
    price_1rrsffr_to: Union[Unset, float] = UNSET,
    price_1rrsufr_from: Union[Unset, float] = UNSET,
    price_1rrsufr_to: Union[Unset, float] = UNSET,
    price_1_online_nspin_from: Union[Unset, float] = UNSET,
    price_1_online_nspin_to: Union[Unset, float] = UNSET,
    price_1regup_from: Union[Unset, float] = UNSET,
    price_1regup_to: Union[Unset, float] = UNSET,
    price_1regdn_from: Union[Unset, float] = UNSET,
    price_1regdn_to: Union[Unset, float] = UNSET,
    price_1_offline_nspin_from: Union[Unset, float] = UNSET,
    price_1_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw1_from: Union[Unset, float] = UNSET,
    quantity_mw1_to: Union[Unset, float] = UNSET,
    block_indicator_2: Union[Unset, bool] = UNSET,
    price_2rrspfr_from: Union[Unset, float] = UNSET,
    price_2rrspfr_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["Ocp-Apim-Subscription-Key"] = ocp_apim_subscription_key

    params: dict[str, Any] = {}

    params["price2RRSFFRFrom"] = price_2rrsffr_from

    params["price2RRSFFRTo"] = price_2rrsffr_to

    params["price2RRSUFRFrom"] = price_2rrsufr_from

    params["price2RRSUFRTo"] = price_2rrsufr_to

    params["price2OnlineNSPINFrom"] = price_2_online_nspin_from

    params["price2OnlineNSPINTo"] = price_2_online_nspin_to

    params["price2REGUPFrom"] = price_2regup_from

    params["price2REGUPTo"] = price_2regup_to

    params["price2REGDNFrom"] = price_2regdn_from

    params["price2REGDNTo"] = price_2regdn_to

    params["price2OfflineNSPINFrom"] = price_2_offline_nspin_from

    params["price2OfflineNSPINTo"] = price_2_offline_nspin_to

    params["quantityMW2From"] = quantity_mw2_from

    params["quantityMW2To"] = quantity_mw2_to

    params["blockIndicator3"] = block_indicator_3

    params["price3RRSPFRFrom"] = price_3rrspfr_from

    params["price3RRSPFRTo"] = price_3rrspfr_to

    params["price3RRSFFRFrom"] = price_3rrsffr_from

    params["price3RRSFFRTo"] = price_3rrsffr_to

    params["price3RRSUFRFrom"] = price_3rrsufr_from

    params["price3RRSUFRTo"] = price_3rrsufr_to

    params["price3OnlineNSPINFrom"] = price_3_online_nspin_from

    params["price3OnlineNSPINTo"] = price_3_online_nspin_to

    params["price3REGUPFrom"] = price_3regup_from

    params["price3REGUPTo"] = price_3regup_to

    params["price3REGDNFrom"] = price_3regdn_from

    params["price3REGDNTo"] = price_3regdn_to

    params["price3OfflineNSPINFrom"] = price_3_offline_nspin_from

    params["price3OfflineNSPINTo"] = price_3_offline_nspin_to

    params["quantityMW3From"] = quantity_mw3_from

    params["quantityMW3To"] = quantity_mw3_to

    params["blockIndicator4"] = block_indicator_4

    params["price4RRSPFRFrom"] = price_4rrspfr_from

    params["price4RRSPFRTo"] = price_4rrspfr_to

    params["price4RRSFFRFrom"] = price_4rrsffr_from

    params["price4RRSFFRTo"] = price_4rrsffr_to

    params["price4RRSUFRFrom"] = price_4rrsufr_from

    params["price4RRSUFRTo"] = price_4rrsufr_to

    params["price4OnlineNSPINFrom"] = price_4_online_nspin_from

    params["price4OnlineNSPINTo"] = price_4_online_nspin_to

    params["price4REGUPFrom"] = price_4regup_from

    params["price4REGUPTo"] = price_4regup_to

    params["price4REGDNFrom"] = price_4regdn_from

    params["price4REGDNTo"] = price_4regdn_to

    params["price4OfflineNSPINFrom"] = price_4_offline_nspin_from

    params["price4OfflineNSPINTo"] = price_4_offline_nspin_to

    params["quantityMW4From"] = quantity_mw4_from

    params["quantityMW4To"] = quantity_mw4_to

    params["blockIndicator5"] = block_indicator_5

    params["price5RRSPFRFrom"] = price_5rrspfr_from

    params["price5RRSPFRTo"] = price_5rrspfr_to

    params["price5RRSFFRFrom"] = price_5rrsffr_from

    params["price5RRSFFRTo"] = price_5rrsffr_to

    params["price5RRSUFRFrom"] = price_5rrsufr_from

    params["price5RRSUFRTo"] = price_5rrsufr_to

    params["price5OnlineNSPINFrom"] = price_5_online_nspin_from

    params["price5OnlineNSPINTo"] = price_5_online_nspin_to

    params["price5REGUPFrom"] = price_5regup_from

    params["price5REGUPTo"] = price_5regup_to

    params["price5REGDNFrom"] = price_5regdn_from

    params["price5REGDNTo"] = price_5regdn_to

    params["price5OfflineNSPINFrom"] = price_5_offline_nspin_from

    params["price5OfflineNSPINTo"] = price_5_offline_nspin_to

    params["quantityMW5From"] = quantity_mw5_from

    params["quantityMW5To"] = quantity_mw5_to

    params["price1ECRSFrom"] = price_1ecrs_from

    params["price1ECRSTo"] = price_1ecrs_to

    params["price1OFFECFrom"] = price_1offec_from

    params["price1OFFECTo"] = price_1offec_to

    params["price2ECRSFrom"] = price_2ecrs_from

    params["price2ECRSTo"] = price_2ecrs_to

    params["price2OFFECFrom"] = price_2offec_from

    params["price2OFFECTo"] = price_2offec_to

    params["price3ECRSFrom"] = price_3ecrs_from

    params["price3ECRSTo"] = price_3ecrs_to

    params["price3OFFECFrom"] = price_3offec_from

    params["price3OFFECTo"] = price_3offec_to

    params["price4ECRSFrom"] = price_4ecrs_from

    params["price4ECRSTo"] = price_4ecrs_to

    params["price4OFFECFrom"] = price_4offec_from

    params["price4OFFECTo"] = price_4offec_to

    params["price5ECRSFrom"] = price_5ecrs_from

    params["price5ECRSTo"] = price_5ecrs_to

    params["price5OFFECFrom"] = price_5offec_from

    params["price5OFFECTo"] = price_5offec_to

    params["deliveryDateFrom"] = delivery_date_from

    params["deliveryDateTo"] = delivery_date_to

    params["hourEndingFrom"] = hour_ending_from

    params["hourEndingTo"] = hour_ending_to

    params["qseName"] = qse_name

    params["dmeName"] = dme_name

    params["resourceName"] = resource_name

    params["multiHourBlock"] = multi_hour_block

    params["blockIndicator1"] = block_indicator_1

    params["price1RRSPFRFrom"] = price_1rrspfr_from

    params["price1RRSPFRTo"] = price_1rrspfr_to

    params["price1RRSFFRFrom"] = price_1rrsffr_from

    params["price1RRSFFRTo"] = price_1rrsffr_to

    params["price1RRSUFRFrom"] = price_1rrsufr_from

    params["price1RRSUFRTo"] = price_1rrsufr_to

    params["price1OnlineNSPINFrom"] = price_1_online_nspin_from

    params["price1OnlineNSPINTo"] = price_1_online_nspin_to

    params["price1REGUPFrom"] = price_1regup_from

    params["price1REGUPTo"] = price_1regup_to

    params["price1REGDNFrom"] = price_1regdn_from

    params["price1REGDNTo"] = price_1regdn_to

    params["price1OfflineNSPINFrom"] = price_1_offline_nspin_from

    params["price1OfflineNSPINTo"] = price_1_offline_nspin_to

    params["quantityMW1From"] = quantity_mw1_from

    params["quantityMW1To"] = quantity_mw1_to

    params["blockIndicator2"] = block_indicator_2

    params["price2RRSPFRFrom"] = price_2rrspfr_from

    params["price2RRSPFRTo"] = price_2rrspfr_to

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np3-966-er/60_dam_gen_res_as_offers",
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
    price_2rrsffr_from: Union[Unset, float] = UNSET,
    price_2rrsffr_to: Union[Unset, float] = UNSET,
    price_2rrsufr_from: Union[Unset, float] = UNSET,
    price_2rrsufr_to: Union[Unset, float] = UNSET,
    price_2_online_nspin_from: Union[Unset, float] = UNSET,
    price_2_online_nspin_to: Union[Unset, float] = UNSET,
    price_2regup_from: Union[Unset, float] = UNSET,
    price_2regup_to: Union[Unset, float] = UNSET,
    price_2regdn_from: Union[Unset, float] = UNSET,
    price_2regdn_to: Union[Unset, float] = UNSET,
    price_2_offline_nspin_from: Union[Unset, float] = UNSET,
    price_2_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw2_from: Union[Unset, float] = UNSET,
    quantity_mw2_to: Union[Unset, float] = UNSET,
    block_indicator_3: Union[Unset, bool] = UNSET,
    price_3rrspfr_from: Union[Unset, float] = UNSET,
    price_3rrspfr_to: Union[Unset, float] = UNSET,
    price_3rrsffr_from: Union[Unset, float] = UNSET,
    price_3rrsffr_to: Union[Unset, float] = UNSET,
    price_3rrsufr_from: Union[Unset, float] = UNSET,
    price_3rrsufr_to: Union[Unset, float] = UNSET,
    price_3_online_nspin_from: Union[Unset, float] = UNSET,
    price_3_online_nspin_to: Union[Unset, float] = UNSET,
    price_3regup_from: Union[Unset, float] = UNSET,
    price_3regup_to: Union[Unset, float] = UNSET,
    price_3regdn_from: Union[Unset, float] = UNSET,
    price_3regdn_to: Union[Unset, float] = UNSET,
    price_3_offline_nspin_from: Union[Unset, float] = UNSET,
    price_3_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw3_from: Union[Unset, float] = UNSET,
    quantity_mw3_to: Union[Unset, float] = UNSET,
    block_indicator_4: Union[Unset, bool] = UNSET,
    price_4rrspfr_from: Union[Unset, float] = UNSET,
    price_4rrspfr_to: Union[Unset, float] = UNSET,
    price_4rrsffr_from: Union[Unset, float] = UNSET,
    price_4rrsffr_to: Union[Unset, float] = UNSET,
    price_4rrsufr_from: Union[Unset, float] = UNSET,
    price_4rrsufr_to: Union[Unset, float] = UNSET,
    price_4_online_nspin_from: Union[Unset, float] = UNSET,
    price_4_online_nspin_to: Union[Unset, float] = UNSET,
    price_4regup_from: Union[Unset, float] = UNSET,
    price_4regup_to: Union[Unset, float] = UNSET,
    price_4regdn_from: Union[Unset, float] = UNSET,
    price_4regdn_to: Union[Unset, float] = UNSET,
    price_4_offline_nspin_from: Union[Unset, float] = UNSET,
    price_4_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw4_from: Union[Unset, float] = UNSET,
    quantity_mw4_to: Union[Unset, float] = UNSET,
    block_indicator_5: Union[Unset, bool] = UNSET,
    price_5rrspfr_from: Union[Unset, float] = UNSET,
    price_5rrspfr_to: Union[Unset, float] = UNSET,
    price_5rrsffr_from: Union[Unset, float] = UNSET,
    price_5rrsffr_to: Union[Unset, float] = UNSET,
    price_5rrsufr_from: Union[Unset, float] = UNSET,
    price_5rrsufr_to: Union[Unset, float] = UNSET,
    price_5_online_nspin_from: Union[Unset, float] = UNSET,
    price_5_online_nspin_to: Union[Unset, float] = UNSET,
    price_5regup_from: Union[Unset, float] = UNSET,
    price_5regup_to: Union[Unset, float] = UNSET,
    price_5regdn_from: Union[Unset, float] = UNSET,
    price_5regdn_to: Union[Unset, float] = UNSET,
    price_5_offline_nspin_from: Union[Unset, float] = UNSET,
    price_5_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw5_from: Union[Unset, float] = UNSET,
    quantity_mw5_to: Union[Unset, float] = UNSET,
    price_1ecrs_from: Union[Unset, float] = UNSET,
    price_1ecrs_to: Union[Unset, float] = UNSET,
    price_1offec_from: Union[Unset, float] = UNSET,
    price_1offec_to: Union[Unset, float] = UNSET,
    price_2ecrs_from: Union[Unset, float] = UNSET,
    price_2ecrs_to: Union[Unset, float] = UNSET,
    price_2offec_from: Union[Unset, float] = UNSET,
    price_2offec_to: Union[Unset, float] = UNSET,
    price_3ecrs_from: Union[Unset, float] = UNSET,
    price_3ecrs_to: Union[Unset, float] = UNSET,
    price_3offec_from: Union[Unset, float] = UNSET,
    price_3offec_to: Union[Unset, float] = UNSET,
    price_4ecrs_from: Union[Unset, float] = UNSET,
    price_4ecrs_to: Union[Unset, float] = UNSET,
    price_4offec_from: Union[Unset, float] = UNSET,
    price_4offec_to: Union[Unset, float] = UNSET,
    price_5ecrs_from: Union[Unset, float] = UNSET,
    price_5ecrs_to: Union[Unset, float] = UNSET,
    price_5offec_from: Union[Unset, float] = UNSET,
    price_5offec_to: Union[Unset, float] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    dme_name: Union[Unset, str] = UNSET,
    resource_name: Union[Unset, str] = UNSET,
    multi_hour_block: Union[Unset, bool] = UNSET,
    block_indicator_1: Union[Unset, bool] = UNSET,
    price_1rrspfr_from: Union[Unset, float] = UNSET,
    price_1rrspfr_to: Union[Unset, float] = UNSET,
    price_1rrsffr_from: Union[Unset, float] = UNSET,
    price_1rrsffr_to: Union[Unset, float] = UNSET,
    price_1rrsufr_from: Union[Unset, float] = UNSET,
    price_1rrsufr_to: Union[Unset, float] = UNSET,
    price_1_online_nspin_from: Union[Unset, float] = UNSET,
    price_1_online_nspin_to: Union[Unset, float] = UNSET,
    price_1regup_from: Union[Unset, float] = UNSET,
    price_1regup_to: Union[Unset, float] = UNSET,
    price_1regdn_from: Union[Unset, float] = UNSET,
    price_1regdn_to: Union[Unset, float] = UNSET,
    price_1_offline_nspin_from: Union[Unset, float] = UNSET,
    price_1_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw1_from: Union[Unset, float] = UNSET,
    quantity_mw1_to: Union[Unset, float] = UNSET,
    block_indicator_2: Union[Unset, bool] = UNSET,
    price_2rrspfr_from: Union[Unset, float] = UNSET,
    price_2rrspfr_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """60-Day DAM Generation Resources AS Offers

     60-Day DAM Generation Resources AS Offers

    Args:
        price_2rrsffr_from (Union[Unset, float]):
        price_2rrsffr_to (Union[Unset, float]):
        price_2rrsufr_from (Union[Unset, float]):
        price_2rrsufr_to (Union[Unset, float]):
        price_2_online_nspin_from (Union[Unset, float]):
        price_2_online_nspin_to (Union[Unset, float]):
        price_2regup_from (Union[Unset, float]):
        price_2regup_to (Union[Unset, float]):
        price_2regdn_from (Union[Unset, float]):
        price_2regdn_to (Union[Unset, float]):
        price_2_offline_nspin_from (Union[Unset, float]):
        price_2_offline_nspin_to (Union[Unset, float]):
        quantity_mw2_from (Union[Unset, float]):
        quantity_mw2_to (Union[Unset, float]):
        block_indicator_3 (Union[Unset, bool]):
        price_3rrspfr_from (Union[Unset, float]):
        price_3rrspfr_to (Union[Unset, float]):
        price_3rrsffr_from (Union[Unset, float]):
        price_3rrsffr_to (Union[Unset, float]):
        price_3rrsufr_from (Union[Unset, float]):
        price_3rrsufr_to (Union[Unset, float]):
        price_3_online_nspin_from (Union[Unset, float]):
        price_3_online_nspin_to (Union[Unset, float]):
        price_3regup_from (Union[Unset, float]):
        price_3regup_to (Union[Unset, float]):
        price_3regdn_from (Union[Unset, float]):
        price_3regdn_to (Union[Unset, float]):
        price_3_offline_nspin_from (Union[Unset, float]):
        price_3_offline_nspin_to (Union[Unset, float]):
        quantity_mw3_from (Union[Unset, float]):
        quantity_mw3_to (Union[Unset, float]):
        block_indicator_4 (Union[Unset, bool]):
        price_4rrspfr_from (Union[Unset, float]):
        price_4rrspfr_to (Union[Unset, float]):
        price_4rrsffr_from (Union[Unset, float]):
        price_4rrsffr_to (Union[Unset, float]):
        price_4rrsufr_from (Union[Unset, float]):
        price_4rrsufr_to (Union[Unset, float]):
        price_4_online_nspin_from (Union[Unset, float]):
        price_4_online_nspin_to (Union[Unset, float]):
        price_4regup_from (Union[Unset, float]):
        price_4regup_to (Union[Unset, float]):
        price_4regdn_from (Union[Unset, float]):
        price_4regdn_to (Union[Unset, float]):
        price_4_offline_nspin_from (Union[Unset, float]):
        price_4_offline_nspin_to (Union[Unset, float]):
        quantity_mw4_from (Union[Unset, float]):
        quantity_mw4_to (Union[Unset, float]):
        block_indicator_5 (Union[Unset, bool]):
        price_5rrspfr_from (Union[Unset, float]):
        price_5rrspfr_to (Union[Unset, float]):
        price_5rrsffr_from (Union[Unset, float]):
        price_5rrsffr_to (Union[Unset, float]):
        price_5rrsufr_from (Union[Unset, float]):
        price_5rrsufr_to (Union[Unset, float]):
        price_5_online_nspin_from (Union[Unset, float]):
        price_5_online_nspin_to (Union[Unset, float]):
        price_5regup_from (Union[Unset, float]):
        price_5regup_to (Union[Unset, float]):
        price_5regdn_from (Union[Unset, float]):
        price_5regdn_to (Union[Unset, float]):
        price_5_offline_nspin_from (Union[Unset, float]):
        price_5_offline_nspin_to (Union[Unset, float]):
        quantity_mw5_from (Union[Unset, float]):
        quantity_mw5_to (Union[Unset, float]):
        price_1ecrs_from (Union[Unset, float]):
        price_1ecrs_to (Union[Unset, float]):
        price_1offec_from (Union[Unset, float]):
        price_1offec_to (Union[Unset, float]):
        price_2ecrs_from (Union[Unset, float]):
        price_2ecrs_to (Union[Unset, float]):
        price_2offec_from (Union[Unset, float]):
        price_2offec_to (Union[Unset, float]):
        price_3ecrs_from (Union[Unset, float]):
        price_3ecrs_to (Union[Unset, float]):
        price_3offec_from (Union[Unset, float]):
        price_3offec_to (Union[Unset, float]):
        price_4ecrs_from (Union[Unset, float]):
        price_4ecrs_to (Union[Unset, float]):
        price_4offec_from (Union[Unset, float]):
        price_4offec_to (Union[Unset, float]):
        price_5ecrs_from (Union[Unset, float]):
        price_5ecrs_to (Union[Unset, float]):
        price_5offec_from (Union[Unset, float]):
        price_5offec_to (Union[Unset, float]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        qse_name (Union[Unset, str]):
        dme_name (Union[Unset, str]):
        resource_name (Union[Unset, str]):
        multi_hour_block (Union[Unset, bool]):
        block_indicator_1 (Union[Unset, bool]):
        price_1rrspfr_from (Union[Unset, float]):
        price_1rrspfr_to (Union[Unset, float]):
        price_1rrsffr_from (Union[Unset, float]):
        price_1rrsffr_to (Union[Unset, float]):
        price_1rrsufr_from (Union[Unset, float]):
        price_1rrsufr_to (Union[Unset, float]):
        price_1_online_nspin_from (Union[Unset, float]):
        price_1_online_nspin_to (Union[Unset, float]):
        price_1regup_from (Union[Unset, float]):
        price_1regup_to (Union[Unset, float]):
        price_1regdn_from (Union[Unset, float]):
        price_1regdn_to (Union[Unset, float]):
        price_1_offline_nspin_from (Union[Unset, float]):
        price_1_offline_nspin_to (Union[Unset, float]):
        quantity_mw1_from (Union[Unset, float]):
        quantity_mw1_to (Union[Unset, float]):
        block_indicator_2 (Union[Unset, bool]):
        price_2rrspfr_from (Union[Unset, float]):
        price_2rrspfr_to (Union[Unset, float]):
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
        price_2rrsffr_from=price_2rrsffr_from,
        price_2rrsffr_to=price_2rrsffr_to,
        price_2rrsufr_from=price_2rrsufr_from,
        price_2rrsufr_to=price_2rrsufr_to,
        price_2_online_nspin_from=price_2_online_nspin_from,
        price_2_online_nspin_to=price_2_online_nspin_to,
        price_2regup_from=price_2regup_from,
        price_2regup_to=price_2regup_to,
        price_2regdn_from=price_2regdn_from,
        price_2regdn_to=price_2regdn_to,
        price_2_offline_nspin_from=price_2_offline_nspin_from,
        price_2_offline_nspin_to=price_2_offline_nspin_to,
        quantity_mw2_from=quantity_mw2_from,
        quantity_mw2_to=quantity_mw2_to,
        block_indicator_3=block_indicator_3,
        price_3rrspfr_from=price_3rrspfr_from,
        price_3rrspfr_to=price_3rrspfr_to,
        price_3rrsffr_from=price_3rrsffr_from,
        price_3rrsffr_to=price_3rrsffr_to,
        price_3rrsufr_from=price_3rrsufr_from,
        price_3rrsufr_to=price_3rrsufr_to,
        price_3_online_nspin_from=price_3_online_nspin_from,
        price_3_online_nspin_to=price_3_online_nspin_to,
        price_3regup_from=price_3regup_from,
        price_3regup_to=price_3regup_to,
        price_3regdn_from=price_3regdn_from,
        price_3regdn_to=price_3regdn_to,
        price_3_offline_nspin_from=price_3_offline_nspin_from,
        price_3_offline_nspin_to=price_3_offline_nspin_to,
        quantity_mw3_from=quantity_mw3_from,
        quantity_mw3_to=quantity_mw3_to,
        block_indicator_4=block_indicator_4,
        price_4rrspfr_from=price_4rrspfr_from,
        price_4rrspfr_to=price_4rrspfr_to,
        price_4rrsffr_from=price_4rrsffr_from,
        price_4rrsffr_to=price_4rrsffr_to,
        price_4rrsufr_from=price_4rrsufr_from,
        price_4rrsufr_to=price_4rrsufr_to,
        price_4_online_nspin_from=price_4_online_nspin_from,
        price_4_online_nspin_to=price_4_online_nspin_to,
        price_4regup_from=price_4regup_from,
        price_4regup_to=price_4regup_to,
        price_4regdn_from=price_4regdn_from,
        price_4regdn_to=price_4regdn_to,
        price_4_offline_nspin_from=price_4_offline_nspin_from,
        price_4_offline_nspin_to=price_4_offline_nspin_to,
        quantity_mw4_from=quantity_mw4_from,
        quantity_mw4_to=quantity_mw4_to,
        block_indicator_5=block_indicator_5,
        price_5rrspfr_from=price_5rrspfr_from,
        price_5rrspfr_to=price_5rrspfr_to,
        price_5rrsffr_from=price_5rrsffr_from,
        price_5rrsffr_to=price_5rrsffr_to,
        price_5rrsufr_from=price_5rrsufr_from,
        price_5rrsufr_to=price_5rrsufr_to,
        price_5_online_nspin_from=price_5_online_nspin_from,
        price_5_online_nspin_to=price_5_online_nspin_to,
        price_5regup_from=price_5regup_from,
        price_5regup_to=price_5regup_to,
        price_5regdn_from=price_5regdn_from,
        price_5regdn_to=price_5regdn_to,
        price_5_offline_nspin_from=price_5_offline_nspin_from,
        price_5_offline_nspin_to=price_5_offline_nspin_to,
        quantity_mw5_from=quantity_mw5_from,
        quantity_mw5_to=quantity_mw5_to,
        price_1ecrs_from=price_1ecrs_from,
        price_1ecrs_to=price_1ecrs_to,
        price_1offec_from=price_1offec_from,
        price_1offec_to=price_1offec_to,
        price_2ecrs_from=price_2ecrs_from,
        price_2ecrs_to=price_2ecrs_to,
        price_2offec_from=price_2offec_from,
        price_2offec_to=price_2offec_to,
        price_3ecrs_from=price_3ecrs_from,
        price_3ecrs_to=price_3ecrs_to,
        price_3offec_from=price_3offec_from,
        price_3offec_to=price_3offec_to,
        price_4ecrs_from=price_4ecrs_from,
        price_4ecrs_to=price_4ecrs_to,
        price_4offec_from=price_4offec_from,
        price_4offec_to=price_4offec_to,
        price_5ecrs_from=price_5ecrs_from,
        price_5ecrs_to=price_5ecrs_to,
        price_5offec_from=price_5offec_from,
        price_5offec_to=price_5offec_to,
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        qse_name=qse_name,
        dme_name=dme_name,
        resource_name=resource_name,
        multi_hour_block=multi_hour_block,
        block_indicator_1=block_indicator_1,
        price_1rrspfr_from=price_1rrspfr_from,
        price_1rrspfr_to=price_1rrspfr_to,
        price_1rrsffr_from=price_1rrsffr_from,
        price_1rrsffr_to=price_1rrsffr_to,
        price_1rrsufr_from=price_1rrsufr_from,
        price_1rrsufr_to=price_1rrsufr_to,
        price_1_online_nspin_from=price_1_online_nspin_from,
        price_1_online_nspin_to=price_1_online_nspin_to,
        price_1regup_from=price_1regup_from,
        price_1regup_to=price_1regup_to,
        price_1regdn_from=price_1regdn_from,
        price_1regdn_to=price_1regdn_to,
        price_1_offline_nspin_from=price_1_offline_nspin_from,
        price_1_offline_nspin_to=price_1_offline_nspin_to,
        quantity_mw1_from=quantity_mw1_from,
        quantity_mw1_to=quantity_mw1_to,
        block_indicator_2=block_indicator_2,
        price_2rrspfr_from=price_2rrspfr_from,
        price_2rrspfr_to=price_2rrspfr_to,
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
    price_2rrsffr_from: Union[Unset, float] = UNSET,
    price_2rrsffr_to: Union[Unset, float] = UNSET,
    price_2rrsufr_from: Union[Unset, float] = UNSET,
    price_2rrsufr_to: Union[Unset, float] = UNSET,
    price_2_online_nspin_from: Union[Unset, float] = UNSET,
    price_2_online_nspin_to: Union[Unset, float] = UNSET,
    price_2regup_from: Union[Unset, float] = UNSET,
    price_2regup_to: Union[Unset, float] = UNSET,
    price_2regdn_from: Union[Unset, float] = UNSET,
    price_2regdn_to: Union[Unset, float] = UNSET,
    price_2_offline_nspin_from: Union[Unset, float] = UNSET,
    price_2_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw2_from: Union[Unset, float] = UNSET,
    quantity_mw2_to: Union[Unset, float] = UNSET,
    block_indicator_3: Union[Unset, bool] = UNSET,
    price_3rrspfr_from: Union[Unset, float] = UNSET,
    price_3rrspfr_to: Union[Unset, float] = UNSET,
    price_3rrsffr_from: Union[Unset, float] = UNSET,
    price_3rrsffr_to: Union[Unset, float] = UNSET,
    price_3rrsufr_from: Union[Unset, float] = UNSET,
    price_3rrsufr_to: Union[Unset, float] = UNSET,
    price_3_online_nspin_from: Union[Unset, float] = UNSET,
    price_3_online_nspin_to: Union[Unset, float] = UNSET,
    price_3regup_from: Union[Unset, float] = UNSET,
    price_3regup_to: Union[Unset, float] = UNSET,
    price_3regdn_from: Union[Unset, float] = UNSET,
    price_3regdn_to: Union[Unset, float] = UNSET,
    price_3_offline_nspin_from: Union[Unset, float] = UNSET,
    price_3_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw3_from: Union[Unset, float] = UNSET,
    quantity_mw3_to: Union[Unset, float] = UNSET,
    block_indicator_4: Union[Unset, bool] = UNSET,
    price_4rrspfr_from: Union[Unset, float] = UNSET,
    price_4rrspfr_to: Union[Unset, float] = UNSET,
    price_4rrsffr_from: Union[Unset, float] = UNSET,
    price_4rrsffr_to: Union[Unset, float] = UNSET,
    price_4rrsufr_from: Union[Unset, float] = UNSET,
    price_4rrsufr_to: Union[Unset, float] = UNSET,
    price_4_online_nspin_from: Union[Unset, float] = UNSET,
    price_4_online_nspin_to: Union[Unset, float] = UNSET,
    price_4regup_from: Union[Unset, float] = UNSET,
    price_4regup_to: Union[Unset, float] = UNSET,
    price_4regdn_from: Union[Unset, float] = UNSET,
    price_4regdn_to: Union[Unset, float] = UNSET,
    price_4_offline_nspin_from: Union[Unset, float] = UNSET,
    price_4_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw4_from: Union[Unset, float] = UNSET,
    quantity_mw4_to: Union[Unset, float] = UNSET,
    block_indicator_5: Union[Unset, bool] = UNSET,
    price_5rrspfr_from: Union[Unset, float] = UNSET,
    price_5rrspfr_to: Union[Unset, float] = UNSET,
    price_5rrsffr_from: Union[Unset, float] = UNSET,
    price_5rrsffr_to: Union[Unset, float] = UNSET,
    price_5rrsufr_from: Union[Unset, float] = UNSET,
    price_5rrsufr_to: Union[Unset, float] = UNSET,
    price_5_online_nspin_from: Union[Unset, float] = UNSET,
    price_5_online_nspin_to: Union[Unset, float] = UNSET,
    price_5regup_from: Union[Unset, float] = UNSET,
    price_5regup_to: Union[Unset, float] = UNSET,
    price_5regdn_from: Union[Unset, float] = UNSET,
    price_5regdn_to: Union[Unset, float] = UNSET,
    price_5_offline_nspin_from: Union[Unset, float] = UNSET,
    price_5_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw5_from: Union[Unset, float] = UNSET,
    quantity_mw5_to: Union[Unset, float] = UNSET,
    price_1ecrs_from: Union[Unset, float] = UNSET,
    price_1ecrs_to: Union[Unset, float] = UNSET,
    price_1offec_from: Union[Unset, float] = UNSET,
    price_1offec_to: Union[Unset, float] = UNSET,
    price_2ecrs_from: Union[Unset, float] = UNSET,
    price_2ecrs_to: Union[Unset, float] = UNSET,
    price_2offec_from: Union[Unset, float] = UNSET,
    price_2offec_to: Union[Unset, float] = UNSET,
    price_3ecrs_from: Union[Unset, float] = UNSET,
    price_3ecrs_to: Union[Unset, float] = UNSET,
    price_3offec_from: Union[Unset, float] = UNSET,
    price_3offec_to: Union[Unset, float] = UNSET,
    price_4ecrs_from: Union[Unset, float] = UNSET,
    price_4ecrs_to: Union[Unset, float] = UNSET,
    price_4offec_from: Union[Unset, float] = UNSET,
    price_4offec_to: Union[Unset, float] = UNSET,
    price_5ecrs_from: Union[Unset, float] = UNSET,
    price_5ecrs_to: Union[Unset, float] = UNSET,
    price_5offec_from: Union[Unset, float] = UNSET,
    price_5offec_to: Union[Unset, float] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    dme_name: Union[Unset, str] = UNSET,
    resource_name: Union[Unset, str] = UNSET,
    multi_hour_block: Union[Unset, bool] = UNSET,
    block_indicator_1: Union[Unset, bool] = UNSET,
    price_1rrspfr_from: Union[Unset, float] = UNSET,
    price_1rrspfr_to: Union[Unset, float] = UNSET,
    price_1rrsffr_from: Union[Unset, float] = UNSET,
    price_1rrsffr_to: Union[Unset, float] = UNSET,
    price_1rrsufr_from: Union[Unset, float] = UNSET,
    price_1rrsufr_to: Union[Unset, float] = UNSET,
    price_1_online_nspin_from: Union[Unset, float] = UNSET,
    price_1_online_nspin_to: Union[Unset, float] = UNSET,
    price_1regup_from: Union[Unset, float] = UNSET,
    price_1regup_to: Union[Unset, float] = UNSET,
    price_1regdn_from: Union[Unset, float] = UNSET,
    price_1regdn_to: Union[Unset, float] = UNSET,
    price_1_offline_nspin_from: Union[Unset, float] = UNSET,
    price_1_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw1_from: Union[Unset, float] = UNSET,
    quantity_mw1_to: Union[Unset, float] = UNSET,
    block_indicator_2: Union[Unset, bool] = UNSET,
    price_2rrspfr_from: Union[Unset, float] = UNSET,
    price_2rrspfr_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """60-Day DAM Generation Resources AS Offers

     60-Day DAM Generation Resources AS Offers

    Args:
        price_2rrsffr_from (Union[Unset, float]):
        price_2rrsffr_to (Union[Unset, float]):
        price_2rrsufr_from (Union[Unset, float]):
        price_2rrsufr_to (Union[Unset, float]):
        price_2_online_nspin_from (Union[Unset, float]):
        price_2_online_nspin_to (Union[Unset, float]):
        price_2regup_from (Union[Unset, float]):
        price_2regup_to (Union[Unset, float]):
        price_2regdn_from (Union[Unset, float]):
        price_2regdn_to (Union[Unset, float]):
        price_2_offline_nspin_from (Union[Unset, float]):
        price_2_offline_nspin_to (Union[Unset, float]):
        quantity_mw2_from (Union[Unset, float]):
        quantity_mw2_to (Union[Unset, float]):
        block_indicator_3 (Union[Unset, bool]):
        price_3rrspfr_from (Union[Unset, float]):
        price_3rrspfr_to (Union[Unset, float]):
        price_3rrsffr_from (Union[Unset, float]):
        price_3rrsffr_to (Union[Unset, float]):
        price_3rrsufr_from (Union[Unset, float]):
        price_3rrsufr_to (Union[Unset, float]):
        price_3_online_nspin_from (Union[Unset, float]):
        price_3_online_nspin_to (Union[Unset, float]):
        price_3regup_from (Union[Unset, float]):
        price_3regup_to (Union[Unset, float]):
        price_3regdn_from (Union[Unset, float]):
        price_3regdn_to (Union[Unset, float]):
        price_3_offline_nspin_from (Union[Unset, float]):
        price_3_offline_nspin_to (Union[Unset, float]):
        quantity_mw3_from (Union[Unset, float]):
        quantity_mw3_to (Union[Unset, float]):
        block_indicator_4 (Union[Unset, bool]):
        price_4rrspfr_from (Union[Unset, float]):
        price_4rrspfr_to (Union[Unset, float]):
        price_4rrsffr_from (Union[Unset, float]):
        price_4rrsffr_to (Union[Unset, float]):
        price_4rrsufr_from (Union[Unset, float]):
        price_4rrsufr_to (Union[Unset, float]):
        price_4_online_nspin_from (Union[Unset, float]):
        price_4_online_nspin_to (Union[Unset, float]):
        price_4regup_from (Union[Unset, float]):
        price_4regup_to (Union[Unset, float]):
        price_4regdn_from (Union[Unset, float]):
        price_4regdn_to (Union[Unset, float]):
        price_4_offline_nspin_from (Union[Unset, float]):
        price_4_offline_nspin_to (Union[Unset, float]):
        quantity_mw4_from (Union[Unset, float]):
        quantity_mw4_to (Union[Unset, float]):
        block_indicator_5 (Union[Unset, bool]):
        price_5rrspfr_from (Union[Unset, float]):
        price_5rrspfr_to (Union[Unset, float]):
        price_5rrsffr_from (Union[Unset, float]):
        price_5rrsffr_to (Union[Unset, float]):
        price_5rrsufr_from (Union[Unset, float]):
        price_5rrsufr_to (Union[Unset, float]):
        price_5_online_nspin_from (Union[Unset, float]):
        price_5_online_nspin_to (Union[Unset, float]):
        price_5regup_from (Union[Unset, float]):
        price_5regup_to (Union[Unset, float]):
        price_5regdn_from (Union[Unset, float]):
        price_5regdn_to (Union[Unset, float]):
        price_5_offline_nspin_from (Union[Unset, float]):
        price_5_offline_nspin_to (Union[Unset, float]):
        quantity_mw5_from (Union[Unset, float]):
        quantity_mw5_to (Union[Unset, float]):
        price_1ecrs_from (Union[Unset, float]):
        price_1ecrs_to (Union[Unset, float]):
        price_1offec_from (Union[Unset, float]):
        price_1offec_to (Union[Unset, float]):
        price_2ecrs_from (Union[Unset, float]):
        price_2ecrs_to (Union[Unset, float]):
        price_2offec_from (Union[Unset, float]):
        price_2offec_to (Union[Unset, float]):
        price_3ecrs_from (Union[Unset, float]):
        price_3ecrs_to (Union[Unset, float]):
        price_3offec_from (Union[Unset, float]):
        price_3offec_to (Union[Unset, float]):
        price_4ecrs_from (Union[Unset, float]):
        price_4ecrs_to (Union[Unset, float]):
        price_4offec_from (Union[Unset, float]):
        price_4offec_to (Union[Unset, float]):
        price_5ecrs_from (Union[Unset, float]):
        price_5ecrs_to (Union[Unset, float]):
        price_5offec_from (Union[Unset, float]):
        price_5offec_to (Union[Unset, float]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        qse_name (Union[Unset, str]):
        dme_name (Union[Unset, str]):
        resource_name (Union[Unset, str]):
        multi_hour_block (Union[Unset, bool]):
        block_indicator_1 (Union[Unset, bool]):
        price_1rrspfr_from (Union[Unset, float]):
        price_1rrspfr_to (Union[Unset, float]):
        price_1rrsffr_from (Union[Unset, float]):
        price_1rrsffr_to (Union[Unset, float]):
        price_1rrsufr_from (Union[Unset, float]):
        price_1rrsufr_to (Union[Unset, float]):
        price_1_online_nspin_from (Union[Unset, float]):
        price_1_online_nspin_to (Union[Unset, float]):
        price_1regup_from (Union[Unset, float]):
        price_1regup_to (Union[Unset, float]):
        price_1regdn_from (Union[Unset, float]):
        price_1regdn_to (Union[Unset, float]):
        price_1_offline_nspin_from (Union[Unset, float]):
        price_1_offline_nspin_to (Union[Unset, float]):
        quantity_mw1_from (Union[Unset, float]):
        quantity_mw1_to (Union[Unset, float]):
        block_indicator_2 (Union[Unset, bool]):
        price_2rrspfr_from (Union[Unset, float]):
        price_2rrspfr_to (Union[Unset, float]):
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
        price_2rrsffr_from=price_2rrsffr_from,
        price_2rrsffr_to=price_2rrsffr_to,
        price_2rrsufr_from=price_2rrsufr_from,
        price_2rrsufr_to=price_2rrsufr_to,
        price_2_online_nspin_from=price_2_online_nspin_from,
        price_2_online_nspin_to=price_2_online_nspin_to,
        price_2regup_from=price_2regup_from,
        price_2regup_to=price_2regup_to,
        price_2regdn_from=price_2regdn_from,
        price_2regdn_to=price_2regdn_to,
        price_2_offline_nspin_from=price_2_offline_nspin_from,
        price_2_offline_nspin_to=price_2_offline_nspin_to,
        quantity_mw2_from=quantity_mw2_from,
        quantity_mw2_to=quantity_mw2_to,
        block_indicator_3=block_indicator_3,
        price_3rrspfr_from=price_3rrspfr_from,
        price_3rrspfr_to=price_3rrspfr_to,
        price_3rrsffr_from=price_3rrsffr_from,
        price_3rrsffr_to=price_3rrsffr_to,
        price_3rrsufr_from=price_3rrsufr_from,
        price_3rrsufr_to=price_3rrsufr_to,
        price_3_online_nspin_from=price_3_online_nspin_from,
        price_3_online_nspin_to=price_3_online_nspin_to,
        price_3regup_from=price_3regup_from,
        price_3regup_to=price_3regup_to,
        price_3regdn_from=price_3regdn_from,
        price_3regdn_to=price_3regdn_to,
        price_3_offline_nspin_from=price_3_offline_nspin_from,
        price_3_offline_nspin_to=price_3_offline_nspin_to,
        quantity_mw3_from=quantity_mw3_from,
        quantity_mw3_to=quantity_mw3_to,
        block_indicator_4=block_indicator_4,
        price_4rrspfr_from=price_4rrspfr_from,
        price_4rrspfr_to=price_4rrspfr_to,
        price_4rrsffr_from=price_4rrsffr_from,
        price_4rrsffr_to=price_4rrsffr_to,
        price_4rrsufr_from=price_4rrsufr_from,
        price_4rrsufr_to=price_4rrsufr_to,
        price_4_online_nspin_from=price_4_online_nspin_from,
        price_4_online_nspin_to=price_4_online_nspin_to,
        price_4regup_from=price_4regup_from,
        price_4regup_to=price_4regup_to,
        price_4regdn_from=price_4regdn_from,
        price_4regdn_to=price_4regdn_to,
        price_4_offline_nspin_from=price_4_offline_nspin_from,
        price_4_offline_nspin_to=price_4_offline_nspin_to,
        quantity_mw4_from=quantity_mw4_from,
        quantity_mw4_to=quantity_mw4_to,
        block_indicator_5=block_indicator_5,
        price_5rrspfr_from=price_5rrspfr_from,
        price_5rrspfr_to=price_5rrspfr_to,
        price_5rrsffr_from=price_5rrsffr_from,
        price_5rrsffr_to=price_5rrsffr_to,
        price_5rrsufr_from=price_5rrsufr_from,
        price_5rrsufr_to=price_5rrsufr_to,
        price_5_online_nspin_from=price_5_online_nspin_from,
        price_5_online_nspin_to=price_5_online_nspin_to,
        price_5regup_from=price_5regup_from,
        price_5regup_to=price_5regup_to,
        price_5regdn_from=price_5regdn_from,
        price_5regdn_to=price_5regdn_to,
        price_5_offline_nspin_from=price_5_offline_nspin_from,
        price_5_offline_nspin_to=price_5_offline_nspin_to,
        quantity_mw5_from=quantity_mw5_from,
        quantity_mw5_to=quantity_mw5_to,
        price_1ecrs_from=price_1ecrs_from,
        price_1ecrs_to=price_1ecrs_to,
        price_1offec_from=price_1offec_from,
        price_1offec_to=price_1offec_to,
        price_2ecrs_from=price_2ecrs_from,
        price_2ecrs_to=price_2ecrs_to,
        price_2offec_from=price_2offec_from,
        price_2offec_to=price_2offec_to,
        price_3ecrs_from=price_3ecrs_from,
        price_3ecrs_to=price_3ecrs_to,
        price_3offec_from=price_3offec_from,
        price_3offec_to=price_3offec_to,
        price_4ecrs_from=price_4ecrs_from,
        price_4ecrs_to=price_4ecrs_to,
        price_4offec_from=price_4offec_from,
        price_4offec_to=price_4offec_to,
        price_5ecrs_from=price_5ecrs_from,
        price_5ecrs_to=price_5ecrs_to,
        price_5offec_from=price_5offec_from,
        price_5offec_to=price_5offec_to,
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        qse_name=qse_name,
        dme_name=dme_name,
        resource_name=resource_name,
        multi_hour_block=multi_hour_block,
        block_indicator_1=block_indicator_1,
        price_1rrspfr_from=price_1rrspfr_from,
        price_1rrspfr_to=price_1rrspfr_to,
        price_1rrsffr_from=price_1rrsffr_from,
        price_1rrsffr_to=price_1rrsffr_to,
        price_1rrsufr_from=price_1rrsufr_from,
        price_1rrsufr_to=price_1rrsufr_to,
        price_1_online_nspin_from=price_1_online_nspin_from,
        price_1_online_nspin_to=price_1_online_nspin_to,
        price_1regup_from=price_1regup_from,
        price_1regup_to=price_1regup_to,
        price_1regdn_from=price_1regdn_from,
        price_1regdn_to=price_1regdn_to,
        price_1_offline_nspin_from=price_1_offline_nspin_from,
        price_1_offline_nspin_to=price_1_offline_nspin_to,
        quantity_mw1_from=quantity_mw1_from,
        quantity_mw1_to=quantity_mw1_to,
        block_indicator_2=block_indicator_2,
        price_2rrspfr_from=price_2rrspfr_from,
        price_2rrspfr_to=price_2rrspfr_to,
        page=page,
        size=size,
        sort=sort,
        dir_=dir_,
        ocp_apim_subscription_key=ocp_apim_subscription_key,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    price_2rrsffr_from: Union[Unset, float] = UNSET,
    price_2rrsffr_to: Union[Unset, float] = UNSET,
    price_2rrsufr_from: Union[Unset, float] = UNSET,
    price_2rrsufr_to: Union[Unset, float] = UNSET,
    price_2_online_nspin_from: Union[Unset, float] = UNSET,
    price_2_online_nspin_to: Union[Unset, float] = UNSET,
    price_2regup_from: Union[Unset, float] = UNSET,
    price_2regup_to: Union[Unset, float] = UNSET,
    price_2regdn_from: Union[Unset, float] = UNSET,
    price_2regdn_to: Union[Unset, float] = UNSET,
    price_2_offline_nspin_from: Union[Unset, float] = UNSET,
    price_2_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw2_from: Union[Unset, float] = UNSET,
    quantity_mw2_to: Union[Unset, float] = UNSET,
    block_indicator_3: Union[Unset, bool] = UNSET,
    price_3rrspfr_from: Union[Unset, float] = UNSET,
    price_3rrspfr_to: Union[Unset, float] = UNSET,
    price_3rrsffr_from: Union[Unset, float] = UNSET,
    price_3rrsffr_to: Union[Unset, float] = UNSET,
    price_3rrsufr_from: Union[Unset, float] = UNSET,
    price_3rrsufr_to: Union[Unset, float] = UNSET,
    price_3_online_nspin_from: Union[Unset, float] = UNSET,
    price_3_online_nspin_to: Union[Unset, float] = UNSET,
    price_3regup_from: Union[Unset, float] = UNSET,
    price_3regup_to: Union[Unset, float] = UNSET,
    price_3regdn_from: Union[Unset, float] = UNSET,
    price_3regdn_to: Union[Unset, float] = UNSET,
    price_3_offline_nspin_from: Union[Unset, float] = UNSET,
    price_3_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw3_from: Union[Unset, float] = UNSET,
    quantity_mw3_to: Union[Unset, float] = UNSET,
    block_indicator_4: Union[Unset, bool] = UNSET,
    price_4rrspfr_from: Union[Unset, float] = UNSET,
    price_4rrspfr_to: Union[Unset, float] = UNSET,
    price_4rrsffr_from: Union[Unset, float] = UNSET,
    price_4rrsffr_to: Union[Unset, float] = UNSET,
    price_4rrsufr_from: Union[Unset, float] = UNSET,
    price_4rrsufr_to: Union[Unset, float] = UNSET,
    price_4_online_nspin_from: Union[Unset, float] = UNSET,
    price_4_online_nspin_to: Union[Unset, float] = UNSET,
    price_4regup_from: Union[Unset, float] = UNSET,
    price_4regup_to: Union[Unset, float] = UNSET,
    price_4regdn_from: Union[Unset, float] = UNSET,
    price_4regdn_to: Union[Unset, float] = UNSET,
    price_4_offline_nspin_from: Union[Unset, float] = UNSET,
    price_4_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw4_from: Union[Unset, float] = UNSET,
    quantity_mw4_to: Union[Unset, float] = UNSET,
    block_indicator_5: Union[Unset, bool] = UNSET,
    price_5rrspfr_from: Union[Unset, float] = UNSET,
    price_5rrspfr_to: Union[Unset, float] = UNSET,
    price_5rrsffr_from: Union[Unset, float] = UNSET,
    price_5rrsffr_to: Union[Unset, float] = UNSET,
    price_5rrsufr_from: Union[Unset, float] = UNSET,
    price_5rrsufr_to: Union[Unset, float] = UNSET,
    price_5_online_nspin_from: Union[Unset, float] = UNSET,
    price_5_online_nspin_to: Union[Unset, float] = UNSET,
    price_5regup_from: Union[Unset, float] = UNSET,
    price_5regup_to: Union[Unset, float] = UNSET,
    price_5regdn_from: Union[Unset, float] = UNSET,
    price_5regdn_to: Union[Unset, float] = UNSET,
    price_5_offline_nspin_from: Union[Unset, float] = UNSET,
    price_5_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw5_from: Union[Unset, float] = UNSET,
    quantity_mw5_to: Union[Unset, float] = UNSET,
    price_1ecrs_from: Union[Unset, float] = UNSET,
    price_1ecrs_to: Union[Unset, float] = UNSET,
    price_1offec_from: Union[Unset, float] = UNSET,
    price_1offec_to: Union[Unset, float] = UNSET,
    price_2ecrs_from: Union[Unset, float] = UNSET,
    price_2ecrs_to: Union[Unset, float] = UNSET,
    price_2offec_from: Union[Unset, float] = UNSET,
    price_2offec_to: Union[Unset, float] = UNSET,
    price_3ecrs_from: Union[Unset, float] = UNSET,
    price_3ecrs_to: Union[Unset, float] = UNSET,
    price_3offec_from: Union[Unset, float] = UNSET,
    price_3offec_to: Union[Unset, float] = UNSET,
    price_4ecrs_from: Union[Unset, float] = UNSET,
    price_4ecrs_to: Union[Unset, float] = UNSET,
    price_4offec_from: Union[Unset, float] = UNSET,
    price_4offec_to: Union[Unset, float] = UNSET,
    price_5ecrs_from: Union[Unset, float] = UNSET,
    price_5ecrs_to: Union[Unset, float] = UNSET,
    price_5offec_from: Union[Unset, float] = UNSET,
    price_5offec_to: Union[Unset, float] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    dme_name: Union[Unset, str] = UNSET,
    resource_name: Union[Unset, str] = UNSET,
    multi_hour_block: Union[Unset, bool] = UNSET,
    block_indicator_1: Union[Unset, bool] = UNSET,
    price_1rrspfr_from: Union[Unset, float] = UNSET,
    price_1rrspfr_to: Union[Unset, float] = UNSET,
    price_1rrsffr_from: Union[Unset, float] = UNSET,
    price_1rrsffr_to: Union[Unset, float] = UNSET,
    price_1rrsufr_from: Union[Unset, float] = UNSET,
    price_1rrsufr_to: Union[Unset, float] = UNSET,
    price_1_online_nspin_from: Union[Unset, float] = UNSET,
    price_1_online_nspin_to: Union[Unset, float] = UNSET,
    price_1regup_from: Union[Unset, float] = UNSET,
    price_1regup_to: Union[Unset, float] = UNSET,
    price_1regdn_from: Union[Unset, float] = UNSET,
    price_1regdn_to: Union[Unset, float] = UNSET,
    price_1_offline_nspin_from: Union[Unset, float] = UNSET,
    price_1_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw1_from: Union[Unset, float] = UNSET,
    quantity_mw1_to: Union[Unset, float] = UNSET,
    block_indicator_2: Union[Unset, bool] = UNSET,
    price_2rrspfr_from: Union[Unset, float] = UNSET,
    price_2rrspfr_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """60-Day DAM Generation Resources AS Offers

     60-Day DAM Generation Resources AS Offers

    Args:
        price_2rrsffr_from (Union[Unset, float]):
        price_2rrsffr_to (Union[Unset, float]):
        price_2rrsufr_from (Union[Unset, float]):
        price_2rrsufr_to (Union[Unset, float]):
        price_2_online_nspin_from (Union[Unset, float]):
        price_2_online_nspin_to (Union[Unset, float]):
        price_2regup_from (Union[Unset, float]):
        price_2regup_to (Union[Unset, float]):
        price_2regdn_from (Union[Unset, float]):
        price_2regdn_to (Union[Unset, float]):
        price_2_offline_nspin_from (Union[Unset, float]):
        price_2_offline_nspin_to (Union[Unset, float]):
        quantity_mw2_from (Union[Unset, float]):
        quantity_mw2_to (Union[Unset, float]):
        block_indicator_3 (Union[Unset, bool]):
        price_3rrspfr_from (Union[Unset, float]):
        price_3rrspfr_to (Union[Unset, float]):
        price_3rrsffr_from (Union[Unset, float]):
        price_3rrsffr_to (Union[Unset, float]):
        price_3rrsufr_from (Union[Unset, float]):
        price_3rrsufr_to (Union[Unset, float]):
        price_3_online_nspin_from (Union[Unset, float]):
        price_3_online_nspin_to (Union[Unset, float]):
        price_3regup_from (Union[Unset, float]):
        price_3regup_to (Union[Unset, float]):
        price_3regdn_from (Union[Unset, float]):
        price_3regdn_to (Union[Unset, float]):
        price_3_offline_nspin_from (Union[Unset, float]):
        price_3_offline_nspin_to (Union[Unset, float]):
        quantity_mw3_from (Union[Unset, float]):
        quantity_mw3_to (Union[Unset, float]):
        block_indicator_4 (Union[Unset, bool]):
        price_4rrspfr_from (Union[Unset, float]):
        price_4rrspfr_to (Union[Unset, float]):
        price_4rrsffr_from (Union[Unset, float]):
        price_4rrsffr_to (Union[Unset, float]):
        price_4rrsufr_from (Union[Unset, float]):
        price_4rrsufr_to (Union[Unset, float]):
        price_4_online_nspin_from (Union[Unset, float]):
        price_4_online_nspin_to (Union[Unset, float]):
        price_4regup_from (Union[Unset, float]):
        price_4regup_to (Union[Unset, float]):
        price_4regdn_from (Union[Unset, float]):
        price_4regdn_to (Union[Unset, float]):
        price_4_offline_nspin_from (Union[Unset, float]):
        price_4_offline_nspin_to (Union[Unset, float]):
        quantity_mw4_from (Union[Unset, float]):
        quantity_mw4_to (Union[Unset, float]):
        block_indicator_5 (Union[Unset, bool]):
        price_5rrspfr_from (Union[Unset, float]):
        price_5rrspfr_to (Union[Unset, float]):
        price_5rrsffr_from (Union[Unset, float]):
        price_5rrsffr_to (Union[Unset, float]):
        price_5rrsufr_from (Union[Unset, float]):
        price_5rrsufr_to (Union[Unset, float]):
        price_5_online_nspin_from (Union[Unset, float]):
        price_5_online_nspin_to (Union[Unset, float]):
        price_5regup_from (Union[Unset, float]):
        price_5regup_to (Union[Unset, float]):
        price_5regdn_from (Union[Unset, float]):
        price_5regdn_to (Union[Unset, float]):
        price_5_offline_nspin_from (Union[Unset, float]):
        price_5_offline_nspin_to (Union[Unset, float]):
        quantity_mw5_from (Union[Unset, float]):
        quantity_mw5_to (Union[Unset, float]):
        price_1ecrs_from (Union[Unset, float]):
        price_1ecrs_to (Union[Unset, float]):
        price_1offec_from (Union[Unset, float]):
        price_1offec_to (Union[Unset, float]):
        price_2ecrs_from (Union[Unset, float]):
        price_2ecrs_to (Union[Unset, float]):
        price_2offec_from (Union[Unset, float]):
        price_2offec_to (Union[Unset, float]):
        price_3ecrs_from (Union[Unset, float]):
        price_3ecrs_to (Union[Unset, float]):
        price_3offec_from (Union[Unset, float]):
        price_3offec_to (Union[Unset, float]):
        price_4ecrs_from (Union[Unset, float]):
        price_4ecrs_to (Union[Unset, float]):
        price_4offec_from (Union[Unset, float]):
        price_4offec_to (Union[Unset, float]):
        price_5ecrs_from (Union[Unset, float]):
        price_5ecrs_to (Union[Unset, float]):
        price_5offec_from (Union[Unset, float]):
        price_5offec_to (Union[Unset, float]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        qse_name (Union[Unset, str]):
        dme_name (Union[Unset, str]):
        resource_name (Union[Unset, str]):
        multi_hour_block (Union[Unset, bool]):
        block_indicator_1 (Union[Unset, bool]):
        price_1rrspfr_from (Union[Unset, float]):
        price_1rrspfr_to (Union[Unset, float]):
        price_1rrsffr_from (Union[Unset, float]):
        price_1rrsffr_to (Union[Unset, float]):
        price_1rrsufr_from (Union[Unset, float]):
        price_1rrsufr_to (Union[Unset, float]):
        price_1_online_nspin_from (Union[Unset, float]):
        price_1_online_nspin_to (Union[Unset, float]):
        price_1regup_from (Union[Unset, float]):
        price_1regup_to (Union[Unset, float]):
        price_1regdn_from (Union[Unset, float]):
        price_1regdn_to (Union[Unset, float]):
        price_1_offline_nspin_from (Union[Unset, float]):
        price_1_offline_nspin_to (Union[Unset, float]):
        quantity_mw1_from (Union[Unset, float]):
        quantity_mw1_to (Union[Unset, float]):
        block_indicator_2 (Union[Unset, bool]):
        price_2rrspfr_from (Union[Unset, float]):
        price_2rrspfr_to (Union[Unset, float]):
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
        price_2rrsffr_from=price_2rrsffr_from,
        price_2rrsffr_to=price_2rrsffr_to,
        price_2rrsufr_from=price_2rrsufr_from,
        price_2rrsufr_to=price_2rrsufr_to,
        price_2_online_nspin_from=price_2_online_nspin_from,
        price_2_online_nspin_to=price_2_online_nspin_to,
        price_2regup_from=price_2regup_from,
        price_2regup_to=price_2regup_to,
        price_2regdn_from=price_2regdn_from,
        price_2regdn_to=price_2regdn_to,
        price_2_offline_nspin_from=price_2_offline_nspin_from,
        price_2_offline_nspin_to=price_2_offline_nspin_to,
        quantity_mw2_from=quantity_mw2_from,
        quantity_mw2_to=quantity_mw2_to,
        block_indicator_3=block_indicator_3,
        price_3rrspfr_from=price_3rrspfr_from,
        price_3rrspfr_to=price_3rrspfr_to,
        price_3rrsffr_from=price_3rrsffr_from,
        price_3rrsffr_to=price_3rrsffr_to,
        price_3rrsufr_from=price_3rrsufr_from,
        price_3rrsufr_to=price_3rrsufr_to,
        price_3_online_nspin_from=price_3_online_nspin_from,
        price_3_online_nspin_to=price_3_online_nspin_to,
        price_3regup_from=price_3regup_from,
        price_3regup_to=price_3regup_to,
        price_3regdn_from=price_3regdn_from,
        price_3regdn_to=price_3regdn_to,
        price_3_offline_nspin_from=price_3_offline_nspin_from,
        price_3_offline_nspin_to=price_3_offline_nspin_to,
        quantity_mw3_from=quantity_mw3_from,
        quantity_mw3_to=quantity_mw3_to,
        block_indicator_4=block_indicator_4,
        price_4rrspfr_from=price_4rrspfr_from,
        price_4rrspfr_to=price_4rrspfr_to,
        price_4rrsffr_from=price_4rrsffr_from,
        price_4rrsffr_to=price_4rrsffr_to,
        price_4rrsufr_from=price_4rrsufr_from,
        price_4rrsufr_to=price_4rrsufr_to,
        price_4_online_nspin_from=price_4_online_nspin_from,
        price_4_online_nspin_to=price_4_online_nspin_to,
        price_4regup_from=price_4regup_from,
        price_4regup_to=price_4regup_to,
        price_4regdn_from=price_4regdn_from,
        price_4regdn_to=price_4regdn_to,
        price_4_offline_nspin_from=price_4_offline_nspin_from,
        price_4_offline_nspin_to=price_4_offline_nspin_to,
        quantity_mw4_from=quantity_mw4_from,
        quantity_mw4_to=quantity_mw4_to,
        block_indicator_5=block_indicator_5,
        price_5rrspfr_from=price_5rrspfr_from,
        price_5rrspfr_to=price_5rrspfr_to,
        price_5rrsffr_from=price_5rrsffr_from,
        price_5rrsffr_to=price_5rrsffr_to,
        price_5rrsufr_from=price_5rrsufr_from,
        price_5rrsufr_to=price_5rrsufr_to,
        price_5_online_nspin_from=price_5_online_nspin_from,
        price_5_online_nspin_to=price_5_online_nspin_to,
        price_5regup_from=price_5regup_from,
        price_5regup_to=price_5regup_to,
        price_5regdn_from=price_5regdn_from,
        price_5regdn_to=price_5regdn_to,
        price_5_offline_nspin_from=price_5_offline_nspin_from,
        price_5_offline_nspin_to=price_5_offline_nspin_to,
        quantity_mw5_from=quantity_mw5_from,
        quantity_mw5_to=quantity_mw5_to,
        price_1ecrs_from=price_1ecrs_from,
        price_1ecrs_to=price_1ecrs_to,
        price_1offec_from=price_1offec_from,
        price_1offec_to=price_1offec_to,
        price_2ecrs_from=price_2ecrs_from,
        price_2ecrs_to=price_2ecrs_to,
        price_2offec_from=price_2offec_from,
        price_2offec_to=price_2offec_to,
        price_3ecrs_from=price_3ecrs_from,
        price_3ecrs_to=price_3ecrs_to,
        price_3offec_from=price_3offec_from,
        price_3offec_to=price_3offec_to,
        price_4ecrs_from=price_4ecrs_from,
        price_4ecrs_to=price_4ecrs_to,
        price_4offec_from=price_4offec_from,
        price_4offec_to=price_4offec_to,
        price_5ecrs_from=price_5ecrs_from,
        price_5ecrs_to=price_5ecrs_to,
        price_5offec_from=price_5offec_from,
        price_5offec_to=price_5offec_to,
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        qse_name=qse_name,
        dme_name=dme_name,
        resource_name=resource_name,
        multi_hour_block=multi_hour_block,
        block_indicator_1=block_indicator_1,
        price_1rrspfr_from=price_1rrspfr_from,
        price_1rrspfr_to=price_1rrspfr_to,
        price_1rrsffr_from=price_1rrsffr_from,
        price_1rrsffr_to=price_1rrsffr_to,
        price_1rrsufr_from=price_1rrsufr_from,
        price_1rrsufr_to=price_1rrsufr_to,
        price_1_online_nspin_from=price_1_online_nspin_from,
        price_1_online_nspin_to=price_1_online_nspin_to,
        price_1regup_from=price_1regup_from,
        price_1regup_to=price_1regup_to,
        price_1regdn_from=price_1regdn_from,
        price_1regdn_to=price_1regdn_to,
        price_1_offline_nspin_from=price_1_offline_nspin_from,
        price_1_offline_nspin_to=price_1_offline_nspin_to,
        quantity_mw1_from=quantity_mw1_from,
        quantity_mw1_to=quantity_mw1_to,
        block_indicator_2=block_indicator_2,
        price_2rrspfr_from=price_2rrspfr_from,
        price_2rrspfr_to=price_2rrspfr_to,
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
    price_2rrsffr_from: Union[Unset, float] = UNSET,
    price_2rrsffr_to: Union[Unset, float] = UNSET,
    price_2rrsufr_from: Union[Unset, float] = UNSET,
    price_2rrsufr_to: Union[Unset, float] = UNSET,
    price_2_online_nspin_from: Union[Unset, float] = UNSET,
    price_2_online_nspin_to: Union[Unset, float] = UNSET,
    price_2regup_from: Union[Unset, float] = UNSET,
    price_2regup_to: Union[Unset, float] = UNSET,
    price_2regdn_from: Union[Unset, float] = UNSET,
    price_2regdn_to: Union[Unset, float] = UNSET,
    price_2_offline_nspin_from: Union[Unset, float] = UNSET,
    price_2_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw2_from: Union[Unset, float] = UNSET,
    quantity_mw2_to: Union[Unset, float] = UNSET,
    block_indicator_3: Union[Unset, bool] = UNSET,
    price_3rrspfr_from: Union[Unset, float] = UNSET,
    price_3rrspfr_to: Union[Unset, float] = UNSET,
    price_3rrsffr_from: Union[Unset, float] = UNSET,
    price_3rrsffr_to: Union[Unset, float] = UNSET,
    price_3rrsufr_from: Union[Unset, float] = UNSET,
    price_3rrsufr_to: Union[Unset, float] = UNSET,
    price_3_online_nspin_from: Union[Unset, float] = UNSET,
    price_3_online_nspin_to: Union[Unset, float] = UNSET,
    price_3regup_from: Union[Unset, float] = UNSET,
    price_3regup_to: Union[Unset, float] = UNSET,
    price_3regdn_from: Union[Unset, float] = UNSET,
    price_3regdn_to: Union[Unset, float] = UNSET,
    price_3_offline_nspin_from: Union[Unset, float] = UNSET,
    price_3_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw3_from: Union[Unset, float] = UNSET,
    quantity_mw3_to: Union[Unset, float] = UNSET,
    block_indicator_4: Union[Unset, bool] = UNSET,
    price_4rrspfr_from: Union[Unset, float] = UNSET,
    price_4rrspfr_to: Union[Unset, float] = UNSET,
    price_4rrsffr_from: Union[Unset, float] = UNSET,
    price_4rrsffr_to: Union[Unset, float] = UNSET,
    price_4rrsufr_from: Union[Unset, float] = UNSET,
    price_4rrsufr_to: Union[Unset, float] = UNSET,
    price_4_online_nspin_from: Union[Unset, float] = UNSET,
    price_4_online_nspin_to: Union[Unset, float] = UNSET,
    price_4regup_from: Union[Unset, float] = UNSET,
    price_4regup_to: Union[Unset, float] = UNSET,
    price_4regdn_from: Union[Unset, float] = UNSET,
    price_4regdn_to: Union[Unset, float] = UNSET,
    price_4_offline_nspin_from: Union[Unset, float] = UNSET,
    price_4_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw4_from: Union[Unset, float] = UNSET,
    quantity_mw4_to: Union[Unset, float] = UNSET,
    block_indicator_5: Union[Unset, bool] = UNSET,
    price_5rrspfr_from: Union[Unset, float] = UNSET,
    price_5rrspfr_to: Union[Unset, float] = UNSET,
    price_5rrsffr_from: Union[Unset, float] = UNSET,
    price_5rrsffr_to: Union[Unset, float] = UNSET,
    price_5rrsufr_from: Union[Unset, float] = UNSET,
    price_5rrsufr_to: Union[Unset, float] = UNSET,
    price_5_online_nspin_from: Union[Unset, float] = UNSET,
    price_5_online_nspin_to: Union[Unset, float] = UNSET,
    price_5regup_from: Union[Unset, float] = UNSET,
    price_5regup_to: Union[Unset, float] = UNSET,
    price_5regdn_from: Union[Unset, float] = UNSET,
    price_5regdn_to: Union[Unset, float] = UNSET,
    price_5_offline_nspin_from: Union[Unset, float] = UNSET,
    price_5_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw5_from: Union[Unset, float] = UNSET,
    quantity_mw5_to: Union[Unset, float] = UNSET,
    price_1ecrs_from: Union[Unset, float] = UNSET,
    price_1ecrs_to: Union[Unset, float] = UNSET,
    price_1offec_from: Union[Unset, float] = UNSET,
    price_1offec_to: Union[Unset, float] = UNSET,
    price_2ecrs_from: Union[Unset, float] = UNSET,
    price_2ecrs_to: Union[Unset, float] = UNSET,
    price_2offec_from: Union[Unset, float] = UNSET,
    price_2offec_to: Union[Unset, float] = UNSET,
    price_3ecrs_from: Union[Unset, float] = UNSET,
    price_3ecrs_to: Union[Unset, float] = UNSET,
    price_3offec_from: Union[Unset, float] = UNSET,
    price_3offec_to: Union[Unset, float] = UNSET,
    price_4ecrs_from: Union[Unset, float] = UNSET,
    price_4ecrs_to: Union[Unset, float] = UNSET,
    price_4offec_from: Union[Unset, float] = UNSET,
    price_4offec_to: Union[Unset, float] = UNSET,
    price_5ecrs_from: Union[Unset, float] = UNSET,
    price_5ecrs_to: Union[Unset, float] = UNSET,
    price_5offec_from: Union[Unset, float] = UNSET,
    price_5offec_to: Union[Unset, float] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    dme_name: Union[Unset, str] = UNSET,
    resource_name: Union[Unset, str] = UNSET,
    multi_hour_block: Union[Unset, bool] = UNSET,
    block_indicator_1: Union[Unset, bool] = UNSET,
    price_1rrspfr_from: Union[Unset, float] = UNSET,
    price_1rrspfr_to: Union[Unset, float] = UNSET,
    price_1rrsffr_from: Union[Unset, float] = UNSET,
    price_1rrsffr_to: Union[Unset, float] = UNSET,
    price_1rrsufr_from: Union[Unset, float] = UNSET,
    price_1rrsufr_to: Union[Unset, float] = UNSET,
    price_1_online_nspin_from: Union[Unset, float] = UNSET,
    price_1_online_nspin_to: Union[Unset, float] = UNSET,
    price_1regup_from: Union[Unset, float] = UNSET,
    price_1regup_to: Union[Unset, float] = UNSET,
    price_1regdn_from: Union[Unset, float] = UNSET,
    price_1regdn_to: Union[Unset, float] = UNSET,
    price_1_offline_nspin_from: Union[Unset, float] = UNSET,
    price_1_offline_nspin_to: Union[Unset, float] = UNSET,
    quantity_mw1_from: Union[Unset, float] = UNSET,
    quantity_mw1_to: Union[Unset, float] = UNSET,
    block_indicator_2: Union[Unset, bool] = UNSET,
    price_2rrspfr_from: Union[Unset, float] = UNSET,
    price_2rrspfr_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """60-Day DAM Generation Resources AS Offers

     60-Day DAM Generation Resources AS Offers

    Args:
        price_2rrsffr_from (Union[Unset, float]):
        price_2rrsffr_to (Union[Unset, float]):
        price_2rrsufr_from (Union[Unset, float]):
        price_2rrsufr_to (Union[Unset, float]):
        price_2_online_nspin_from (Union[Unset, float]):
        price_2_online_nspin_to (Union[Unset, float]):
        price_2regup_from (Union[Unset, float]):
        price_2regup_to (Union[Unset, float]):
        price_2regdn_from (Union[Unset, float]):
        price_2regdn_to (Union[Unset, float]):
        price_2_offline_nspin_from (Union[Unset, float]):
        price_2_offline_nspin_to (Union[Unset, float]):
        quantity_mw2_from (Union[Unset, float]):
        quantity_mw2_to (Union[Unset, float]):
        block_indicator_3 (Union[Unset, bool]):
        price_3rrspfr_from (Union[Unset, float]):
        price_3rrspfr_to (Union[Unset, float]):
        price_3rrsffr_from (Union[Unset, float]):
        price_3rrsffr_to (Union[Unset, float]):
        price_3rrsufr_from (Union[Unset, float]):
        price_3rrsufr_to (Union[Unset, float]):
        price_3_online_nspin_from (Union[Unset, float]):
        price_3_online_nspin_to (Union[Unset, float]):
        price_3regup_from (Union[Unset, float]):
        price_3regup_to (Union[Unset, float]):
        price_3regdn_from (Union[Unset, float]):
        price_3regdn_to (Union[Unset, float]):
        price_3_offline_nspin_from (Union[Unset, float]):
        price_3_offline_nspin_to (Union[Unset, float]):
        quantity_mw3_from (Union[Unset, float]):
        quantity_mw3_to (Union[Unset, float]):
        block_indicator_4 (Union[Unset, bool]):
        price_4rrspfr_from (Union[Unset, float]):
        price_4rrspfr_to (Union[Unset, float]):
        price_4rrsffr_from (Union[Unset, float]):
        price_4rrsffr_to (Union[Unset, float]):
        price_4rrsufr_from (Union[Unset, float]):
        price_4rrsufr_to (Union[Unset, float]):
        price_4_online_nspin_from (Union[Unset, float]):
        price_4_online_nspin_to (Union[Unset, float]):
        price_4regup_from (Union[Unset, float]):
        price_4regup_to (Union[Unset, float]):
        price_4regdn_from (Union[Unset, float]):
        price_4regdn_to (Union[Unset, float]):
        price_4_offline_nspin_from (Union[Unset, float]):
        price_4_offline_nspin_to (Union[Unset, float]):
        quantity_mw4_from (Union[Unset, float]):
        quantity_mw4_to (Union[Unset, float]):
        block_indicator_5 (Union[Unset, bool]):
        price_5rrspfr_from (Union[Unset, float]):
        price_5rrspfr_to (Union[Unset, float]):
        price_5rrsffr_from (Union[Unset, float]):
        price_5rrsffr_to (Union[Unset, float]):
        price_5rrsufr_from (Union[Unset, float]):
        price_5rrsufr_to (Union[Unset, float]):
        price_5_online_nspin_from (Union[Unset, float]):
        price_5_online_nspin_to (Union[Unset, float]):
        price_5regup_from (Union[Unset, float]):
        price_5regup_to (Union[Unset, float]):
        price_5regdn_from (Union[Unset, float]):
        price_5regdn_to (Union[Unset, float]):
        price_5_offline_nspin_from (Union[Unset, float]):
        price_5_offline_nspin_to (Union[Unset, float]):
        quantity_mw5_from (Union[Unset, float]):
        quantity_mw5_to (Union[Unset, float]):
        price_1ecrs_from (Union[Unset, float]):
        price_1ecrs_to (Union[Unset, float]):
        price_1offec_from (Union[Unset, float]):
        price_1offec_to (Union[Unset, float]):
        price_2ecrs_from (Union[Unset, float]):
        price_2ecrs_to (Union[Unset, float]):
        price_2offec_from (Union[Unset, float]):
        price_2offec_to (Union[Unset, float]):
        price_3ecrs_from (Union[Unset, float]):
        price_3ecrs_to (Union[Unset, float]):
        price_3offec_from (Union[Unset, float]):
        price_3offec_to (Union[Unset, float]):
        price_4ecrs_from (Union[Unset, float]):
        price_4ecrs_to (Union[Unset, float]):
        price_4offec_from (Union[Unset, float]):
        price_4offec_to (Union[Unset, float]):
        price_5ecrs_from (Union[Unset, float]):
        price_5ecrs_to (Union[Unset, float]):
        price_5offec_from (Union[Unset, float]):
        price_5offec_to (Union[Unset, float]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        qse_name (Union[Unset, str]):
        dme_name (Union[Unset, str]):
        resource_name (Union[Unset, str]):
        multi_hour_block (Union[Unset, bool]):
        block_indicator_1 (Union[Unset, bool]):
        price_1rrspfr_from (Union[Unset, float]):
        price_1rrspfr_to (Union[Unset, float]):
        price_1rrsffr_from (Union[Unset, float]):
        price_1rrsffr_to (Union[Unset, float]):
        price_1rrsufr_from (Union[Unset, float]):
        price_1rrsufr_to (Union[Unset, float]):
        price_1_online_nspin_from (Union[Unset, float]):
        price_1_online_nspin_to (Union[Unset, float]):
        price_1regup_from (Union[Unset, float]):
        price_1regup_to (Union[Unset, float]):
        price_1regdn_from (Union[Unset, float]):
        price_1regdn_to (Union[Unset, float]):
        price_1_offline_nspin_from (Union[Unset, float]):
        price_1_offline_nspin_to (Union[Unset, float]):
        quantity_mw1_from (Union[Unset, float]):
        quantity_mw1_to (Union[Unset, float]):
        block_indicator_2 (Union[Unset, bool]):
        price_2rrspfr_from (Union[Unset, float]):
        price_2rrspfr_to (Union[Unset, float]):
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
            price_2rrsffr_from=price_2rrsffr_from,
            price_2rrsffr_to=price_2rrsffr_to,
            price_2rrsufr_from=price_2rrsufr_from,
            price_2rrsufr_to=price_2rrsufr_to,
            price_2_online_nspin_from=price_2_online_nspin_from,
            price_2_online_nspin_to=price_2_online_nspin_to,
            price_2regup_from=price_2regup_from,
            price_2regup_to=price_2regup_to,
            price_2regdn_from=price_2regdn_from,
            price_2regdn_to=price_2regdn_to,
            price_2_offline_nspin_from=price_2_offline_nspin_from,
            price_2_offline_nspin_to=price_2_offline_nspin_to,
            quantity_mw2_from=quantity_mw2_from,
            quantity_mw2_to=quantity_mw2_to,
            block_indicator_3=block_indicator_3,
            price_3rrspfr_from=price_3rrspfr_from,
            price_3rrspfr_to=price_3rrspfr_to,
            price_3rrsffr_from=price_3rrsffr_from,
            price_3rrsffr_to=price_3rrsffr_to,
            price_3rrsufr_from=price_3rrsufr_from,
            price_3rrsufr_to=price_3rrsufr_to,
            price_3_online_nspin_from=price_3_online_nspin_from,
            price_3_online_nspin_to=price_3_online_nspin_to,
            price_3regup_from=price_3regup_from,
            price_3regup_to=price_3regup_to,
            price_3regdn_from=price_3regdn_from,
            price_3regdn_to=price_3regdn_to,
            price_3_offline_nspin_from=price_3_offline_nspin_from,
            price_3_offline_nspin_to=price_3_offline_nspin_to,
            quantity_mw3_from=quantity_mw3_from,
            quantity_mw3_to=quantity_mw3_to,
            block_indicator_4=block_indicator_4,
            price_4rrspfr_from=price_4rrspfr_from,
            price_4rrspfr_to=price_4rrspfr_to,
            price_4rrsffr_from=price_4rrsffr_from,
            price_4rrsffr_to=price_4rrsffr_to,
            price_4rrsufr_from=price_4rrsufr_from,
            price_4rrsufr_to=price_4rrsufr_to,
            price_4_online_nspin_from=price_4_online_nspin_from,
            price_4_online_nspin_to=price_4_online_nspin_to,
            price_4regup_from=price_4regup_from,
            price_4regup_to=price_4regup_to,
            price_4regdn_from=price_4regdn_from,
            price_4regdn_to=price_4regdn_to,
            price_4_offline_nspin_from=price_4_offline_nspin_from,
            price_4_offline_nspin_to=price_4_offline_nspin_to,
            quantity_mw4_from=quantity_mw4_from,
            quantity_mw4_to=quantity_mw4_to,
            block_indicator_5=block_indicator_5,
            price_5rrspfr_from=price_5rrspfr_from,
            price_5rrspfr_to=price_5rrspfr_to,
            price_5rrsffr_from=price_5rrsffr_from,
            price_5rrsffr_to=price_5rrsffr_to,
            price_5rrsufr_from=price_5rrsufr_from,
            price_5rrsufr_to=price_5rrsufr_to,
            price_5_online_nspin_from=price_5_online_nspin_from,
            price_5_online_nspin_to=price_5_online_nspin_to,
            price_5regup_from=price_5regup_from,
            price_5regup_to=price_5regup_to,
            price_5regdn_from=price_5regdn_from,
            price_5regdn_to=price_5regdn_to,
            price_5_offline_nspin_from=price_5_offline_nspin_from,
            price_5_offline_nspin_to=price_5_offline_nspin_to,
            quantity_mw5_from=quantity_mw5_from,
            quantity_mw5_to=quantity_mw5_to,
            price_1ecrs_from=price_1ecrs_from,
            price_1ecrs_to=price_1ecrs_to,
            price_1offec_from=price_1offec_from,
            price_1offec_to=price_1offec_to,
            price_2ecrs_from=price_2ecrs_from,
            price_2ecrs_to=price_2ecrs_to,
            price_2offec_from=price_2offec_from,
            price_2offec_to=price_2offec_to,
            price_3ecrs_from=price_3ecrs_from,
            price_3ecrs_to=price_3ecrs_to,
            price_3offec_from=price_3offec_from,
            price_3offec_to=price_3offec_to,
            price_4ecrs_from=price_4ecrs_from,
            price_4ecrs_to=price_4ecrs_to,
            price_4offec_from=price_4offec_from,
            price_4offec_to=price_4offec_to,
            price_5ecrs_from=price_5ecrs_from,
            price_5ecrs_to=price_5ecrs_to,
            price_5offec_from=price_5offec_from,
            price_5offec_to=price_5offec_to,
            delivery_date_from=delivery_date_from,
            delivery_date_to=delivery_date_to,
            hour_ending_from=hour_ending_from,
            hour_ending_to=hour_ending_to,
            qse_name=qse_name,
            dme_name=dme_name,
            resource_name=resource_name,
            multi_hour_block=multi_hour_block,
            block_indicator_1=block_indicator_1,
            price_1rrspfr_from=price_1rrspfr_from,
            price_1rrspfr_to=price_1rrspfr_to,
            price_1rrsffr_from=price_1rrsffr_from,
            price_1rrsffr_to=price_1rrsffr_to,
            price_1rrsufr_from=price_1rrsufr_from,
            price_1rrsufr_to=price_1rrsufr_to,
            price_1_online_nspin_from=price_1_online_nspin_from,
            price_1_online_nspin_to=price_1_online_nspin_to,
            price_1regup_from=price_1regup_from,
            price_1regup_to=price_1regup_to,
            price_1regdn_from=price_1regdn_from,
            price_1regdn_to=price_1regdn_to,
            price_1_offline_nspin_from=price_1_offline_nspin_from,
            price_1_offline_nspin_to=price_1_offline_nspin_to,
            quantity_mw1_from=quantity_mw1_from,
            quantity_mw1_to=quantity_mw1_to,
            block_indicator_2=block_indicator_2,
            price_2rrspfr_from=price_2rrspfr_from,
            price_2rrspfr_to=price_2rrspfr_to,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
