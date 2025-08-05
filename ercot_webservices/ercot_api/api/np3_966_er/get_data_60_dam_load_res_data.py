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
    ecrsmcpc_from: Union[Unset, float] = UNSET,
    ecrsmcpc_to: Union[Unset, float] = UNSET,
    regupmcpc_from: Union[Unset, float] = UNSET,
    regupmcpc_to: Union[Unset, float] = UNSET,
    regdn_awarded_from: Union[Unset, float] = UNSET,
    regdn_awarded_to: Union[Unset, float] = UNSET,
    regdnmcpc_from: Union[Unset, float] = UNSET,
    regdnmcpc_to: Union[Unset, float] = UNSET,
    rrspfr_awarded_from: Union[Unset, float] = UNSET,
    rrspfr_awarded_to: Union[Unset, float] = UNSET,
    rrsffr_awarded_from: Union[Unset, float] = UNSET,
    rrsffr_awarded_to: Union[Unset, float] = UNSET,
    rrsufr_awarded_from: Union[Unset, float] = UNSET,
    rrsufr_awarded_to: Union[Unset, float] = UNSET,
    rrsmcpc_from: Union[Unset, float] = UNSET,
    rrsmcpc_to: Union[Unset, float] = UNSET,
    nspin_awarded_from: Union[Unset, float] = UNSET,
    nspin_awarded_to: Union[Unset, float] = UNSET,
    nspinmcpc_from: Union[Unset, float] = UNSET,
    nspinmcpc_to: Union[Unset, float] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    load_resource_name: Union[Unset, str] = UNSET,
    max_power_consumption_from: Union[Unset, float] = UNSET,
    max_power_consumption_to: Union[Unset, float] = UNSET,
    ecrssd_awarded_from: Union[Unset, float] = UNSET,
    ecrssd_awarded_to: Union[Unset, float] = UNSET,
    low_power_consumption_from: Union[Unset, float] = UNSET,
    low_power_consumption_to: Union[Unset, float] = UNSET,
    ecrsmd_awarded_from: Union[Unset, float] = UNSET,
    ecrsmd_awarded_to: Union[Unset, float] = UNSET,
    regup_awarded_from: Union[Unset, float] = UNSET,
    regup_awarded_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["Ocp-Apim-Subscription-Key"] = ocp_apim_subscription_key

    params: dict[str, Any] = {}

    params["ECRSMCPCFrom"] = ecrsmcpc_from

    params["ECRSMCPCTo"] = ecrsmcpc_to

    params["REGUPMCPCFrom"] = regupmcpc_from

    params["REGUPMCPCTo"] = regupmcpc_to

    params["REGDNAwardedFrom"] = regdn_awarded_from

    params["REGDNAwardedTo"] = regdn_awarded_to

    params["REGDNMCPCFrom"] = regdnmcpc_from

    params["REGDNMCPCTo"] = regdnmcpc_to

    params["RRSPFRAwardedFrom"] = rrspfr_awarded_from

    params["RRSPFRAwardedTo"] = rrspfr_awarded_to

    params["RRSFFRAwardedFrom"] = rrsffr_awarded_from

    params["RRSFFRAwardedTo"] = rrsffr_awarded_to

    params["RRSUFRAwardedFrom"] = rrsufr_awarded_from

    params["RRSUFRAwardedTo"] = rrsufr_awarded_to

    params["RRSMCPCFrom"] = rrsmcpc_from

    params["RRSMCPCTo"] = rrsmcpc_to

    params["NSPINAwardedFrom"] = nspin_awarded_from

    params["NSPINAwardedTo"] = nspin_awarded_to

    params["NSPINMCPCFrom"] = nspinmcpc_from

    params["NSPINMCPCTo"] = nspinmcpc_to

    params["deliveryDateFrom"] = delivery_date_from

    params["deliveryDateTo"] = delivery_date_to

    params["hourEndingFrom"] = hour_ending_from

    params["hourEndingTo"] = hour_ending_to

    params["loadResourceName"] = load_resource_name

    params["maxPowerConsumptionFrom"] = max_power_consumption_from

    params["maxPowerConsumptionTo"] = max_power_consumption_to

    params["ECRSSDAwardedFrom"] = ecrssd_awarded_from

    params["ECRSSDAwardedTo"] = ecrssd_awarded_to

    params["lowPowerConsumptionFrom"] = low_power_consumption_from

    params["lowPowerConsumptionTo"] = low_power_consumption_to

    params["ECRSMDAwardedFrom"] = ecrsmd_awarded_from

    params["ECRSMDAwardedTo"] = ecrsmd_awarded_to

    params["REGUPAwardedFrom"] = regup_awarded_from

    params["REGUPAwardedTo"] = regup_awarded_to

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np3-966-er/60_dam_load_res_data",
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
    ecrsmcpc_from: Union[Unset, float] = UNSET,
    ecrsmcpc_to: Union[Unset, float] = UNSET,
    regupmcpc_from: Union[Unset, float] = UNSET,
    regupmcpc_to: Union[Unset, float] = UNSET,
    regdn_awarded_from: Union[Unset, float] = UNSET,
    regdn_awarded_to: Union[Unset, float] = UNSET,
    regdnmcpc_from: Union[Unset, float] = UNSET,
    regdnmcpc_to: Union[Unset, float] = UNSET,
    rrspfr_awarded_from: Union[Unset, float] = UNSET,
    rrspfr_awarded_to: Union[Unset, float] = UNSET,
    rrsffr_awarded_from: Union[Unset, float] = UNSET,
    rrsffr_awarded_to: Union[Unset, float] = UNSET,
    rrsufr_awarded_from: Union[Unset, float] = UNSET,
    rrsufr_awarded_to: Union[Unset, float] = UNSET,
    rrsmcpc_from: Union[Unset, float] = UNSET,
    rrsmcpc_to: Union[Unset, float] = UNSET,
    nspin_awarded_from: Union[Unset, float] = UNSET,
    nspin_awarded_to: Union[Unset, float] = UNSET,
    nspinmcpc_from: Union[Unset, float] = UNSET,
    nspinmcpc_to: Union[Unset, float] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    load_resource_name: Union[Unset, str] = UNSET,
    max_power_consumption_from: Union[Unset, float] = UNSET,
    max_power_consumption_to: Union[Unset, float] = UNSET,
    ecrssd_awarded_from: Union[Unset, float] = UNSET,
    ecrssd_awarded_to: Union[Unset, float] = UNSET,
    low_power_consumption_from: Union[Unset, float] = UNSET,
    low_power_consumption_to: Union[Unset, float] = UNSET,
    ecrsmd_awarded_from: Union[Unset, float] = UNSET,
    ecrsmd_awarded_to: Union[Unset, float] = UNSET,
    regup_awarded_from: Union[Unset, float] = UNSET,
    regup_awarded_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """60-Day DAM Load Resource Data

     60-Day DAM Load Resource Data

    Args:
        ecrsmcpc_from (Union[Unset, float]):
        ecrsmcpc_to (Union[Unset, float]):
        regupmcpc_from (Union[Unset, float]):
        regupmcpc_to (Union[Unset, float]):
        regdn_awarded_from (Union[Unset, float]):
        regdn_awarded_to (Union[Unset, float]):
        regdnmcpc_from (Union[Unset, float]):
        regdnmcpc_to (Union[Unset, float]):
        rrspfr_awarded_from (Union[Unset, float]):
        rrspfr_awarded_to (Union[Unset, float]):
        rrsffr_awarded_from (Union[Unset, float]):
        rrsffr_awarded_to (Union[Unset, float]):
        rrsufr_awarded_from (Union[Unset, float]):
        rrsufr_awarded_to (Union[Unset, float]):
        rrsmcpc_from (Union[Unset, float]):
        rrsmcpc_to (Union[Unset, float]):
        nspin_awarded_from (Union[Unset, float]):
        nspin_awarded_to (Union[Unset, float]):
        nspinmcpc_from (Union[Unset, float]):
        nspinmcpc_to (Union[Unset, float]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        load_resource_name (Union[Unset, str]):
        max_power_consumption_from (Union[Unset, float]):
        max_power_consumption_to (Union[Unset, float]):
        ecrssd_awarded_from (Union[Unset, float]):
        ecrssd_awarded_to (Union[Unset, float]):
        low_power_consumption_from (Union[Unset, float]):
        low_power_consumption_to (Union[Unset, float]):
        ecrsmd_awarded_from (Union[Unset, float]):
        ecrsmd_awarded_to (Union[Unset, float]):
        regup_awarded_from (Union[Unset, float]):
        regup_awarded_to (Union[Unset, float]):
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
        ecrsmcpc_from=ecrsmcpc_from,
        ecrsmcpc_to=ecrsmcpc_to,
        regupmcpc_from=regupmcpc_from,
        regupmcpc_to=regupmcpc_to,
        regdn_awarded_from=regdn_awarded_from,
        regdn_awarded_to=regdn_awarded_to,
        regdnmcpc_from=regdnmcpc_from,
        regdnmcpc_to=regdnmcpc_to,
        rrspfr_awarded_from=rrspfr_awarded_from,
        rrspfr_awarded_to=rrspfr_awarded_to,
        rrsffr_awarded_from=rrsffr_awarded_from,
        rrsffr_awarded_to=rrsffr_awarded_to,
        rrsufr_awarded_from=rrsufr_awarded_from,
        rrsufr_awarded_to=rrsufr_awarded_to,
        rrsmcpc_from=rrsmcpc_from,
        rrsmcpc_to=rrsmcpc_to,
        nspin_awarded_from=nspin_awarded_from,
        nspin_awarded_to=nspin_awarded_to,
        nspinmcpc_from=nspinmcpc_from,
        nspinmcpc_to=nspinmcpc_to,
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        load_resource_name=load_resource_name,
        max_power_consumption_from=max_power_consumption_from,
        max_power_consumption_to=max_power_consumption_to,
        ecrssd_awarded_from=ecrssd_awarded_from,
        ecrssd_awarded_to=ecrssd_awarded_to,
        low_power_consumption_from=low_power_consumption_from,
        low_power_consumption_to=low_power_consumption_to,
        ecrsmd_awarded_from=ecrsmd_awarded_from,
        ecrsmd_awarded_to=ecrsmd_awarded_to,
        regup_awarded_from=regup_awarded_from,
        regup_awarded_to=regup_awarded_to,
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
    ecrsmcpc_from: Union[Unset, float] = UNSET,
    ecrsmcpc_to: Union[Unset, float] = UNSET,
    regupmcpc_from: Union[Unset, float] = UNSET,
    regupmcpc_to: Union[Unset, float] = UNSET,
    regdn_awarded_from: Union[Unset, float] = UNSET,
    regdn_awarded_to: Union[Unset, float] = UNSET,
    regdnmcpc_from: Union[Unset, float] = UNSET,
    regdnmcpc_to: Union[Unset, float] = UNSET,
    rrspfr_awarded_from: Union[Unset, float] = UNSET,
    rrspfr_awarded_to: Union[Unset, float] = UNSET,
    rrsffr_awarded_from: Union[Unset, float] = UNSET,
    rrsffr_awarded_to: Union[Unset, float] = UNSET,
    rrsufr_awarded_from: Union[Unset, float] = UNSET,
    rrsufr_awarded_to: Union[Unset, float] = UNSET,
    rrsmcpc_from: Union[Unset, float] = UNSET,
    rrsmcpc_to: Union[Unset, float] = UNSET,
    nspin_awarded_from: Union[Unset, float] = UNSET,
    nspin_awarded_to: Union[Unset, float] = UNSET,
    nspinmcpc_from: Union[Unset, float] = UNSET,
    nspinmcpc_to: Union[Unset, float] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    load_resource_name: Union[Unset, str] = UNSET,
    max_power_consumption_from: Union[Unset, float] = UNSET,
    max_power_consumption_to: Union[Unset, float] = UNSET,
    ecrssd_awarded_from: Union[Unset, float] = UNSET,
    ecrssd_awarded_to: Union[Unset, float] = UNSET,
    low_power_consumption_from: Union[Unset, float] = UNSET,
    low_power_consumption_to: Union[Unset, float] = UNSET,
    ecrsmd_awarded_from: Union[Unset, float] = UNSET,
    ecrsmd_awarded_to: Union[Unset, float] = UNSET,
    regup_awarded_from: Union[Unset, float] = UNSET,
    regup_awarded_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """60-Day DAM Load Resource Data

     60-Day DAM Load Resource Data

    Args:
        ecrsmcpc_from (Union[Unset, float]):
        ecrsmcpc_to (Union[Unset, float]):
        regupmcpc_from (Union[Unset, float]):
        regupmcpc_to (Union[Unset, float]):
        regdn_awarded_from (Union[Unset, float]):
        regdn_awarded_to (Union[Unset, float]):
        regdnmcpc_from (Union[Unset, float]):
        regdnmcpc_to (Union[Unset, float]):
        rrspfr_awarded_from (Union[Unset, float]):
        rrspfr_awarded_to (Union[Unset, float]):
        rrsffr_awarded_from (Union[Unset, float]):
        rrsffr_awarded_to (Union[Unset, float]):
        rrsufr_awarded_from (Union[Unset, float]):
        rrsufr_awarded_to (Union[Unset, float]):
        rrsmcpc_from (Union[Unset, float]):
        rrsmcpc_to (Union[Unset, float]):
        nspin_awarded_from (Union[Unset, float]):
        nspin_awarded_to (Union[Unset, float]):
        nspinmcpc_from (Union[Unset, float]):
        nspinmcpc_to (Union[Unset, float]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        load_resource_name (Union[Unset, str]):
        max_power_consumption_from (Union[Unset, float]):
        max_power_consumption_to (Union[Unset, float]):
        ecrssd_awarded_from (Union[Unset, float]):
        ecrssd_awarded_to (Union[Unset, float]):
        low_power_consumption_from (Union[Unset, float]):
        low_power_consumption_to (Union[Unset, float]):
        ecrsmd_awarded_from (Union[Unset, float]):
        ecrsmd_awarded_to (Union[Unset, float]):
        regup_awarded_from (Union[Unset, float]):
        regup_awarded_to (Union[Unset, float]):
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
        ecrsmcpc_from=ecrsmcpc_from,
        ecrsmcpc_to=ecrsmcpc_to,
        regupmcpc_from=regupmcpc_from,
        regupmcpc_to=regupmcpc_to,
        regdn_awarded_from=regdn_awarded_from,
        regdn_awarded_to=regdn_awarded_to,
        regdnmcpc_from=regdnmcpc_from,
        regdnmcpc_to=regdnmcpc_to,
        rrspfr_awarded_from=rrspfr_awarded_from,
        rrspfr_awarded_to=rrspfr_awarded_to,
        rrsffr_awarded_from=rrsffr_awarded_from,
        rrsffr_awarded_to=rrsffr_awarded_to,
        rrsufr_awarded_from=rrsufr_awarded_from,
        rrsufr_awarded_to=rrsufr_awarded_to,
        rrsmcpc_from=rrsmcpc_from,
        rrsmcpc_to=rrsmcpc_to,
        nspin_awarded_from=nspin_awarded_from,
        nspin_awarded_to=nspin_awarded_to,
        nspinmcpc_from=nspinmcpc_from,
        nspinmcpc_to=nspinmcpc_to,
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        load_resource_name=load_resource_name,
        max_power_consumption_from=max_power_consumption_from,
        max_power_consumption_to=max_power_consumption_to,
        ecrssd_awarded_from=ecrssd_awarded_from,
        ecrssd_awarded_to=ecrssd_awarded_to,
        low_power_consumption_from=low_power_consumption_from,
        low_power_consumption_to=low_power_consumption_to,
        ecrsmd_awarded_from=ecrsmd_awarded_from,
        ecrsmd_awarded_to=ecrsmd_awarded_to,
        regup_awarded_from=regup_awarded_from,
        regup_awarded_to=regup_awarded_to,
        page=page,
        size=size,
        sort=sort,
        dir_=dir_,
        ocp_apim_subscription_key=ocp_apim_subscription_key,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    ecrsmcpc_from: Union[Unset, float] = UNSET,
    ecrsmcpc_to: Union[Unset, float] = UNSET,
    regupmcpc_from: Union[Unset, float] = UNSET,
    regupmcpc_to: Union[Unset, float] = UNSET,
    regdn_awarded_from: Union[Unset, float] = UNSET,
    regdn_awarded_to: Union[Unset, float] = UNSET,
    regdnmcpc_from: Union[Unset, float] = UNSET,
    regdnmcpc_to: Union[Unset, float] = UNSET,
    rrspfr_awarded_from: Union[Unset, float] = UNSET,
    rrspfr_awarded_to: Union[Unset, float] = UNSET,
    rrsffr_awarded_from: Union[Unset, float] = UNSET,
    rrsffr_awarded_to: Union[Unset, float] = UNSET,
    rrsufr_awarded_from: Union[Unset, float] = UNSET,
    rrsufr_awarded_to: Union[Unset, float] = UNSET,
    rrsmcpc_from: Union[Unset, float] = UNSET,
    rrsmcpc_to: Union[Unset, float] = UNSET,
    nspin_awarded_from: Union[Unset, float] = UNSET,
    nspin_awarded_to: Union[Unset, float] = UNSET,
    nspinmcpc_from: Union[Unset, float] = UNSET,
    nspinmcpc_to: Union[Unset, float] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    load_resource_name: Union[Unset, str] = UNSET,
    max_power_consumption_from: Union[Unset, float] = UNSET,
    max_power_consumption_to: Union[Unset, float] = UNSET,
    ecrssd_awarded_from: Union[Unset, float] = UNSET,
    ecrssd_awarded_to: Union[Unset, float] = UNSET,
    low_power_consumption_from: Union[Unset, float] = UNSET,
    low_power_consumption_to: Union[Unset, float] = UNSET,
    ecrsmd_awarded_from: Union[Unset, float] = UNSET,
    ecrsmd_awarded_to: Union[Unset, float] = UNSET,
    regup_awarded_from: Union[Unset, float] = UNSET,
    regup_awarded_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """60-Day DAM Load Resource Data

     60-Day DAM Load Resource Data

    Args:
        ecrsmcpc_from (Union[Unset, float]):
        ecrsmcpc_to (Union[Unset, float]):
        regupmcpc_from (Union[Unset, float]):
        regupmcpc_to (Union[Unset, float]):
        regdn_awarded_from (Union[Unset, float]):
        regdn_awarded_to (Union[Unset, float]):
        regdnmcpc_from (Union[Unset, float]):
        regdnmcpc_to (Union[Unset, float]):
        rrspfr_awarded_from (Union[Unset, float]):
        rrspfr_awarded_to (Union[Unset, float]):
        rrsffr_awarded_from (Union[Unset, float]):
        rrsffr_awarded_to (Union[Unset, float]):
        rrsufr_awarded_from (Union[Unset, float]):
        rrsufr_awarded_to (Union[Unset, float]):
        rrsmcpc_from (Union[Unset, float]):
        rrsmcpc_to (Union[Unset, float]):
        nspin_awarded_from (Union[Unset, float]):
        nspin_awarded_to (Union[Unset, float]):
        nspinmcpc_from (Union[Unset, float]):
        nspinmcpc_to (Union[Unset, float]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        load_resource_name (Union[Unset, str]):
        max_power_consumption_from (Union[Unset, float]):
        max_power_consumption_to (Union[Unset, float]):
        ecrssd_awarded_from (Union[Unset, float]):
        ecrssd_awarded_to (Union[Unset, float]):
        low_power_consumption_from (Union[Unset, float]):
        low_power_consumption_to (Union[Unset, float]):
        ecrsmd_awarded_from (Union[Unset, float]):
        ecrsmd_awarded_to (Union[Unset, float]):
        regup_awarded_from (Union[Unset, float]):
        regup_awarded_to (Union[Unset, float]):
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
        ecrsmcpc_from=ecrsmcpc_from,
        ecrsmcpc_to=ecrsmcpc_to,
        regupmcpc_from=regupmcpc_from,
        regupmcpc_to=regupmcpc_to,
        regdn_awarded_from=regdn_awarded_from,
        regdn_awarded_to=regdn_awarded_to,
        regdnmcpc_from=regdnmcpc_from,
        regdnmcpc_to=regdnmcpc_to,
        rrspfr_awarded_from=rrspfr_awarded_from,
        rrspfr_awarded_to=rrspfr_awarded_to,
        rrsffr_awarded_from=rrsffr_awarded_from,
        rrsffr_awarded_to=rrsffr_awarded_to,
        rrsufr_awarded_from=rrsufr_awarded_from,
        rrsufr_awarded_to=rrsufr_awarded_to,
        rrsmcpc_from=rrsmcpc_from,
        rrsmcpc_to=rrsmcpc_to,
        nspin_awarded_from=nspin_awarded_from,
        nspin_awarded_to=nspin_awarded_to,
        nspinmcpc_from=nspinmcpc_from,
        nspinmcpc_to=nspinmcpc_to,
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        load_resource_name=load_resource_name,
        max_power_consumption_from=max_power_consumption_from,
        max_power_consumption_to=max_power_consumption_to,
        ecrssd_awarded_from=ecrssd_awarded_from,
        ecrssd_awarded_to=ecrssd_awarded_to,
        low_power_consumption_from=low_power_consumption_from,
        low_power_consumption_to=low_power_consumption_to,
        ecrsmd_awarded_from=ecrsmd_awarded_from,
        ecrsmd_awarded_to=ecrsmd_awarded_to,
        regup_awarded_from=regup_awarded_from,
        regup_awarded_to=regup_awarded_to,
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
    ecrsmcpc_from: Union[Unset, float] = UNSET,
    ecrsmcpc_to: Union[Unset, float] = UNSET,
    regupmcpc_from: Union[Unset, float] = UNSET,
    regupmcpc_to: Union[Unset, float] = UNSET,
    regdn_awarded_from: Union[Unset, float] = UNSET,
    regdn_awarded_to: Union[Unset, float] = UNSET,
    regdnmcpc_from: Union[Unset, float] = UNSET,
    regdnmcpc_to: Union[Unset, float] = UNSET,
    rrspfr_awarded_from: Union[Unset, float] = UNSET,
    rrspfr_awarded_to: Union[Unset, float] = UNSET,
    rrsffr_awarded_from: Union[Unset, float] = UNSET,
    rrsffr_awarded_to: Union[Unset, float] = UNSET,
    rrsufr_awarded_from: Union[Unset, float] = UNSET,
    rrsufr_awarded_to: Union[Unset, float] = UNSET,
    rrsmcpc_from: Union[Unset, float] = UNSET,
    rrsmcpc_to: Union[Unset, float] = UNSET,
    nspin_awarded_from: Union[Unset, float] = UNSET,
    nspin_awarded_to: Union[Unset, float] = UNSET,
    nspinmcpc_from: Union[Unset, float] = UNSET,
    nspinmcpc_to: Union[Unset, float] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    load_resource_name: Union[Unset, str] = UNSET,
    max_power_consumption_from: Union[Unset, float] = UNSET,
    max_power_consumption_to: Union[Unset, float] = UNSET,
    ecrssd_awarded_from: Union[Unset, float] = UNSET,
    ecrssd_awarded_to: Union[Unset, float] = UNSET,
    low_power_consumption_from: Union[Unset, float] = UNSET,
    low_power_consumption_to: Union[Unset, float] = UNSET,
    ecrsmd_awarded_from: Union[Unset, float] = UNSET,
    ecrsmd_awarded_to: Union[Unset, float] = UNSET,
    regup_awarded_from: Union[Unset, float] = UNSET,
    regup_awarded_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """60-Day DAM Load Resource Data

     60-Day DAM Load Resource Data

    Args:
        ecrsmcpc_from (Union[Unset, float]):
        ecrsmcpc_to (Union[Unset, float]):
        regupmcpc_from (Union[Unset, float]):
        regupmcpc_to (Union[Unset, float]):
        regdn_awarded_from (Union[Unset, float]):
        regdn_awarded_to (Union[Unset, float]):
        regdnmcpc_from (Union[Unset, float]):
        regdnmcpc_to (Union[Unset, float]):
        rrspfr_awarded_from (Union[Unset, float]):
        rrspfr_awarded_to (Union[Unset, float]):
        rrsffr_awarded_from (Union[Unset, float]):
        rrsffr_awarded_to (Union[Unset, float]):
        rrsufr_awarded_from (Union[Unset, float]):
        rrsufr_awarded_to (Union[Unset, float]):
        rrsmcpc_from (Union[Unset, float]):
        rrsmcpc_to (Union[Unset, float]):
        nspin_awarded_from (Union[Unset, float]):
        nspin_awarded_to (Union[Unset, float]):
        nspinmcpc_from (Union[Unset, float]):
        nspinmcpc_to (Union[Unset, float]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        load_resource_name (Union[Unset, str]):
        max_power_consumption_from (Union[Unset, float]):
        max_power_consumption_to (Union[Unset, float]):
        ecrssd_awarded_from (Union[Unset, float]):
        ecrssd_awarded_to (Union[Unset, float]):
        low_power_consumption_from (Union[Unset, float]):
        low_power_consumption_to (Union[Unset, float]):
        ecrsmd_awarded_from (Union[Unset, float]):
        ecrsmd_awarded_to (Union[Unset, float]):
        regup_awarded_from (Union[Unset, float]):
        regup_awarded_to (Union[Unset, float]):
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
            ecrsmcpc_from=ecrsmcpc_from,
            ecrsmcpc_to=ecrsmcpc_to,
            regupmcpc_from=regupmcpc_from,
            regupmcpc_to=regupmcpc_to,
            regdn_awarded_from=regdn_awarded_from,
            regdn_awarded_to=regdn_awarded_to,
            regdnmcpc_from=regdnmcpc_from,
            regdnmcpc_to=regdnmcpc_to,
            rrspfr_awarded_from=rrspfr_awarded_from,
            rrspfr_awarded_to=rrspfr_awarded_to,
            rrsffr_awarded_from=rrsffr_awarded_from,
            rrsffr_awarded_to=rrsffr_awarded_to,
            rrsufr_awarded_from=rrsufr_awarded_from,
            rrsufr_awarded_to=rrsufr_awarded_to,
            rrsmcpc_from=rrsmcpc_from,
            rrsmcpc_to=rrsmcpc_to,
            nspin_awarded_from=nspin_awarded_from,
            nspin_awarded_to=nspin_awarded_to,
            nspinmcpc_from=nspinmcpc_from,
            nspinmcpc_to=nspinmcpc_to,
            delivery_date_from=delivery_date_from,
            delivery_date_to=delivery_date_to,
            hour_ending_from=hour_ending_from,
            hour_ending_to=hour_ending_to,
            load_resource_name=load_resource_name,
            max_power_consumption_from=max_power_consumption_from,
            max_power_consumption_to=max_power_consumption_to,
            ecrssd_awarded_from=ecrssd_awarded_from,
            ecrssd_awarded_to=ecrssd_awarded_to,
            low_power_consumption_from=low_power_consumption_from,
            low_power_consumption_to=low_power_consumption_to,
            ecrsmd_awarded_from=ecrsmd_awarded_from,
            ecrsmd_awarded_to=ecrsmd_awarded_to,
            regup_awarded_from=regup_awarded_from,
            regup_awarded_to=regup_awarded_to,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
