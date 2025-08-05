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
    qse_name: Union[Unset, str] = UNSET,
    resource_name: Union[Unset, str] = UNSET,
    hour_ending: Union[Unset, str] = UNSET,
    status: Union[Unset, str] = UNSET,
    high_sustained_limit_from: Union[Unset, float] = UNSET,
    high_sustained_limit_to: Union[Unset, float] = UNSET,
    low_sustained_limit_from: Union[Unset, float] = UNSET,
    low_sustained_limit_to: Union[Unset, float] = UNSET,
    high_emergency_limit_from: Union[Unset, float] = UNSET,
    high_emergency_limit_to: Union[Unset, float] = UNSET,
    low_emergency_limit_from: Union[Unset, float] = UNSET,
    low_emergency_limit_to: Union[Unset, float] = UNSET,
    regup_from: Union[Unset, float] = UNSET,
    regup_to: Union[Unset, float] = UNSET,
    regdn_from: Union[Unset, float] = UNSET,
    regdn_to: Union[Unset, float] = UNSET,
    rrspfr_from: Union[Unset, float] = UNSET,
    rrspfr_to: Union[Unset, float] = UNSET,
    rrsffr_from: Union[Unset, float] = UNSET,
    rrsffr_to: Union[Unset, float] = UNSET,
    rrsufr_from: Union[Unset, float] = UNSET,
    rrsufr_to: Union[Unset, float] = UNSET,
    nspin_from: Union[Unset, float] = UNSET,
    nspin_to: Union[Unset, float] = UNSET,
    ecrs_from: Union[Unset, float] = UNSET,
    ecrs_to: Union[Unset, float] = UNSET,
    min_soc_from: Union[Unset, float] = UNSET,
    min_soc_to: Union[Unset, float] = UNSET,
    max_soc_from: Union[Unset, float] = UNSET,
    max_soc_to: Union[Unset, float] = UNSET,
    hour_beginning_planned_soc_from: Union[Unset, float] = UNSET,
    hour_beginning_planned_soc_to: Union[Unset, float] = UNSET,
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

    params["qseName"] = qse_name

    params["resourceName"] = resource_name

    params["hourEnding"] = hour_ending

    params["status"] = status

    params["highSustainedLimitFrom"] = high_sustained_limit_from

    params["highSustainedLimitTo"] = high_sustained_limit_to

    params["lowSustainedLimitFrom"] = low_sustained_limit_from

    params["lowSustainedLimitTo"] = low_sustained_limit_to

    params["highEmergencyLimitFrom"] = high_emergency_limit_from

    params["highEmergencyLimitTo"] = high_emergency_limit_to

    params["lowEmergencyLimitFrom"] = low_emergency_limit_from

    params["lowEmergencyLimitTo"] = low_emergency_limit_to

    params["REGUPFrom"] = regup_from

    params["REGUPTo"] = regup_to

    params["REGDNFrom"] = regdn_from

    params["REGDNTo"] = regdn_to

    params["RRSPFRFrom"] = rrspfr_from

    params["RRSPFRTo"] = rrspfr_to

    params["RRSFFRFrom"] = rrsffr_from

    params["RRSFFRTo"] = rrsffr_to

    params["RRSUFRFrom"] = rrsufr_from

    params["RRSUFRTo"] = rrsufr_to

    params["NSPINFrom"] = nspin_from

    params["NSPINTo"] = nspin_to

    params["ECRSFrom"] = ecrs_from

    params["ECRSTo"] = ecrs_to

    params["minSOCFrom"] = min_soc_from

    params["minSOCTo"] = min_soc_to

    params["maxSOCFrom"] = max_soc_from

    params["maxSOCTo"] = max_soc_to

    params["hourBeginningPlannedSOCFrom"] = hour_beginning_planned_soc_from

    params["hourBeginningPlannedSOCTo"] = hour_beginning_planned_soc_to

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np1-301/60_cop_adj_period_snapshot",
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
    qse_name: Union[Unset, str] = UNSET,
    resource_name: Union[Unset, str] = UNSET,
    hour_ending: Union[Unset, str] = UNSET,
    status: Union[Unset, str] = UNSET,
    high_sustained_limit_from: Union[Unset, float] = UNSET,
    high_sustained_limit_to: Union[Unset, float] = UNSET,
    low_sustained_limit_from: Union[Unset, float] = UNSET,
    low_sustained_limit_to: Union[Unset, float] = UNSET,
    high_emergency_limit_from: Union[Unset, float] = UNSET,
    high_emergency_limit_to: Union[Unset, float] = UNSET,
    low_emergency_limit_from: Union[Unset, float] = UNSET,
    low_emergency_limit_to: Union[Unset, float] = UNSET,
    regup_from: Union[Unset, float] = UNSET,
    regup_to: Union[Unset, float] = UNSET,
    regdn_from: Union[Unset, float] = UNSET,
    regdn_to: Union[Unset, float] = UNSET,
    rrspfr_from: Union[Unset, float] = UNSET,
    rrspfr_to: Union[Unset, float] = UNSET,
    rrsffr_from: Union[Unset, float] = UNSET,
    rrsffr_to: Union[Unset, float] = UNSET,
    rrsufr_from: Union[Unset, float] = UNSET,
    rrsufr_to: Union[Unset, float] = UNSET,
    nspin_from: Union[Unset, float] = UNSET,
    nspin_to: Union[Unset, float] = UNSET,
    ecrs_from: Union[Unset, float] = UNSET,
    ecrs_to: Union[Unset, float] = UNSET,
    min_soc_from: Union[Unset, float] = UNSET,
    min_soc_to: Union[Unset, float] = UNSET,
    max_soc_from: Union[Unset, float] = UNSET,
    max_soc_to: Union[Unset, float] = UNSET,
    hour_beginning_planned_soc_from: Union[Unset, float] = UNSET,
    hour_beginning_planned_soc_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """60-Day COP Adjustment Period Snapshot

     60-Day COP Adjustment Period Snapshot

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        qse_name (Union[Unset, str]):
        resource_name (Union[Unset, str]):
        hour_ending (Union[Unset, str]):
        status (Union[Unset, str]):
        high_sustained_limit_from (Union[Unset, float]):
        high_sustained_limit_to (Union[Unset, float]):
        low_sustained_limit_from (Union[Unset, float]):
        low_sustained_limit_to (Union[Unset, float]):
        high_emergency_limit_from (Union[Unset, float]):
        high_emergency_limit_to (Union[Unset, float]):
        low_emergency_limit_from (Union[Unset, float]):
        low_emergency_limit_to (Union[Unset, float]):
        regup_from (Union[Unset, float]):
        regup_to (Union[Unset, float]):
        regdn_from (Union[Unset, float]):
        regdn_to (Union[Unset, float]):
        rrspfr_from (Union[Unset, float]):
        rrspfr_to (Union[Unset, float]):
        rrsffr_from (Union[Unset, float]):
        rrsffr_to (Union[Unset, float]):
        rrsufr_from (Union[Unset, float]):
        rrsufr_to (Union[Unset, float]):
        nspin_from (Union[Unset, float]):
        nspin_to (Union[Unset, float]):
        ecrs_from (Union[Unset, float]):
        ecrs_to (Union[Unset, float]):
        min_soc_from (Union[Unset, float]):
        min_soc_to (Union[Unset, float]):
        max_soc_from (Union[Unset, float]):
        max_soc_to (Union[Unset, float]):
        hour_beginning_planned_soc_from (Union[Unset, float]):
        hour_beginning_planned_soc_to (Union[Unset, float]):
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
        qse_name=qse_name,
        resource_name=resource_name,
        hour_ending=hour_ending,
        status=status,
        high_sustained_limit_from=high_sustained_limit_from,
        high_sustained_limit_to=high_sustained_limit_to,
        low_sustained_limit_from=low_sustained_limit_from,
        low_sustained_limit_to=low_sustained_limit_to,
        high_emergency_limit_from=high_emergency_limit_from,
        high_emergency_limit_to=high_emergency_limit_to,
        low_emergency_limit_from=low_emergency_limit_from,
        low_emergency_limit_to=low_emergency_limit_to,
        regup_from=regup_from,
        regup_to=regup_to,
        regdn_from=regdn_from,
        regdn_to=regdn_to,
        rrspfr_from=rrspfr_from,
        rrspfr_to=rrspfr_to,
        rrsffr_from=rrsffr_from,
        rrsffr_to=rrsffr_to,
        rrsufr_from=rrsufr_from,
        rrsufr_to=rrsufr_to,
        nspin_from=nspin_from,
        nspin_to=nspin_to,
        ecrs_from=ecrs_from,
        ecrs_to=ecrs_to,
        min_soc_from=min_soc_from,
        min_soc_to=min_soc_to,
        max_soc_from=max_soc_from,
        max_soc_to=max_soc_to,
        hour_beginning_planned_soc_from=hour_beginning_planned_soc_from,
        hour_beginning_planned_soc_to=hour_beginning_planned_soc_to,
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
    qse_name: Union[Unset, str] = UNSET,
    resource_name: Union[Unset, str] = UNSET,
    hour_ending: Union[Unset, str] = UNSET,
    status: Union[Unset, str] = UNSET,
    high_sustained_limit_from: Union[Unset, float] = UNSET,
    high_sustained_limit_to: Union[Unset, float] = UNSET,
    low_sustained_limit_from: Union[Unset, float] = UNSET,
    low_sustained_limit_to: Union[Unset, float] = UNSET,
    high_emergency_limit_from: Union[Unset, float] = UNSET,
    high_emergency_limit_to: Union[Unset, float] = UNSET,
    low_emergency_limit_from: Union[Unset, float] = UNSET,
    low_emergency_limit_to: Union[Unset, float] = UNSET,
    regup_from: Union[Unset, float] = UNSET,
    regup_to: Union[Unset, float] = UNSET,
    regdn_from: Union[Unset, float] = UNSET,
    regdn_to: Union[Unset, float] = UNSET,
    rrspfr_from: Union[Unset, float] = UNSET,
    rrspfr_to: Union[Unset, float] = UNSET,
    rrsffr_from: Union[Unset, float] = UNSET,
    rrsffr_to: Union[Unset, float] = UNSET,
    rrsufr_from: Union[Unset, float] = UNSET,
    rrsufr_to: Union[Unset, float] = UNSET,
    nspin_from: Union[Unset, float] = UNSET,
    nspin_to: Union[Unset, float] = UNSET,
    ecrs_from: Union[Unset, float] = UNSET,
    ecrs_to: Union[Unset, float] = UNSET,
    min_soc_from: Union[Unset, float] = UNSET,
    min_soc_to: Union[Unset, float] = UNSET,
    max_soc_from: Union[Unset, float] = UNSET,
    max_soc_to: Union[Unset, float] = UNSET,
    hour_beginning_planned_soc_from: Union[Unset, float] = UNSET,
    hour_beginning_planned_soc_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """60-Day COP Adjustment Period Snapshot

     60-Day COP Adjustment Period Snapshot

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        qse_name (Union[Unset, str]):
        resource_name (Union[Unset, str]):
        hour_ending (Union[Unset, str]):
        status (Union[Unset, str]):
        high_sustained_limit_from (Union[Unset, float]):
        high_sustained_limit_to (Union[Unset, float]):
        low_sustained_limit_from (Union[Unset, float]):
        low_sustained_limit_to (Union[Unset, float]):
        high_emergency_limit_from (Union[Unset, float]):
        high_emergency_limit_to (Union[Unset, float]):
        low_emergency_limit_from (Union[Unset, float]):
        low_emergency_limit_to (Union[Unset, float]):
        regup_from (Union[Unset, float]):
        regup_to (Union[Unset, float]):
        regdn_from (Union[Unset, float]):
        regdn_to (Union[Unset, float]):
        rrspfr_from (Union[Unset, float]):
        rrspfr_to (Union[Unset, float]):
        rrsffr_from (Union[Unset, float]):
        rrsffr_to (Union[Unset, float]):
        rrsufr_from (Union[Unset, float]):
        rrsufr_to (Union[Unset, float]):
        nspin_from (Union[Unset, float]):
        nspin_to (Union[Unset, float]):
        ecrs_from (Union[Unset, float]):
        ecrs_to (Union[Unset, float]):
        min_soc_from (Union[Unset, float]):
        min_soc_to (Union[Unset, float]):
        max_soc_from (Union[Unset, float]):
        max_soc_to (Union[Unset, float]):
        hour_beginning_planned_soc_from (Union[Unset, float]):
        hour_beginning_planned_soc_to (Union[Unset, float]):
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
        qse_name=qse_name,
        resource_name=resource_name,
        hour_ending=hour_ending,
        status=status,
        high_sustained_limit_from=high_sustained_limit_from,
        high_sustained_limit_to=high_sustained_limit_to,
        low_sustained_limit_from=low_sustained_limit_from,
        low_sustained_limit_to=low_sustained_limit_to,
        high_emergency_limit_from=high_emergency_limit_from,
        high_emergency_limit_to=high_emergency_limit_to,
        low_emergency_limit_from=low_emergency_limit_from,
        low_emergency_limit_to=low_emergency_limit_to,
        regup_from=regup_from,
        regup_to=regup_to,
        regdn_from=regdn_from,
        regdn_to=regdn_to,
        rrspfr_from=rrspfr_from,
        rrspfr_to=rrspfr_to,
        rrsffr_from=rrsffr_from,
        rrsffr_to=rrsffr_to,
        rrsufr_from=rrsufr_from,
        rrsufr_to=rrsufr_to,
        nspin_from=nspin_from,
        nspin_to=nspin_to,
        ecrs_from=ecrs_from,
        ecrs_to=ecrs_to,
        min_soc_from=min_soc_from,
        min_soc_to=min_soc_to,
        max_soc_from=max_soc_from,
        max_soc_to=max_soc_to,
        hour_beginning_planned_soc_from=hour_beginning_planned_soc_from,
        hour_beginning_planned_soc_to=hour_beginning_planned_soc_to,
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
    qse_name: Union[Unset, str] = UNSET,
    resource_name: Union[Unset, str] = UNSET,
    hour_ending: Union[Unset, str] = UNSET,
    status: Union[Unset, str] = UNSET,
    high_sustained_limit_from: Union[Unset, float] = UNSET,
    high_sustained_limit_to: Union[Unset, float] = UNSET,
    low_sustained_limit_from: Union[Unset, float] = UNSET,
    low_sustained_limit_to: Union[Unset, float] = UNSET,
    high_emergency_limit_from: Union[Unset, float] = UNSET,
    high_emergency_limit_to: Union[Unset, float] = UNSET,
    low_emergency_limit_from: Union[Unset, float] = UNSET,
    low_emergency_limit_to: Union[Unset, float] = UNSET,
    regup_from: Union[Unset, float] = UNSET,
    regup_to: Union[Unset, float] = UNSET,
    regdn_from: Union[Unset, float] = UNSET,
    regdn_to: Union[Unset, float] = UNSET,
    rrspfr_from: Union[Unset, float] = UNSET,
    rrspfr_to: Union[Unset, float] = UNSET,
    rrsffr_from: Union[Unset, float] = UNSET,
    rrsffr_to: Union[Unset, float] = UNSET,
    rrsufr_from: Union[Unset, float] = UNSET,
    rrsufr_to: Union[Unset, float] = UNSET,
    nspin_from: Union[Unset, float] = UNSET,
    nspin_to: Union[Unset, float] = UNSET,
    ecrs_from: Union[Unset, float] = UNSET,
    ecrs_to: Union[Unset, float] = UNSET,
    min_soc_from: Union[Unset, float] = UNSET,
    min_soc_to: Union[Unset, float] = UNSET,
    max_soc_from: Union[Unset, float] = UNSET,
    max_soc_to: Union[Unset, float] = UNSET,
    hour_beginning_planned_soc_from: Union[Unset, float] = UNSET,
    hour_beginning_planned_soc_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """60-Day COP Adjustment Period Snapshot

     60-Day COP Adjustment Period Snapshot

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        qse_name (Union[Unset, str]):
        resource_name (Union[Unset, str]):
        hour_ending (Union[Unset, str]):
        status (Union[Unset, str]):
        high_sustained_limit_from (Union[Unset, float]):
        high_sustained_limit_to (Union[Unset, float]):
        low_sustained_limit_from (Union[Unset, float]):
        low_sustained_limit_to (Union[Unset, float]):
        high_emergency_limit_from (Union[Unset, float]):
        high_emergency_limit_to (Union[Unset, float]):
        low_emergency_limit_from (Union[Unset, float]):
        low_emergency_limit_to (Union[Unset, float]):
        regup_from (Union[Unset, float]):
        regup_to (Union[Unset, float]):
        regdn_from (Union[Unset, float]):
        regdn_to (Union[Unset, float]):
        rrspfr_from (Union[Unset, float]):
        rrspfr_to (Union[Unset, float]):
        rrsffr_from (Union[Unset, float]):
        rrsffr_to (Union[Unset, float]):
        rrsufr_from (Union[Unset, float]):
        rrsufr_to (Union[Unset, float]):
        nspin_from (Union[Unset, float]):
        nspin_to (Union[Unset, float]):
        ecrs_from (Union[Unset, float]):
        ecrs_to (Union[Unset, float]):
        min_soc_from (Union[Unset, float]):
        min_soc_to (Union[Unset, float]):
        max_soc_from (Union[Unset, float]):
        max_soc_to (Union[Unset, float]):
        hour_beginning_planned_soc_from (Union[Unset, float]):
        hour_beginning_planned_soc_to (Union[Unset, float]):
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
        qse_name=qse_name,
        resource_name=resource_name,
        hour_ending=hour_ending,
        status=status,
        high_sustained_limit_from=high_sustained_limit_from,
        high_sustained_limit_to=high_sustained_limit_to,
        low_sustained_limit_from=low_sustained_limit_from,
        low_sustained_limit_to=low_sustained_limit_to,
        high_emergency_limit_from=high_emergency_limit_from,
        high_emergency_limit_to=high_emergency_limit_to,
        low_emergency_limit_from=low_emergency_limit_from,
        low_emergency_limit_to=low_emergency_limit_to,
        regup_from=regup_from,
        regup_to=regup_to,
        regdn_from=regdn_from,
        regdn_to=regdn_to,
        rrspfr_from=rrspfr_from,
        rrspfr_to=rrspfr_to,
        rrsffr_from=rrsffr_from,
        rrsffr_to=rrsffr_to,
        rrsufr_from=rrsufr_from,
        rrsufr_to=rrsufr_to,
        nspin_from=nspin_from,
        nspin_to=nspin_to,
        ecrs_from=ecrs_from,
        ecrs_to=ecrs_to,
        min_soc_from=min_soc_from,
        min_soc_to=min_soc_to,
        max_soc_from=max_soc_from,
        max_soc_to=max_soc_to,
        hour_beginning_planned_soc_from=hour_beginning_planned_soc_from,
        hour_beginning_planned_soc_to=hour_beginning_planned_soc_to,
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
    qse_name: Union[Unset, str] = UNSET,
    resource_name: Union[Unset, str] = UNSET,
    hour_ending: Union[Unset, str] = UNSET,
    status: Union[Unset, str] = UNSET,
    high_sustained_limit_from: Union[Unset, float] = UNSET,
    high_sustained_limit_to: Union[Unset, float] = UNSET,
    low_sustained_limit_from: Union[Unset, float] = UNSET,
    low_sustained_limit_to: Union[Unset, float] = UNSET,
    high_emergency_limit_from: Union[Unset, float] = UNSET,
    high_emergency_limit_to: Union[Unset, float] = UNSET,
    low_emergency_limit_from: Union[Unset, float] = UNSET,
    low_emergency_limit_to: Union[Unset, float] = UNSET,
    regup_from: Union[Unset, float] = UNSET,
    regup_to: Union[Unset, float] = UNSET,
    regdn_from: Union[Unset, float] = UNSET,
    regdn_to: Union[Unset, float] = UNSET,
    rrspfr_from: Union[Unset, float] = UNSET,
    rrspfr_to: Union[Unset, float] = UNSET,
    rrsffr_from: Union[Unset, float] = UNSET,
    rrsffr_to: Union[Unset, float] = UNSET,
    rrsufr_from: Union[Unset, float] = UNSET,
    rrsufr_to: Union[Unset, float] = UNSET,
    nspin_from: Union[Unset, float] = UNSET,
    nspin_to: Union[Unset, float] = UNSET,
    ecrs_from: Union[Unset, float] = UNSET,
    ecrs_to: Union[Unset, float] = UNSET,
    min_soc_from: Union[Unset, float] = UNSET,
    min_soc_to: Union[Unset, float] = UNSET,
    max_soc_from: Union[Unset, float] = UNSET,
    max_soc_to: Union[Unset, float] = UNSET,
    hour_beginning_planned_soc_from: Union[Unset, float] = UNSET,
    hour_beginning_planned_soc_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """60-Day COP Adjustment Period Snapshot

     60-Day COP Adjustment Period Snapshot

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        qse_name (Union[Unset, str]):
        resource_name (Union[Unset, str]):
        hour_ending (Union[Unset, str]):
        status (Union[Unset, str]):
        high_sustained_limit_from (Union[Unset, float]):
        high_sustained_limit_to (Union[Unset, float]):
        low_sustained_limit_from (Union[Unset, float]):
        low_sustained_limit_to (Union[Unset, float]):
        high_emergency_limit_from (Union[Unset, float]):
        high_emergency_limit_to (Union[Unset, float]):
        low_emergency_limit_from (Union[Unset, float]):
        low_emergency_limit_to (Union[Unset, float]):
        regup_from (Union[Unset, float]):
        regup_to (Union[Unset, float]):
        regdn_from (Union[Unset, float]):
        regdn_to (Union[Unset, float]):
        rrspfr_from (Union[Unset, float]):
        rrspfr_to (Union[Unset, float]):
        rrsffr_from (Union[Unset, float]):
        rrsffr_to (Union[Unset, float]):
        rrsufr_from (Union[Unset, float]):
        rrsufr_to (Union[Unset, float]):
        nspin_from (Union[Unset, float]):
        nspin_to (Union[Unset, float]):
        ecrs_from (Union[Unset, float]):
        ecrs_to (Union[Unset, float]):
        min_soc_from (Union[Unset, float]):
        min_soc_to (Union[Unset, float]):
        max_soc_from (Union[Unset, float]):
        max_soc_to (Union[Unset, float]):
        hour_beginning_planned_soc_from (Union[Unset, float]):
        hour_beginning_planned_soc_to (Union[Unset, float]):
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
            qse_name=qse_name,
            resource_name=resource_name,
            hour_ending=hour_ending,
            status=status,
            high_sustained_limit_from=high_sustained_limit_from,
            high_sustained_limit_to=high_sustained_limit_to,
            low_sustained_limit_from=low_sustained_limit_from,
            low_sustained_limit_to=low_sustained_limit_to,
            high_emergency_limit_from=high_emergency_limit_from,
            high_emergency_limit_to=high_emergency_limit_to,
            low_emergency_limit_from=low_emergency_limit_from,
            low_emergency_limit_to=low_emergency_limit_to,
            regup_from=regup_from,
            regup_to=regup_to,
            regdn_from=regdn_from,
            regdn_to=regdn_to,
            rrspfr_from=rrspfr_from,
            rrspfr_to=rrspfr_to,
            rrsffr_from=rrsffr_from,
            rrsffr_to=rrsffr_to,
            rrsufr_from=rrsufr_from,
            rrsufr_to=rrsufr_to,
            nspin_from=nspin_from,
            nspin_to=nspin_to,
            ecrs_from=ecrs_from,
            ecrs_to=ecrs_to,
            min_soc_from=min_soc_from,
            min_soc_to=min_soc_to,
            max_soc_from=max_soc_from,
            max_soc_to=max_soc_to,
            hour_beginning_planned_soc_from=hour_beginning_planned_soc_from,
            hour_beginning_planned_soc_to=hour_beginning_planned_soc_to,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
