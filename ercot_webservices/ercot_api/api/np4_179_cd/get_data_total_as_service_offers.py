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
    hour_ending: Union[Unset, str] = UNSET,
    regdn_from: Union[Unset, float] = UNSET,
    regdn_to: Union[Unset, float] = UNSET,
    regup_from: Union[Unset, float] = UNSET,
    regup_to: Union[Unset, float] = UNSET,
    rrspfr_from: Union[Unset, float] = UNSET,
    rrspfr_to: Union[Unset, float] = UNSET,
    rrsffr_from: Union[Unset, float] = UNSET,
    rrsffr_to: Union[Unset, float] = UNSET,
    rrsufr_from: Union[Unset, float] = UNSET,
    rrsufr_to: Union[Unset, float] = UNSET,
    ecrssd_from: Union[Unset, float] = UNSET,
    ecrssd_to: Union[Unset, float] = UNSET,
    ecrsmd_from: Union[Unset, float] = UNSET,
    ecrsmd_to: Union[Unset, float] = UNSET,
    nspin_from: Union[Unset, float] = UNSET,
    nspin_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
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

    params["hourEnding"] = hour_ending

    params["REGDNFrom"] = regdn_from

    params["REGDNTo"] = regdn_to

    params["REGUPFrom"] = regup_from

    params["REGUPTo"] = regup_to

    params["RRSPFRFrom"] = rrspfr_from

    params["RRSPFRTo"] = rrspfr_to

    params["RRSFFRFrom"] = rrsffr_from

    params["RRSFFRTo"] = rrsffr_to

    params["RRSUFRFrom"] = rrsufr_from

    params["RRSUFRTo"] = rrsufr_to

    params["ECRSSDFrom"] = ecrssd_from

    params["ECRSSDTo"] = ecrssd_to

    params["ECRSMDFrom"] = ecrsmd_from

    params["ECRSMDTo"] = ecrsmd_to

    params["NSPINFrom"] = nspin_from

    params["NSPINTo"] = nspin_to

    params["DSTFlag"] = dst_flag

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np4-179-cd/total_as_service_offers",
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
    hour_ending: Union[Unset, str] = UNSET,
    regdn_from: Union[Unset, float] = UNSET,
    regdn_to: Union[Unset, float] = UNSET,
    regup_from: Union[Unset, float] = UNSET,
    regup_to: Union[Unset, float] = UNSET,
    rrspfr_from: Union[Unset, float] = UNSET,
    rrspfr_to: Union[Unset, float] = UNSET,
    rrsffr_from: Union[Unset, float] = UNSET,
    rrsffr_to: Union[Unset, float] = UNSET,
    rrsufr_from: Union[Unset, float] = UNSET,
    rrsufr_to: Union[Unset, float] = UNSET,
    ecrssd_from: Union[Unset, float] = UNSET,
    ecrssd_to: Union[Unset, float] = UNSET,
    ecrsmd_from: Union[Unset, float] = UNSET,
    ecrsmd_to: Union[Unset, float] = UNSET,
    nspin_from: Union[Unset, float] = UNSET,
    nspin_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """Total Ancillary Service Offers

     Total Ancillary Service Offers

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending (Union[Unset, str]):
        regdn_from (Union[Unset, float]):
        regdn_to (Union[Unset, float]):
        regup_from (Union[Unset, float]):
        regup_to (Union[Unset, float]):
        rrspfr_from (Union[Unset, float]):
        rrspfr_to (Union[Unset, float]):
        rrsffr_from (Union[Unset, float]):
        rrsffr_to (Union[Unset, float]):
        rrsufr_from (Union[Unset, float]):
        rrsufr_to (Union[Unset, float]):
        ecrssd_from (Union[Unset, float]):
        ecrssd_to (Union[Unset, float]):
        ecrsmd_from (Union[Unset, float]):
        ecrsmd_to (Union[Unset, float]):
        nspin_from (Union[Unset, float]):
        nspin_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
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
        hour_ending=hour_ending,
        regdn_from=regdn_from,
        regdn_to=regdn_to,
        regup_from=regup_from,
        regup_to=regup_to,
        rrspfr_from=rrspfr_from,
        rrspfr_to=rrspfr_to,
        rrsffr_from=rrsffr_from,
        rrsffr_to=rrsffr_to,
        rrsufr_from=rrsufr_from,
        rrsufr_to=rrsufr_to,
        ecrssd_from=ecrssd_from,
        ecrssd_to=ecrssd_to,
        ecrsmd_from=ecrsmd_from,
        ecrsmd_to=ecrsmd_to,
        nspin_from=nspin_from,
        nspin_to=nspin_to,
        dst_flag=dst_flag,
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
    hour_ending: Union[Unset, str] = UNSET,
    regdn_from: Union[Unset, float] = UNSET,
    regdn_to: Union[Unset, float] = UNSET,
    regup_from: Union[Unset, float] = UNSET,
    regup_to: Union[Unset, float] = UNSET,
    rrspfr_from: Union[Unset, float] = UNSET,
    rrspfr_to: Union[Unset, float] = UNSET,
    rrsffr_from: Union[Unset, float] = UNSET,
    rrsffr_to: Union[Unset, float] = UNSET,
    rrsufr_from: Union[Unset, float] = UNSET,
    rrsufr_to: Union[Unset, float] = UNSET,
    ecrssd_from: Union[Unset, float] = UNSET,
    ecrssd_to: Union[Unset, float] = UNSET,
    ecrsmd_from: Union[Unset, float] = UNSET,
    ecrsmd_to: Union[Unset, float] = UNSET,
    nspin_from: Union[Unset, float] = UNSET,
    nspin_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """Total Ancillary Service Offers

     Total Ancillary Service Offers

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending (Union[Unset, str]):
        regdn_from (Union[Unset, float]):
        regdn_to (Union[Unset, float]):
        regup_from (Union[Unset, float]):
        regup_to (Union[Unset, float]):
        rrspfr_from (Union[Unset, float]):
        rrspfr_to (Union[Unset, float]):
        rrsffr_from (Union[Unset, float]):
        rrsffr_to (Union[Unset, float]):
        rrsufr_from (Union[Unset, float]):
        rrsufr_to (Union[Unset, float]):
        ecrssd_from (Union[Unset, float]):
        ecrssd_to (Union[Unset, float]):
        ecrsmd_from (Union[Unset, float]):
        ecrsmd_to (Union[Unset, float]):
        nspin_from (Union[Unset, float]):
        nspin_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
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
        hour_ending=hour_ending,
        regdn_from=regdn_from,
        regdn_to=regdn_to,
        regup_from=regup_from,
        regup_to=regup_to,
        rrspfr_from=rrspfr_from,
        rrspfr_to=rrspfr_to,
        rrsffr_from=rrsffr_from,
        rrsffr_to=rrsffr_to,
        rrsufr_from=rrsufr_from,
        rrsufr_to=rrsufr_to,
        ecrssd_from=ecrssd_from,
        ecrssd_to=ecrssd_to,
        ecrsmd_from=ecrsmd_from,
        ecrsmd_to=ecrsmd_to,
        nspin_from=nspin_from,
        nspin_to=nspin_to,
        dst_flag=dst_flag,
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
    hour_ending: Union[Unset, str] = UNSET,
    regdn_from: Union[Unset, float] = UNSET,
    regdn_to: Union[Unset, float] = UNSET,
    regup_from: Union[Unset, float] = UNSET,
    regup_to: Union[Unset, float] = UNSET,
    rrspfr_from: Union[Unset, float] = UNSET,
    rrspfr_to: Union[Unset, float] = UNSET,
    rrsffr_from: Union[Unset, float] = UNSET,
    rrsffr_to: Union[Unset, float] = UNSET,
    rrsufr_from: Union[Unset, float] = UNSET,
    rrsufr_to: Union[Unset, float] = UNSET,
    ecrssd_from: Union[Unset, float] = UNSET,
    ecrssd_to: Union[Unset, float] = UNSET,
    ecrsmd_from: Union[Unset, float] = UNSET,
    ecrsmd_to: Union[Unset, float] = UNSET,
    nspin_from: Union[Unset, float] = UNSET,
    nspin_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """Total Ancillary Service Offers

     Total Ancillary Service Offers

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending (Union[Unset, str]):
        regdn_from (Union[Unset, float]):
        regdn_to (Union[Unset, float]):
        regup_from (Union[Unset, float]):
        regup_to (Union[Unset, float]):
        rrspfr_from (Union[Unset, float]):
        rrspfr_to (Union[Unset, float]):
        rrsffr_from (Union[Unset, float]):
        rrsffr_to (Union[Unset, float]):
        rrsufr_from (Union[Unset, float]):
        rrsufr_to (Union[Unset, float]):
        ecrssd_from (Union[Unset, float]):
        ecrssd_to (Union[Unset, float]):
        ecrsmd_from (Union[Unset, float]):
        ecrsmd_to (Union[Unset, float]):
        nspin_from (Union[Unset, float]):
        nspin_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
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
        hour_ending=hour_ending,
        regdn_from=regdn_from,
        regdn_to=regdn_to,
        regup_from=regup_from,
        regup_to=regup_to,
        rrspfr_from=rrspfr_from,
        rrspfr_to=rrspfr_to,
        rrsffr_from=rrsffr_from,
        rrsffr_to=rrsffr_to,
        rrsufr_from=rrsufr_from,
        rrsufr_to=rrsufr_to,
        ecrssd_from=ecrssd_from,
        ecrssd_to=ecrssd_to,
        ecrsmd_from=ecrsmd_from,
        ecrsmd_to=ecrsmd_to,
        nspin_from=nspin_from,
        nspin_to=nspin_to,
        dst_flag=dst_flag,
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
    hour_ending: Union[Unset, str] = UNSET,
    regdn_from: Union[Unset, float] = UNSET,
    regdn_to: Union[Unset, float] = UNSET,
    regup_from: Union[Unset, float] = UNSET,
    regup_to: Union[Unset, float] = UNSET,
    rrspfr_from: Union[Unset, float] = UNSET,
    rrspfr_to: Union[Unset, float] = UNSET,
    rrsffr_from: Union[Unset, float] = UNSET,
    rrsffr_to: Union[Unset, float] = UNSET,
    rrsufr_from: Union[Unset, float] = UNSET,
    rrsufr_to: Union[Unset, float] = UNSET,
    ecrssd_from: Union[Unset, float] = UNSET,
    ecrssd_to: Union[Unset, float] = UNSET,
    ecrsmd_from: Union[Unset, float] = UNSET,
    ecrsmd_to: Union[Unset, float] = UNSET,
    nspin_from: Union[Unset, float] = UNSET,
    nspin_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """Total Ancillary Service Offers

     Total Ancillary Service Offers

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending (Union[Unset, str]):
        regdn_from (Union[Unset, float]):
        regdn_to (Union[Unset, float]):
        regup_from (Union[Unset, float]):
        regup_to (Union[Unset, float]):
        rrspfr_from (Union[Unset, float]):
        rrspfr_to (Union[Unset, float]):
        rrsffr_from (Union[Unset, float]):
        rrsffr_to (Union[Unset, float]):
        rrsufr_from (Union[Unset, float]):
        rrsufr_to (Union[Unset, float]):
        ecrssd_from (Union[Unset, float]):
        ecrssd_to (Union[Unset, float]):
        ecrsmd_from (Union[Unset, float]):
        ecrsmd_to (Union[Unset, float]):
        nspin_from (Union[Unset, float]):
        nspin_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
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
            hour_ending=hour_ending,
            regdn_from=regdn_from,
            regdn_to=regdn_to,
            regup_from=regup_from,
            regup_to=regup_to,
            rrspfr_from=rrspfr_from,
            rrspfr_to=rrspfr_to,
            rrsffr_from=rrsffr_from,
            rrsffr_to=rrsffr_to,
            rrsufr_from=rrsufr_from,
            rrsufr_to=rrsufr_to,
            ecrssd_from=ecrssd_from,
            ecrssd_to=ecrssd_to,
            ecrsmd_from=ecrsmd_from,
            ecrsmd_to=ecrsmd_to,
            nspin_from=nspin_from,
            nspin_to=nspin_to,
            dst_flag=dst_flag,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
