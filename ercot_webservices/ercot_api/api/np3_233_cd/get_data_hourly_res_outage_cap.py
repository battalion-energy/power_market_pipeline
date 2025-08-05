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
    operating_date_from: Union[Unset, str] = UNSET,
    operating_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    total_resource_mw_zone_south_from: Union[Unset, int] = UNSET,
    total_resource_mw_zone_south_to: Union[Unset, int] = UNSET,
    total_resource_mw_zone_north_from: Union[Unset, int] = UNSET,
    total_resource_mw_zone_north_to: Union[Unset, int] = UNSET,
    total_resource_mw_zone_west_from: Union[Unset, int] = UNSET,
    total_resource_mw_zone_west_to: Union[Unset, int] = UNSET,
    total_resource_mw_zone_houston_from: Union[Unset, int] = UNSET,
    total_resource_mw_zone_houston_to: Union[Unset, int] = UNSET,
    total_irrmw_zone_south_from: Union[Unset, int] = UNSET,
    total_irrmw_zone_south_to: Union[Unset, int] = UNSET,
    total_irrmw_zone_north_from: Union[Unset, int] = UNSET,
    total_irrmw_zone_north_to: Union[Unset, int] = UNSET,
    total_irrmw_zone_west_from: Union[Unset, int] = UNSET,
    total_irrmw_zone_west_to: Union[Unset, int] = UNSET,
    total_irrmw_zone_houston_from: Union[Unset, int] = UNSET,
    total_irrmw_zone_houston_to: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_south_from: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_south_to: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_north_from: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_north_to: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_west_from: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_west_to: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_houston_from: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_houston_to: Union[Unset, int] = UNSET,
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

    params["operatingDateFrom"] = operating_date_from

    params["operatingDateTo"] = operating_date_to

    params["hourEndingFrom"] = hour_ending_from

    params["hourEndingTo"] = hour_ending_to

    params["totalResourceMWZoneSouthFrom"] = total_resource_mw_zone_south_from

    params["totalResourceMWZoneSouthTo"] = total_resource_mw_zone_south_to

    params["totalResourceMWZoneNorthFrom"] = total_resource_mw_zone_north_from

    params["totalResourceMWZoneNorthTo"] = total_resource_mw_zone_north_to

    params["totalResourceMWZoneWestFrom"] = total_resource_mw_zone_west_from

    params["totalResourceMWZoneWestTo"] = total_resource_mw_zone_west_to

    params["totalResourceMWZoneHoustonFrom"] = total_resource_mw_zone_houston_from

    params["totalResourceMWZoneHoustonTo"] = total_resource_mw_zone_houston_to

    params["totalIRRMWZoneSouthFrom"] = total_irrmw_zone_south_from

    params["totalIRRMWZoneSouthTo"] = total_irrmw_zone_south_to

    params["totalIRRMWZoneNorthFrom"] = total_irrmw_zone_north_from

    params["totalIRRMWZoneNorthTo"] = total_irrmw_zone_north_to

    params["totalIRRMWZoneWestFrom"] = total_irrmw_zone_west_from

    params["totalIRRMWZoneWestTo"] = total_irrmw_zone_west_to

    params["totalIRRMWZoneHoustonFrom"] = total_irrmw_zone_houston_from

    params["totalIRRMWZoneHoustonTo"] = total_irrmw_zone_houston_to

    params["totalNewEquipResourceMWZoneSouthFrom"] = total_new_equip_resource_mw_zone_south_from

    params["totalNewEquipResourceMWZoneSouthTo"] = total_new_equip_resource_mw_zone_south_to

    params["totalNewEquipResourceMWZoneNorthFrom"] = total_new_equip_resource_mw_zone_north_from

    params["totalNewEquipResourceMWZoneNorthTo"] = total_new_equip_resource_mw_zone_north_to

    params["totalNewEquipResourceMWZoneWestFrom"] = total_new_equip_resource_mw_zone_west_from

    params["totalNewEquipResourceMWZoneWestTo"] = total_new_equip_resource_mw_zone_west_to

    params["totalNewEquipResourceMWZoneHoustonFrom"] = total_new_equip_resource_mw_zone_houston_from

    params["totalNewEquipResourceMWZoneHoustonTo"] = total_new_equip_resource_mw_zone_houston_to

    params["postedDatetimeFrom"] = posted_datetime_from

    params["postedDatetimeTo"] = posted_datetime_to

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np3-233-cd/hourly_res_outage_cap",
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
    operating_date_from: Union[Unset, str] = UNSET,
    operating_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    total_resource_mw_zone_south_from: Union[Unset, int] = UNSET,
    total_resource_mw_zone_south_to: Union[Unset, int] = UNSET,
    total_resource_mw_zone_north_from: Union[Unset, int] = UNSET,
    total_resource_mw_zone_north_to: Union[Unset, int] = UNSET,
    total_resource_mw_zone_west_from: Union[Unset, int] = UNSET,
    total_resource_mw_zone_west_to: Union[Unset, int] = UNSET,
    total_resource_mw_zone_houston_from: Union[Unset, int] = UNSET,
    total_resource_mw_zone_houston_to: Union[Unset, int] = UNSET,
    total_irrmw_zone_south_from: Union[Unset, int] = UNSET,
    total_irrmw_zone_south_to: Union[Unset, int] = UNSET,
    total_irrmw_zone_north_from: Union[Unset, int] = UNSET,
    total_irrmw_zone_north_to: Union[Unset, int] = UNSET,
    total_irrmw_zone_west_from: Union[Unset, int] = UNSET,
    total_irrmw_zone_west_to: Union[Unset, int] = UNSET,
    total_irrmw_zone_houston_from: Union[Unset, int] = UNSET,
    total_irrmw_zone_houston_to: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_south_from: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_south_to: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_north_from: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_north_to: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_west_from: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_west_to: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_houston_from: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_houston_to: Union[Unset, int] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """Hourly Resource Outage Capacity

     Hourly Resource Outage Capacity

    Args:
        operating_date_from (Union[Unset, str]):
        operating_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        total_resource_mw_zone_south_from (Union[Unset, int]):
        total_resource_mw_zone_south_to (Union[Unset, int]):
        total_resource_mw_zone_north_from (Union[Unset, int]):
        total_resource_mw_zone_north_to (Union[Unset, int]):
        total_resource_mw_zone_west_from (Union[Unset, int]):
        total_resource_mw_zone_west_to (Union[Unset, int]):
        total_resource_mw_zone_houston_from (Union[Unset, int]):
        total_resource_mw_zone_houston_to (Union[Unset, int]):
        total_irrmw_zone_south_from (Union[Unset, int]):
        total_irrmw_zone_south_to (Union[Unset, int]):
        total_irrmw_zone_north_from (Union[Unset, int]):
        total_irrmw_zone_north_to (Union[Unset, int]):
        total_irrmw_zone_west_from (Union[Unset, int]):
        total_irrmw_zone_west_to (Union[Unset, int]):
        total_irrmw_zone_houston_from (Union[Unset, int]):
        total_irrmw_zone_houston_to (Union[Unset, int]):
        total_new_equip_resource_mw_zone_south_from (Union[Unset, int]):
        total_new_equip_resource_mw_zone_south_to (Union[Unset, int]):
        total_new_equip_resource_mw_zone_north_from (Union[Unset, int]):
        total_new_equip_resource_mw_zone_north_to (Union[Unset, int]):
        total_new_equip_resource_mw_zone_west_from (Union[Unset, int]):
        total_new_equip_resource_mw_zone_west_to (Union[Unset, int]):
        total_new_equip_resource_mw_zone_houston_from (Union[Unset, int]):
        total_new_equip_resource_mw_zone_houston_to (Union[Unset, int]):
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
        operating_date_from=operating_date_from,
        operating_date_to=operating_date_to,
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        total_resource_mw_zone_south_from=total_resource_mw_zone_south_from,
        total_resource_mw_zone_south_to=total_resource_mw_zone_south_to,
        total_resource_mw_zone_north_from=total_resource_mw_zone_north_from,
        total_resource_mw_zone_north_to=total_resource_mw_zone_north_to,
        total_resource_mw_zone_west_from=total_resource_mw_zone_west_from,
        total_resource_mw_zone_west_to=total_resource_mw_zone_west_to,
        total_resource_mw_zone_houston_from=total_resource_mw_zone_houston_from,
        total_resource_mw_zone_houston_to=total_resource_mw_zone_houston_to,
        total_irrmw_zone_south_from=total_irrmw_zone_south_from,
        total_irrmw_zone_south_to=total_irrmw_zone_south_to,
        total_irrmw_zone_north_from=total_irrmw_zone_north_from,
        total_irrmw_zone_north_to=total_irrmw_zone_north_to,
        total_irrmw_zone_west_from=total_irrmw_zone_west_from,
        total_irrmw_zone_west_to=total_irrmw_zone_west_to,
        total_irrmw_zone_houston_from=total_irrmw_zone_houston_from,
        total_irrmw_zone_houston_to=total_irrmw_zone_houston_to,
        total_new_equip_resource_mw_zone_south_from=total_new_equip_resource_mw_zone_south_from,
        total_new_equip_resource_mw_zone_south_to=total_new_equip_resource_mw_zone_south_to,
        total_new_equip_resource_mw_zone_north_from=total_new_equip_resource_mw_zone_north_from,
        total_new_equip_resource_mw_zone_north_to=total_new_equip_resource_mw_zone_north_to,
        total_new_equip_resource_mw_zone_west_from=total_new_equip_resource_mw_zone_west_from,
        total_new_equip_resource_mw_zone_west_to=total_new_equip_resource_mw_zone_west_to,
        total_new_equip_resource_mw_zone_houston_from=total_new_equip_resource_mw_zone_houston_from,
        total_new_equip_resource_mw_zone_houston_to=total_new_equip_resource_mw_zone_houston_to,
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
    operating_date_from: Union[Unset, str] = UNSET,
    operating_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    total_resource_mw_zone_south_from: Union[Unset, int] = UNSET,
    total_resource_mw_zone_south_to: Union[Unset, int] = UNSET,
    total_resource_mw_zone_north_from: Union[Unset, int] = UNSET,
    total_resource_mw_zone_north_to: Union[Unset, int] = UNSET,
    total_resource_mw_zone_west_from: Union[Unset, int] = UNSET,
    total_resource_mw_zone_west_to: Union[Unset, int] = UNSET,
    total_resource_mw_zone_houston_from: Union[Unset, int] = UNSET,
    total_resource_mw_zone_houston_to: Union[Unset, int] = UNSET,
    total_irrmw_zone_south_from: Union[Unset, int] = UNSET,
    total_irrmw_zone_south_to: Union[Unset, int] = UNSET,
    total_irrmw_zone_north_from: Union[Unset, int] = UNSET,
    total_irrmw_zone_north_to: Union[Unset, int] = UNSET,
    total_irrmw_zone_west_from: Union[Unset, int] = UNSET,
    total_irrmw_zone_west_to: Union[Unset, int] = UNSET,
    total_irrmw_zone_houston_from: Union[Unset, int] = UNSET,
    total_irrmw_zone_houston_to: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_south_from: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_south_to: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_north_from: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_north_to: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_west_from: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_west_to: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_houston_from: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_houston_to: Union[Unset, int] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """Hourly Resource Outage Capacity

     Hourly Resource Outage Capacity

    Args:
        operating_date_from (Union[Unset, str]):
        operating_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        total_resource_mw_zone_south_from (Union[Unset, int]):
        total_resource_mw_zone_south_to (Union[Unset, int]):
        total_resource_mw_zone_north_from (Union[Unset, int]):
        total_resource_mw_zone_north_to (Union[Unset, int]):
        total_resource_mw_zone_west_from (Union[Unset, int]):
        total_resource_mw_zone_west_to (Union[Unset, int]):
        total_resource_mw_zone_houston_from (Union[Unset, int]):
        total_resource_mw_zone_houston_to (Union[Unset, int]):
        total_irrmw_zone_south_from (Union[Unset, int]):
        total_irrmw_zone_south_to (Union[Unset, int]):
        total_irrmw_zone_north_from (Union[Unset, int]):
        total_irrmw_zone_north_to (Union[Unset, int]):
        total_irrmw_zone_west_from (Union[Unset, int]):
        total_irrmw_zone_west_to (Union[Unset, int]):
        total_irrmw_zone_houston_from (Union[Unset, int]):
        total_irrmw_zone_houston_to (Union[Unset, int]):
        total_new_equip_resource_mw_zone_south_from (Union[Unset, int]):
        total_new_equip_resource_mw_zone_south_to (Union[Unset, int]):
        total_new_equip_resource_mw_zone_north_from (Union[Unset, int]):
        total_new_equip_resource_mw_zone_north_to (Union[Unset, int]):
        total_new_equip_resource_mw_zone_west_from (Union[Unset, int]):
        total_new_equip_resource_mw_zone_west_to (Union[Unset, int]):
        total_new_equip_resource_mw_zone_houston_from (Union[Unset, int]):
        total_new_equip_resource_mw_zone_houston_to (Union[Unset, int]):
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
        operating_date_from=operating_date_from,
        operating_date_to=operating_date_to,
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        total_resource_mw_zone_south_from=total_resource_mw_zone_south_from,
        total_resource_mw_zone_south_to=total_resource_mw_zone_south_to,
        total_resource_mw_zone_north_from=total_resource_mw_zone_north_from,
        total_resource_mw_zone_north_to=total_resource_mw_zone_north_to,
        total_resource_mw_zone_west_from=total_resource_mw_zone_west_from,
        total_resource_mw_zone_west_to=total_resource_mw_zone_west_to,
        total_resource_mw_zone_houston_from=total_resource_mw_zone_houston_from,
        total_resource_mw_zone_houston_to=total_resource_mw_zone_houston_to,
        total_irrmw_zone_south_from=total_irrmw_zone_south_from,
        total_irrmw_zone_south_to=total_irrmw_zone_south_to,
        total_irrmw_zone_north_from=total_irrmw_zone_north_from,
        total_irrmw_zone_north_to=total_irrmw_zone_north_to,
        total_irrmw_zone_west_from=total_irrmw_zone_west_from,
        total_irrmw_zone_west_to=total_irrmw_zone_west_to,
        total_irrmw_zone_houston_from=total_irrmw_zone_houston_from,
        total_irrmw_zone_houston_to=total_irrmw_zone_houston_to,
        total_new_equip_resource_mw_zone_south_from=total_new_equip_resource_mw_zone_south_from,
        total_new_equip_resource_mw_zone_south_to=total_new_equip_resource_mw_zone_south_to,
        total_new_equip_resource_mw_zone_north_from=total_new_equip_resource_mw_zone_north_from,
        total_new_equip_resource_mw_zone_north_to=total_new_equip_resource_mw_zone_north_to,
        total_new_equip_resource_mw_zone_west_from=total_new_equip_resource_mw_zone_west_from,
        total_new_equip_resource_mw_zone_west_to=total_new_equip_resource_mw_zone_west_to,
        total_new_equip_resource_mw_zone_houston_from=total_new_equip_resource_mw_zone_houston_from,
        total_new_equip_resource_mw_zone_houston_to=total_new_equip_resource_mw_zone_houston_to,
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
    operating_date_from: Union[Unset, str] = UNSET,
    operating_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    total_resource_mw_zone_south_from: Union[Unset, int] = UNSET,
    total_resource_mw_zone_south_to: Union[Unset, int] = UNSET,
    total_resource_mw_zone_north_from: Union[Unset, int] = UNSET,
    total_resource_mw_zone_north_to: Union[Unset, int] = UNSET,
    total_resource_mw_zone_west_from: Union[Unset, int] = UNSET,
    total_resource_mw_zone_west_to: Union[Unset, int] = UNSET,
    total_resource_mw_zone_houston_from: Union[Unset, int] = UNSET,
    total_resource_mw_zone_houston_to: Union[Unset, int] = UNSET,
    total_irrmw_zone_south_from: Union[Unset, int] = UNSET,
    total_irrmw_zone_south_to: Union[Unset, int] = UNSET,
    total_irrmw_zone_north_from: Union[Unset, int] = UNSET,
    total_irrmw_zone_north_to: Union[Unset, int] = UNSET,
    total_irrmw_zone_west_from: Union[Unset, int] = UNSET,
    total_irrmw_zone_west_to: Union[Unset, int] = UNSET,
    total_irrmw_zone_houston_from: Union[Unset, int] = UNSET,
    total_irrmw_zone_houston_to: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_south_from: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_south_to: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_north_from: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_north_to: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_west_from: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_west_to: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_houston_from: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_houston_to: Union[Unset, int] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """Hourly Resource Outage Capacity

     Hourly Resource Outage Capacity

    Args:
        operating_date_from (Union[Unset, str]):
        operating_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        total_resource_mw_zone_south_from (Union[Unset, int]):
        total_resource_mw_zone_south_to (Union[Unset, int]):
        total_resource_mw_zone_north_from (Union[Unset, int]):
        total_resource_mw_zone_north_to (Union[Unset, int]):
        total_resource_mw_zone_west_from (Union[Unset, int]):
        total_resource_mw_zone_west_to (Union[Unset, int]):
        total_resource_mw_zone_houston_from (Union[Unset, int]):
        total_resource_mw_zone_houston_to (Union[Unset, int]):
        total_irrmw_zone_south_from (Union[Unset, int]):
        total_irrmw_zone_south_to (Union[Unset, int]):
        total_irrmw_zone_north_from (Union[Unset, int]):
        total_irrmw_zone_north_to (Union[Unset, int]):
        total_irrmw_zone_west_from (Union[Unset, int]):
        total_irrmw_zone_west_to (Union[Unset, int]):
        total_irrmw_zone_houston_from (Union[Unset, int]):
        total_irrmw_zone_houston_to (Union[Unset, int]):
        total_new_equip_resource_mw_zone_south_from (Union[Unset, int]):
        total_new_equip_resource_mw_zone_south_to (Union[Unset, int]):
        total_new_equip_resource_mw_zone_north_from (Union[Unset, int]):
        total_new_equip_resource_mw_zone_north_to (Union[Unset, int]):
        total_new_equip_resource_mw_zone_west_from (Union[Unset, int]):
        total_new_equip_resource_mw_zone_west_to (Union[Unset, int]):
        total_new_equip_resource_mw_zone_houston_from (Union[Unset, int]):
        total_new_equip_resource_mw_zone_houston_to (Union[Unset, int]):
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
        operating_date_from=operating_date_from,
        operating_date_to=operating_date_to,
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        total_resource_mw_zone_south_from=total_resource_mw_zone_south_from,
        total_resource_mw_zone_south_to=total_resource_mw_zone_south_to,
        total_resource_mw_zone_north_from=total_resource_mw_zone_north_from,
        total_resource_mw_zone_north_to=total_resource_mw_zone_north_to,
        total_resource_mw_zone_west_from=total_resource_mw_zone_west_from,
        total_resource_mw_zone_west_to=total_resource_mw_zone_west_to,
        total_resource_mw_zone_houston_from=total_resource_mw_zone_houston_from,
        total_resource_mw_zone_houston_to=total_resource_mw_zone_houston_to,
        total_irrmw_zone_south_from=total_irrmw_zone_south_from,
        total_irrmw_zone_south_to=total_irrmw_zone_south_to,
        total_irrmw_zone_north_from=total_irrmw_zone_north_from,
        total_irrmw_zone_north_to=total_irrmw_zone_north_to,
        total_irrmw_zone_west_from=total_irrmw_zone_west_from,
        total_irrmw_zone_west_to=total_irrmw_zone_west_to,
        total_irrmw_zone_houston_from=total_irrmw_zone_houston_from,
        total_irrmw_zone_houston_to=total_irrmw_zone_houston_to,
        total_new_equip_resource_mw_zone_south_from=total_new_equip_resource_mw_zone_south_from,
        total_new_equip_resource_mw_zone_south_to=total_new_equip_resource_mw_zone_south_to,
        total_new_equip_resource_mw_zone_north_from=total_new_equip_resource_mw_zone_north_from,
        total_new_equip_resource_mw_zone_north_to=total_new_equip_resource_mw_zone_north_to,
        total_new_equip_resource_mw_zone_west_from=total_new_equip_resource_mw_zone_west_from,
        total_new_equip_resource_mw_zone_west_to=total_new_equip_resource_mw_zone_west_to,
        total_new_equip_resource_mw_zone_houston_from=total_new_equip_resource_mw_zone_houston_from,
        total_new_equip_resource_mw_zone_houston_to=total_new_equip_resource_mw_zone_houston_to,
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
    operating_date_from: Union[Unset, str] = UNSET,
    operating_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    total_resource_mw_zone_south_from: Union[Unset, int] = UNSET,
    total_resource_mw_zone_south_to: Union[Unset, int] = UNSET,
    total_resource_mw_zone_north_from: Union[Unset, int] = UNSET,
    total_resource_mw_zone_north_to: Union[Unset, int] = UNSET,
    total_resource_mw_zone_west_from: Union[Unset, int] = UNSET,
    total_resource_mw_zone_west_to: Union[Unset, int] = UNSET,
    total_resource_mw_zone_houston_from: Union[Unset, int] = UNSET,
    total_resource_mw_zone_houston_to: Union[Unset, int] = UNSET,
    total_irrmw_zone_south_from: Union[Unset, int] = UNSET,
    total_irrmw_zone_south_to: Union[Unset, int] = UNSET,
    total_irrmw_zone_north_from: Union[Unset, int] = UNSET,
    total_irrmw_zone_north_to: Union[Unset, int] = UNSET,
    total_irrmw_zone_west_from: Union[Unset, int] = UNSET,
    total_irrmw_zone_west_to: Union[Unset, int] = UNSET,
    total_irrmw_zone_houston_from: Union[Unset, int] = UNSET,
    total_irrmw_zone_houston_to: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_south_from: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_south_to: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_north_from: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_north_to: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_west_from: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_west_to: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_houston_from: Union[Unset, int] = UNSET,
    total_new_equip_resource_mw_zone_houston_to: Union[Unset, int] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """Hourly Resource Outage Capacity

     Hourly Resource Outage Capacity

    Args:
        operating_date_from (Union[Unset, str]):
        operating_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        total_resource_mw_zone_south_from (Union[Unset, int]):
        total_resource_mw_zone_south_to (Union[Unset, int]):
        total_resource_mw_zone_north_from (Union[Unset, int]):
        total_resource_mw_zone_north_to (Union[Unset, int]):
        total_resource_mw_zone_west_from (Union[Unset, int]):
        total_resource_mw_zone_west_to (Union[Unset, int]):
        total_resource_mw_zone_houston_from (Union[Unset, int]):
        total_resource_mw_zone_houston_to (Union[Unset, int]):
        total_irrmw_zone_south_from (Union[Unset, int]):
        total_irrmw_zone_south_to (Union[Unset, int]):
        total_irrmw_zone_north_from (Union[Unset, int]):
        total_irrmw_zone_north_to (Union[Unset, int]):
        total_irrmw_zone_west_from (Union[Unset, int]):
        total_irrmw_zone_west_to (Union[Unset, int]):
        total_irrmw_zone_houston_from (Union[Unset, int]):
        total_irrmw_zone_houston_to (Union[Unset, int]):
        total_new_equip_resource_mw_zone_south_from (Union[Unset, int]):
        total_new_equip_resource_mw_zone_south_to (Union[Unset, int]):
        total_new_equip_resource_mw_zone_north_from (Union[Unset, int]):
        total_new_equip_resource_mw_zone_north_to (Union[Unset, int]):
        total_new_equip_resource_mw_zone_west_from (Union[Unset, int]):
        total_new_equip_resource_mw_zone_west_to (Union[Unset, int]):
        total_new_equip_resource_mw_zone_houston_from (Union[Unset, int]):
        total_new_equip_resource_mw_zone_houston_to (Union[Unset, int]):
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
            operating_date_from=operating_date_from,
            operating_date_to=operating_date_to,
            hour_ending_from=hour_ending_from,
            hour_ending_to=hour_ending_to,
            total_resource_mw_zone_south_from=total_resource_mw_zone_south_from,
            total_resource_mw_zone_south_to=total_resource_mw_zone_south_to,
            total_resource_mw_zone_north_from=total_resource_mw_zone_north_from,
            total_resource_mw_zone_north_to=total_resource_mw_zone_north_to,
            total_resource_mw_zone_west_from=total_resource_mw_zone_west_from,
            total_resource_mw_zone_west_to=total_resource_mw_zone_west_to,
            total_resource_mw_zone_houston_from=total_resource_mw_zone_houston_from,
            total_resource_mw_zone_houston_to=total_resource_mw_zone_houston_to,
            total_irrmw_zone_south_from=total_irrmw_zone_south_from,
            total_irrmw_zone_south_to=total_irrmw_zone_south_to,
            total_irrmw_zone_north_from=total_irrmw_zone_north_from,
            total_irrmw_zone_north_to=total_irrmw_zone_north_to,
            total_irrmw_zone_west_from=total_irrmw_zone_west_from,
            total_irrmw_zone_west_to=total_irrmw_zone_west_to,
            total_irrmw_zone_houston_from=total_irrmw_zone_houston_from,
            total_irrmw_zone_houston_to=total_irrmw_zone_houston_to,
            total_new_equip_resource_mw_zone_south_from=total_new_equip_resource_mw_zone_south_from,
            total_new_equip_resource_mw_zone_south_to=total_new_equip_resource_mw_zone_south_to,
            total_new_equip_resource_mw_zone_north_from=total_new_equip_resource_mw_zone_north_from,
            total_new_equip_resource_mw_zone_north_to=total_new_equip_resource_mw_zone_north_to,
            total_new_equip_resource_mw_zone_west_from=total_new_equip_resource_mw_zone_west_from,
            total_new_equip_resource_mw_zone_west_to=total_new_equip_resource_mw_zone_west_to,
            total_new_equip_resource_mw_zone_houston_from=total_new_equip_resource_mw_zone_houston_from,
            total_new_equip_resource_mw_zone_houston_to=total_new_equip_resource_mw_zone_houston_to,
            posted_datetime_from=posted_datetime_from,
            posted_datetime_to=posted_datetime_to,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
