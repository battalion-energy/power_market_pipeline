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
    total_self_arranged_asrrsufr_from: Union[Unset, float] = UNSET,
    total_self_arranged_asrrsufr_to: Union[Unset, float] = UNSET,
    total_self_arranged_asecrssd_from: Union[Unset, float] = UNSET,
    total_self_arranged_asecrssd_to: Union[Unset, float] = UNSET,
    total_self_arranged_asecrsmd_from: Union[Unset, float] = UNSET,
    total_self_arranged_asecrsmd_to: Union[Unset, float] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    total_self_arranged_asregup_from: Union[Unset, float] = UNSET,
    total_self_arranged_asregup_to: Union[Unset, float] = UNSET,
    total_self_arranged_asregdn_from: Union[Unset, float] = UNSET,
    total_self_arranged_asregdn_to: Union[Unset, float] = UNSET,
    total_self_arranged_asnspin_from: Union[Unset, float] = UNSET,
    total_self_arranged_asnspin_to: Union[Unset, float] = UNSET,
    total_self_arranged_asnspnm_from: Union[Unset, float] = UNSET,
    total_self_arranged_asnspnm_to: Union[Unset, float] = UNSET,
    total_self_arranged_asrrspfr_from: Union[Unset, float] = UNSET,
    total_self_arranged_asrrspfr_to: Union[Unset, float] = UNSET,
    total_self_arranged_asrrsffr_from: Union[Unset, float] = UNSET,
    total_self_arranged_asrrsffr_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["Ocp-Apim-Subscription-Key"] = ocp_apim_subscription_key

    params: dict[str, Any] = {}

    params["totalSelfArrangedASRRSUFRFrom"] = total_self_arranged_asrrsufr_from

    params["totalSelfArrangedASRRSUFRTo"] = total_self_arranged_asrrsufr_to

    params["totalSelfArrangedASECRSSDFrom"] = total_self_arranged_asecrssd_from

    params["totalSelfArrangedASECRSSDTo"] = total_self_arranged_asecrssd_to

    params["totalSelfArrangedASECRSMDFrom"] = total_self_arranged_asecrsmd_from

    params["totalSelfArrangedASECRSMDTo"] = total_self_arranged_asecrsmd_to

    params["deliveryDateFrom"] = delivery_date_from

    params["deliveryDateTo"] = delivery_date_to

    params["hourEndingFrom"] = hour_ending_from

    params["hourEndingTo"] = hour_ending_to

    params["qseName"] = qse_name

    params["totalSelfArrangedASREGUPFrom"] = total_self_arranged_asregup_from

    params["totalSelfArrangedASREGUPTo"] = total_self_arranged_asregup_to

    params["totalSelfArrangedASREGDNFrom"] = total_self_arranged_asregdn_from

    params["totalSelfArrangedASREGDNTo"] = total_self_arranged_asregdn_to

    params["totalSelfArrangedASNSPINFrom"] = total_self_arranged_asnspin_from

    params["totalSelfArrangedASNSPINTo"] = total_self_arranged_asnspin_to

    params["totalSelfArrangedASNSPNMFrom"] = total_self_arranged_asnspnm_from

    params["totalSelfArrangedASNSPNMTo"] = total_self_arranged_asnspnm_to

    params["totalSelfArrangedASRRSPFRFrom"] = total_self_arranged_asrrspfr_from

    params["totalSelfArrangedASRRSPFRTo"] = total_self_arranged_asrrspfr_to

    params["totalSelfArrangedASRRSFFRFrom"] = total_self_arranged_asrrsffr_from

    params["totalSelfArrangedASRRSFFRTo"] = total_self_arranged_asrrsffr_to

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np3-966-er/60_dam_qse_self_as",
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
    total_self_arranged_asrrsufr_from: Union[Unset, float] = UNSET,
    total_self_arranged_asrrsufr_to: Union[Unset, float] = UNSET,
    total_self_arranged_asecrssd_from: Union[Unset, float] = UNSET,
    total_self_arranged_asecrssd_to: Union[Unset, float] = UNSET,
    total_self_arranged_asecrsmd_from: Union[Unset, float] = UNSET,
    total_self_arranged_asecrsmd_to: Union[Unset, float] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    total_self_arranged_asregup_from: Union[Unset, float] = UNSET,
    total_self_arranged_asregup_to: Union[Unset, float] = UNSET,
    total_self_arranged_asregdn_from: Union[Unset, float] = UNSET,
    total_self_arranged_asregdn_to: Union[Unset, float] = UNSET,
    total_self_arranged_asnspin_from: Union[Unset, float] = UNSET,
    total_self_arranged_asnspin_to: Union[Unset, float] = UNSET,
    total_self_arranged_asnspnm_from: Union[Unset, float] = UNSET,
    total_self_arranged_asnspnm_to: Union[Unset, float] = UNSET,
    total_self_arranged_asrrspfr_from: Union[Unset, float] = UNSET,
    total_self_arranged_asrrspfr_to: Union[Unset, float] = UNSET,
    total_self_arranged_asrrsffr_from: Union[Unset, float] = UNSET,
    total_self_arranged_asrrsffr_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """60-Day DAM QSE Self Arranged AS

     60-Day DAM QSE Self Arranged AS

    Args:
        total_self_arranged_asrrsufr_from (Union[Unset, float]):
        total_self_arranged_asrrsufr_to (Union[Unset, float]):
        total_self_arranged_asecrssd_from (Union[Unset, float]):
        total_self_arranged_asecrssd_to (Union[Unset, float]):
        total_self_arranged_asecrsmd_from (Union[Unset, float]):
        total_self_arranged_asecrsmd_to (Union[Unset, float]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        qse_name (Union[Unset, str]):
        total_self_arranged_asregup_from (Union[Unset, float]):
        total_self_arranged_asregup_to (Union[Unset, float]):
        total_self_arranged_asregdn_from (Union[Unset, float]):
        total_self_arranged_asregdn_to (Union[Unset, float]):
        total_self_arranged_asnspin_from (Union[Unset, float]):
        total_self_arranged_asnspin_to (Union[Unset, float]):
        total_self_arranged_asnspnm_from (Union[Unset, float]):
        total_self_arranged_asnspnm_to (Union[Unset, float]):
        total_self_arranged_asrrspfr_from (Union[Unset, float]):
        total_self_arranged_asrrspfr_to (Union[Unset, float]):
        total_self_arranged_asrrsffr_from (Union[Unset, float]):
        total_self_arranged_asrrsffr_to (Union[Unset, float]):
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
        total_self_arranged_asrrsufr_from=total_self_arranged_asrrsufr_from,
        total_self_arranged_asrrsufr_to=total_self_arranged_asrrsufr_to,
        total_self_arranged_asecrssd_from=total_self_arranged_asecrssd_from,
        total_self_arranged_asecrssd_to=total_self_arranged_asecrssd_to,
        total_self_arranged_asecrsmd_from=total_self_arranged_asecrsmd_from,
        total_self_arranged_asecrsmd_to=total_self_arranged_asecrsmd_to,
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        qse_name=qse_name,
        total_self_arranged_asregup_from=total_self_arranged_asregup_from,
        total_self_arranged_asregup_to=total_self_arranged_asregup_to,
        total_self_arranged_asregdn_from=total_self_arranged_asregdn_from,
        total_self_arranged_asregdn_to=total_self_arranged_asregdn_to,
        total_self_arranged_asnspin_from=total_self_arranged_asnspin_from,
        total_self_arranged_asnspin_to=total_self_arranged_asnspin_to,
        total_self_arranged_asnspnm_from=total_self_arranged_asnspnm_from,
        total_self_arranged_asnspnm_to=total_self_arranged_asnspnm_to,
        total_self_arranged_asrrspfr_from=total_self_arranged_asrrspfr_from,
        total_self_arranged_asrrspfr_to=total_self_arranged_asrrspfr_to,
        total_self_arranged_asrrsffr_from=total_self_arranged_asrrsffr_from,
        total_self_arranged_asrrsffr_to=total_self_arranged_asrrsffr_to,
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
    total_self_arranged_asrrsufr_from: Union[Unset, float] = UNSET,
    total_self_arranged_asrrsufr_to: Union[Unset, float] = UNSET,
    total_self_arranged_asecrssd_from: Union[Unset, float] = UNSET,
    total_self_arranged_asecrssd_to: Union[Unset, float] = UNSET,
    total_self_arranged_asecrsmd_from: Union[Unset, float] = UNSET,
    total_self_arranged_asecrsmd_to: Union[Unset, float] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    total_self_arranged_asregup_from: Union[Unset, float] = UNSET,
    total_self_arranged_asregup_to: Union[Unset, float] = UNSET,
    total_self_arranged_asregdn_from: Union[Unset, float] = UNSET,
    total_self_arranged_asregdn_to: Union[Unset, float] = UNSET,
    total_self_arranged_asnspin_from: Union[Unset, float] = UNSET,
    total_self_arranged_asnspin_to: Union[Unset, float] = UNSET,
    total_self_arranged_asnspnm_from: Union[Unset, float] = UNSET,
    total_self_arranged_asnspnm_to: Union[Unset, float] = UNSET,
    total_self_arranged_asrrspfr_from: Union[Unset, float] = UNSET,
    total_self_arranged_asrrspfr_to: Union[Unset, float] = UNSET,
    total_self_arranged_asrrsffr_from: Union[Unset, float] = UNSET,
    total_self_arranged_asrrsffr_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """60-Day DAM QSE Self Arranged AS

     60-Day DAM QSE Self Arranged AS

    Args:
        total_self_arranged_asrrsufr_from (Union[Unset, float]):
        total_self_arranged_asrrsufr_to (Union[Unset, float]):
        total_self_arranged_asecrssd_from (Union[Unset, float]):
        total_self_arranged_asecrssd_to (Union[Unset, float]):
        total_self_arranged_asecrsmd_from (Union[Unset, float]):
        total_self_arranged_asecrsmd_to (Union[Unset, float]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        qse_name (Union[Unset, str]):
        total_self_arranged_asregup_from (Union[Unset, float]):
        total_self_arranged_asregup_to (Union[Unset, float]):
        total_self_arranged_asregdn_from (Union[Unset, float]):
        total_self_arranged_asregdn_to (Union[Unset, float]):
        total_self_arranged_asnspin_from (Union[Unset, float]):
        total_self_arranged_asnspin_to (Union[Unset, float]):
        total_self_arranged_asnspnm_from (Union[Unset, float]):
        total_self_arranged_asnspnm_to (Union[Unset, float]):
        total_self_arranged_asrrspfr_from (Union[Unset, float]):
        total_self_arranged_asrrspfr_to (Union[Unset, float]):
        total_self_arranged_asrrsffr_from (Union[Unset, float]):
        total_self_arranged_asrrsffr_to (Union[Unset, float]):
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
        total_self_arranged_asrrsufr_from=total_self_arranged_asrrsufr_from,
        total_self_arranged_asrrsufr_to=total_self_arranged_asrrsufr_to,
        total_self_arranged_asecrssd_from=total_self_arranged_asecrssd_from,
        total_self_arranged_asecrssd_to=total_self_arranged_asecrssd_to,
        total_self_arranged_asecrsmd_from=total_self_arranged_asecrsmd_from,
        total_self_arranged_asecrsmd_to=total_self_arranged_asecrsmd_to,
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        qse_name=qse_name,
        total_self_arranged_asregup_from=total_self_arranged_asregup_from,
        total_self_arranged_asregup_to=total_self_arranged_asregup_to,
        total_self_arranged_asregdn_from=total_self_arranged_asregdn_from,
        total_self_arranged_asregdn_to=total_self_arranged_asregdn_to,
        total_self_arranged_asnspin_from=total_self_arranged_asnspin_from,
        total_self_arranged_asnspin_to=total_self_arranged_asnspin_to,
        total_self_arranged_asnspnm_from=total_self_arranged_asnspnm_from,
        total_self_arranged_asnspnm_to=total_self_arranged_asnspnm_to,
        total_self_arranged_asrrspfr_from=total_self_arranged_asrrspfr_from,
        total_self_arranged_asrrspfr_to=total_self_arranged_asrrspfr_to,
        total_self_arranged_asrrsffr_from=total_self_arranged_asrrsffr_from,
        total_self_arranged_asrrsffr_to=total_self_arranged_asrrsffr_to,
        page=page,
        size=size,
        sort=sort,
        dir_=dir_,
        ocp_apim_subscription_key=ocp_apim_subscription_key,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    total_self_arranged_asrrsufr_from: Union[Unset, float] = UNSET,
    total_self_arranged_asrrsufr_to: Union[Unset, float] = UNSET,
    total_self_arranged_asecrssd_from: Union[Unset, float] = UNSET,
    total_self_arranged_asecrssd_to: Union[Unset, float] = UNSET,
    total_self_arranged_asecrsmd_from: Union[Unset, float] = UNSET,
    total_self_arranged_asecrsmd_to: Union[Unset, float] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    total_self_arranged_asregup_from: Union[Unset, float] = UNSET,
    total_self_arranged_asregup_to: Union[Unset, float] = UNSET,
    total_self_arranged_asregdn_from: Union[Unset, float] = UNSET,
    total_self_arranged_asregdn_to: Union[Unset, float] = UNSET,
    total_self_arranged_asnspin_from: Union[Unset, float] = UNSET,
    total_self_arranged_asnspin_to: Union[Unset, float] = UNSET,
    total_self_arranged_asnspnm_from: Union[Unset, float] = UNSET,
    total_self_arranged_asnspnm_to: Union[Unset, float] = UNSET,
    total_self_arranged_asrrspfr_from: Union[Unset, float] = UNSET,
    total_self_arranged_asrrspfr_to: Union[Unset, float] = UNSET,
    total_self_arranged_asrrsffr_from: Union[Unset, float] = UNSET,
    total_self_arranged_asrrsffr_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """60-Day DAM QSE Self Arranged AS

     60-Day DAM QSE Self Arranged AS

    Args:
        total_self_arranged_asrrsufr_from (Union[Unset, float]):
        total_self_arranged_asrrsufr_to (Union[Unset, float]):
        total_self_arranged_asecrssd_from (Union[Unset, float]):
        total_self_arranged_asecrssd_to (Union[Unset, float]):
        total_self_arranged_asecrsmd_from (Union[Unset, float]):
        total_self_arranged_asecrsmd_to (Union[Unset, float]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        qse_name (Union[Unset, str]):
        total_self_arranged_asregup_from (Union[Unset, float]):
        total_self_arranged_asregup_to (Union[Unset, float]):
        total_self_arranged_asregdn_from (Union[Unset, float]):
        total_self_arranged_asregdn_to (Union[Unset, float]):
        total_self_arranged_asnspin_from (Union[Unset, float]):
        total_self_arranged_asnspin_to (Union[Unset, float]):
        total_self_arranged_asnspnm_from (Union[Unset, float]):
        total_self_arranged_asnspnm_to (Union[Unset, float]):
        total_self_arranged_asrrspfr_from (Union[Unset, float]):
        total_self_arranged_asrrspfr_to (Union[Unset, float]):
        total_self_arranged_asrrsffr_from (Union[Unset, float]):
        total_self_arranged_asrrsffr_to (Union[Unset, float]):
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
        total_self_arranged_asrrsufr_from=total_self_arranged_asrrsufr_from,
        total_self_arranged_asrrsufr_to=total_self_arranged_asrrsufr_to,
        total_self_arranged_asecrssd_from=total_self_arranged_asecrssd_from,
        total_self_arranged_asecrssd_to=total_self_arranged_asecrssd_to,
        total_self_arranged_asecrsmd_from=total_self_arranged_asecrsmd_from,
        total_self_arranged_asecrsmd_to=total_self_arranged_asecrsmd_to,
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        qse_name=qse_name,
        total_self_arranged_asregup_from=total_self_arranged_asregup_from,
        total_self_arranged_asregup_to=total_self_arranged_asregup_to,
        total_self_arranged_asregdn_from=total_self_arranged_asregdn_from,
        total_self_arranged_asregdn_to=total_self_arranged_asregdn_to,
        total_self_arranged_asnspin_from=total_self_arranged_asnspin_from,
        total_self_arranged_asnspin_to=total_self_arranged_asnspin_to,
        total_self_arranged_asnspnm_from=total_self_arranged_asnspnm_from,
        total_self_arranged_asnspnm_to=total_self_arranged_asnspnm_to,
        total_self_arranged_asrrspfr_from=total_self_arranged_asrrspfr_from,
        total_self_arranged_asrrspfr_to=total_self_arranged_asrrspfr_to,
        total_self_arranged_asrrsffr_from=total_self_arranged_asrrsffr_from,
        total_self_arranged_asrrsffr_to=total_self_arranged_asrrsffr_to,
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
    total_self_arranged_asrrsufr_from: Union[Unset, float] = UNSET,
    total_self_arranged_asrrsufr_to: Union[Unset, float] = UNSET,
    total_self_arranged_asecrssd_from: Union[Unset, float] = UNSET,
    total_self_arranged_asecrssd_to: Union[Unset, float] = UNSET,
    total_self_arranged_asecrsmd_from: Union[Unset, float] = UNSET,
    total_self_arranged_asecrsmd_to: Union[Unset, float] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    total_self_arranged_asregup_from: Union[Unset, float] = UNSET,
    total_self_arranged_asregup_to: Union[Unset, float] = UNSET,
    total_self_arranged_asregdn_from: Union[Unset, float] = UNSET,
    total_self_arranged_asregdn_to: Union[Unset, float] = UNSET,
    total_self_arranged_asnspin_from: Union[Unset, float] = UNSET,
    total_self_arranged_asnspin_to: Union[Unset, float] = UNSET,
    total_self_arranged_asnspnm_from: Union[Unset, float] = UNSET,
    total_self_arranged_asnspnm_to: Union[Unset, float] = UNSET,
    total_self_arranged_asrrspfr_from: Union[Unset, float] = UNSET,
    total_self_arranged_asrrspfr_to: Union[Unset, float] = UNSET,
    total_self_arranged_asrrsffr_from: Union[Unset, float] = UNSET,
    total_self_arranged_asrrsffr_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """60-Day DAM QSE Self Arranged AS

     60-Day DAM QSE Self Arranged AS

    Args:
        total_self_arranged_asrrsufr_from (Union[Unset, float]):
        total_self_arranged_asrrsufr_to (Union[Unset, float]):
        total_self_arranged_asecrssd_from (Union[Unset, float]):
        total_self_arranged_asecrssd_to (Union[Unset, float]):
        total_self_arranged_asecrsmd_from (Union[Unset, float]):
        total_self_arranged_asecrsmd_to (Union[Unset, float]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        qse_name (Union[Unset, str]):
        total_self_arranged_asregup_from (Union[Unset, float]):
        total_self_arranged_asregup_to (Union[Unset, float]):
        total_self_arranged_asregdn_from (Union[Unset, float]):
        total_self_arranged_asregdn_to (Union[Unset, float]):
        total_self_arranged_asnspin_from (Union[Unset, float]):
        total_self_arranged_asnspin_to (Union[Unset, float]):
        total_self_arranged_asnspnm_from (Union[Unset, float]):
        total_self_arranged_asnspnm_to (Union[Unset, float]):
        total_self_arranged_asrrspfr_from (Union[Unset, float]):
        total_self_arranged_asrrspfr_to (Union[Unset, float]):
        total_self_arranged_asrrsffr_from (Union[Unset, float]):
        total_self_arranged_asrrsffr_to (Union[Unset, float]):
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
            total_self_arranged_asrrsufr_from=total_self_arranged_asrrsufr_from,
            total_self_arranged_asrrsufr_to=total_self_arranged_asrrsufr_to,
            total_self_arranged_asecrssd_from=total_self_arranged_asecrssd_from,
            total_self_arranged_asecrssd_to=total_self_arranged_asecrssd_to,
            total_self_arranged_asecrsmd_from=total_self_arranged_asecrsmd_from,
            total_self_arranged_asecrsmd_to=total_self_arranged_asecrsmd_to,
            delivery_date_from=delivery_date_from,
            delivery_date_to=delivery_date_to,
            hour_ending_from=hour_ending_from,
            hour_ending_to=hour_ending_to,
            qse_name=qse_name,
            total_self_arranged_asregup_from=total_self_arranged_asregup_from,
            total_self_arranged_asregup_to=total_self_arranged_asregup_to,
            total_self_arranged_asregdn_from=total_self_arranged_asregdn_from,
            total_self_arranged_asregdn_to=total_self_arranged_asregdn_to,
            total_self_arranged_asnspin_from=total_self_arranged_asnspin_from,
            total_self_arranged_asnspin_to=total_self_arranged_asnspin_to,
            total_self_arranged_asnspnm_from=total_self_arranged_asnspnm_from,
            total_self_arranged_asnspnm_to=total_self_arranged_asnspnm_to,
            total_self_arranged_asrrspfr_from=total_self_arranged_asrrspfr_from,
            total_self_arranged_asrrspfr_to=total_self_arranged_asrrspfr_to,
            total_self_arranged_asrrsffr_from=total_self_arranged_asrrsffr_from,
            total_self_arranged_asrrsffr_to=total_self_arranged_asrrsffr_to,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
