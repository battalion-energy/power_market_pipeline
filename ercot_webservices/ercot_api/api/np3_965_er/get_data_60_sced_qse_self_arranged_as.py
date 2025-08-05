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
    sced_timestamp_from: Union[Unset, str] = UNSET,
    sced_timestamp_to: Union[Unset, str] = UNSET,
    repeat_hour_flag: Union[Unset, bool] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    regup_from: Union[Unset, float] = UNSET,
    regup_to: Union[Unset, float] = UNSET,
    regdn_from: Union[Unset, float] = UNSET,
    regdn_to: Union[Unset, float] = UNSET,
    nspin_from: Union[Unset, float] = UNSET,
    nspin_to: Union[Unset, float] = UNSET,
    nspnm_from: Union[Unset, float] = UNSET,
    nspnm_to: Union[Unset, float] = UNSET,
    rrspfr_from: Union[Unset, float] = UNSET,
    rrspfr_to: Union[Unset, float] = UNSET,
    rrsffr_from: Union[Unset, float] = UNSET,
    rrsffr_to: Union[Unset, float] = UNSET,
    rrsufr_from: Union[Unset, float] = UNSET,
    rrsufr_to: Union[Unset, float] = UNSET,
    ecrss_from: Union[Unset, float] = UNSET,
    ecrss_to: Union[Unset, float] = UNSET,
    ecrsm_from: Union[Unset, float] = UNSET,
    ecrsm_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["Ocp-Apim-Subscription-Key"] = ocp_apim_subscription_key

    params: dict[str, Any] = {}

    params["SCEDTimestampFrom"] = sced_timestamp_from

    params["SCEDTimestampTo"] = sced_timestamp_to

    params["repeatHourFlag"] = repeat_hour_flag

    params["qseName"] = qse_name

    params["REGUPFrom"] = regup_from

    params["REGUPTo"] = regup_to

    params["REGDNFrom"] = regdn_from

    params["REGDNTo"] = regdn_to

    params["NSPINFrom"] = nspin_from

    params["NSPINTo"] = nspin_to

    params["NSPNMFrom"] = nspnm_from

    params["NSPNMTo"] = nspnm_to

    params["RRSPFRFrom"] = rrspfr_from

    params["RRSPFRTo"] = rrspfr_to

    params["RRSFFRFrom"] = rrsffr_from

    params["RRSFFRTo"] = rrsffr_to

    params["RRSUFRFrom"] = rrsufr_from

    params["RRSUFRTo"] = rrsufr_to

    params["ECRSSFrom"] = ecrss_from

    params["ECRSSTo"] = ecrss_to

    params["ECRSMFrom"] = ecrsm_from

    params["ECRSMTo"] = ecrsm_to

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np3-965-er/60_sced_qse_self_arranged_as",
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
    sced_timestamp_from: Union[Unset, str] = UNSET,
    sced_timestamp_to: Union[Unset, str] = UNSET,
    repeat_hour_flag: Union[Unset, bool] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    regup_from: Union[Unset, float] = UNSET,
    regup_to: Union[Unset, float] = UNSET,
    regdn_from: Union[Unset, float] = UNSET,
    regdn_to: Union[Unset, float] = UNSET,
    nspin_from: Union[Unset, float] = UNSET,
    nspin_to: Union[Unset, float] = UNSET,
    nspnm_from: Union[Unset, float] = UNSET,
    nspnm_to: Union[Unset, float] = UNSET,
    rrspfr_from: Union[Unset, float] = UNSET,
    rrspfr_to: Union[Unset, float] = UNSET,
    rrsffr_from: Union[Unset, float] = UNSET,
    rrsffr_to: Union[Unset, float] = UNSET,
    rrsufr_from: Union[Unset, float] = UNSET,
    rrsufr_to: Union[Unset, float] = UNSET,
    ecrss_from: Union[Unset, float] = UNSET,
    ecrss_to: Union[Unset, float] = UNSET,
    ecrsm_from: Union[Unset, float] = UNSET,
    ecrsm_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """60-Day QSE-specific Self-Arranged AS in SCED

     60-Day QSE-specific Self-Arranged AS in SCED

    Args:
        sced_timestamp_from (Union[Unset, str]):
        sced_timestamp_to (Union[Unset, str]):
        repeat_hour_flag (Union[Unset, bool]):
        qse_name (Union[Unset, str]):
        regup_from (Union[Unset, float]):
        regup_to (Union[Unset, float]):
        regdn_from (Union[Unset, float]):
        regdn_to (Union[Unset, float]):
        nspin_from (Union[Unset, float]):
        nspin_to (Union[Unset, float]):
        nspnm_from (Union[Unset, float]):
        nspnm_to (Union[Unset, float]):
        rrspfr_from (Union[Unset, float]):
        rrspfr_to (Union[Unset, float]):
        rrsffr_from (Union[Unset, float]):
        rrsffr_to (Union[Unset, float]):
        rrsufr_from (Union[Unset, float]):
        rrsufr_to (Union[Unset, float]):
        ecrss_from (Union[Unset, float]):
        ecrss_to (Union[Unset, float]):
        ecrsm_from (Union[Unset, float]):
        ecrsm_to (Union[Unset, float]):
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
        sced_timestamp_from=sced_timestamp_from,
        sced_timestamp_to=sced_timestamp_to,
        repeat_hour_flag=repeat_hour_flag,
        qse_name=qse_name,
        regup_from=regup_from,
        regup_to=regup_to,
        regdn_from=regdn_from,
        regdn_to=regdn_to,
        nspin_from=nspin_from,
        nspin_to=nspin_to,
        nspnm_from=nspnm_from,
        nspnm_to=nspnm_to,
        rrspfr_from=rrspfr_from,
        rrspfr_to=rrspfr_to,
        rrsffr_from=rrsffr_from,
        rrsffr_to=rrsffr_to,
        rrsufr_from=rrsufr_from,
        rrsufr_to=rrsufr_to,
        ecrss_from=ecrss_from,
        ecrss_to=ecrss_to,
        ecrsm_from=ecrsm_from,
        ecrsm_to=ecrsm_to,
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
    sced_timestamp_from: Union[Unset, str] = UNSET,
    sced_timestamp_to: Union[Unset, str] = UNSET,
    repeat_hour_flag: Union[Unset, bool] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    regup_from: Union[Unset, float] = UNSET,
    regup_to: Union[Unset, float] = UNSET,
    regdn_from: Union[Unset, float] = UNSET,
    regdn_to: Union[Unset, float] = UNSET,
    nspin_from: Union[Unset, float] = UNSET,
    nspin_to: Union[Unset, float] = UNSET,
    nspnm_from: Union[Unset, float] = UNSET,
    nspnm_to: Union[Unset, float] = UNSET,
    rrspfr_from: Union[Unset, float] = UNSET,
    rrspfr_to: Union[Unset, float] = UNSET,
    rrsffr_from: Union[Unset, float] = UNSET,
    rrsffr_to: Union[Unset, float] = UNSET,
    rrsufr_from: Union[Unset, float] = UNSET,
    rrsufr_to: Union[Unset, float] = UNSET,
    ecrss_from: Union[Unset, float] = UNSET,
    ecrss_to: Union[Unset, float] = UNSET,
    ecrsm_from: Union[Unset, float] = UNSET,
    ecrsm_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """60-Day QSE-specific Self-Arranged AS in SCED

     60-Day QSE-specific Self-Arranged AS in SCED

    Args:
        sced_timestamp_from (Union[Unset, str]):
        sced_timestamp_to (Union[Unset, str]):
        repeat_hour_flag (Union[Unset, bool]):
        qse_name (Union[Unset, str]):
        regup_from (Union[Unset, float]):
        regup_to (Union[Unset, float]):
        regdn_from (Union[Unset, float]):
        regdn_to (Union[Unset, float]):
        nspin_from (Union[Unset, float]):
        nspin_to (Union[Unset, float]):
        nspnm_from (Union[Unset, float]):
        nspnm_to (Union[Unset, float]):
        rrspfr_from (Union[Unset, float]):
        rrspfr_to (Union[Unset, float]):
        rrsffr_from (Union[Unset, float]):
        rrsffr_to (Union[Unset, float]):
        rrsufr_from (Union[Unset, float]):
        rrsufr_to (Union[Unset, float]):
        ecrss_from (Union[Unset, float]):
        ecrss_to (Union[Unset, float]):
        ecrsm_from (Union[Unset, float]):
        ecrsm_to (Union[Unset, float]):
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
        sced_timestamp_from=sced_timestamp_from,
        sced_timestamp_to=sced_timestamp_to,
        repeat_hour_flag=repeat_hour_flag,
        qse_name=qse_name,
        regup_from=regup_from,
        regup_to=regup_to,
        regdn_from=regdn_from,
        regdn_to=regdn_to,
        nspin_from=nspin_from,
        nspin_to=nspin_to,
        nspnm_from=nspnm_from,
        nspnm_to=nspnm_to,
        rrspfr_from=rrspfr_from,
        rrspfr_to=rrspfr_to,
        rrsffr_from=rrsffr_from,
        rrsffr_to=rrsffr_to,
        rrsufr_from=rrsufr_from,
        rrsufr_to=rrsufr_to,
        ecrss_from=ecrss_from,
        ecrss_to=ecrss_to,
        ecrsm_from=ecrsm_from,
        ecrsm_to=ecrsm_to,
        page=page,
        size=size,
        sort=sort,
        dir_=dir_,
        ocp_apim_subscription_key=ocp_apim_subscription_key,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    sced_timestamp_from: Union[Unset, str] = UNSET,
    sced_timestamp_to: Union[Unset, str] = UNSET,
    repeat_hour_flag: Union[Unset, bool] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    regup_from: Union[Unset, float] = UNSET,
    regup_to: Union[Unset, float] = UNSET,
    regdn_from: Union[Unset, float] = UNSET,
    regdn_to: Union[Unset, float] = UNSET,
    nspin_from: Union[Unset, float] = UNSET,
    nspin_to: Union[Unset, float] = UNSET,
    nspnm_from: Union[Unset, float] = UNSET,
    nspnm_to: Union[Unset, float] = UNSET,
    rrspfr_from: Union[Unset, float] = UNSET,
    rrspfr_to: Union[Unset, float] = UNSET,
    rrsffr_from: Union[Unset, float] = UNSET,
    rrsffr_to: Union[Unset, float] = UNSET,
    rrsufr_from: Union[Unset, float] = UNSET,
    rrsufr_to: Union[Unset, float] = UNSET,
    ecrss_from: Union[Unset, float] = UNSET,
    ecrss_to: Union[Unset, float] = UNSET,
    ecrsm_from: Union[Unset, float] = UNSET,
    ecrsm_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """60-Day QSE-specific Self-Arranged AS in SCED

     60-Day QSE-specific Self-Arranged AS in SCED

    Args:
        sced_timestamp_from (Union[Unset, str]):
        sced_timestamp_to (Union[Unset, str]):
        repeat_hour_flag (Union[Unset, bool]):
        qse_name (Union[Unset, str]):
        regup_from (Union[Unset, float]):
        regup_to (Union[Unset, float]):
        regdn_from (Union[Unset, float]):
        regdn_to (Union[Unset, float]):
        nspin_from (Union[Unset, float]):
        nspin_to (Union[Unset, float]):
        nspnm_from (Union[Unset, float]):
        nspnm_to (Union[Unset, float]):
        rrspfr_from (Union[Unset, float]):
        rrspfr_to (Union[Unset, float]):
        rrsffr_from (Union[Unset, float]):
        rrsffr_to (Union[Unset, float]):
        rrsufr_from (Union[Unset, float]):
        rrsufr_to (Union[Unset, float]):
        ecrss_from (Union[Unset, float]):
        ecrss_to (Union[Unset, float]):
        ecrsm_from (Union[Unset, float]):
        ecrsm_to (Union[Unset, float]):
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
        sced_timestamp_from=sced_timestamp_from,
        sced_timestamp_to=sced_timestamp_to,
        repeat_hour_flag=repeat_hour_flag,
        qse_name=qse_name,
        regup_from=regup_from,
        regup_to=regup_to,
        regdn_from=regdn_from,
        regdn_to=regdn_to,
        nspin_from=nspin_from,
        nspin_to=nspin_to,
        nspnm_from=nspnm_from,
        nspnm_to=nspnm_to,
        rrspfr_from=rrspfr_from,
        rrspfr_to=rrspfr_to,
        rrsffr_from=rrsffr_from,
        rrsffr_to=rrsffr_to,
        rrsufr_from=rrsufr_from,
        rrsufr_to=rrsufr_to,
        ecrss_from=ecrss_from,
        ecrss_to=ecrss_to,
        ecrsm_from=ecrsm_from,
        ecrsm_to=ecrsm_to,
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
    sced_timestamp_from: Union[Unset, str] = UNSET,
    sced_timestamp_to: Union[Unset, str] = UNSET,
    repeat_hour_flag: Union[Unset, bool] = UNSET,
    qse_name: Union[Unset, str] = UNSET,
    regup_from: Union[Unset, float] = UNSET,
    regup_to: Union[Unset, float] = UNSET,
    regdn_from: Union[Unset, float] = UNSET,
    regdn_to: Union[Unset, float] = UNSET,
    nspin_from: Union[Unset, float] = UNSET,
    nspin_to: Union[Unset, float] = UNSET,
    nspnm_from: Union[Unset, float] = UNSET,
    nspnm_to: Union[Unset, float] = UNSET,
    rrspfr_from: Union[Unset, float] = UNSET,
    rrspfr_to: Union[Unset, float] = UNSET,
    rrsffr_from: Union[Unset, float] = UNSET,
    rrsffr_to: Union[Unset, float] = UNSET,
    rrsufr_from: Union[Unset, float] = UNSET,
    rrsufr_to: Union[Unset, float] = UNSET,
    ecrss_from: Union[Unset, float] = UNSET,
    ecrss_to: Union[Unset, float] = UNSET,
    ecrsm_from: Union[Unset, float] = UNSET,
    ecrsm_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """60-Day QSE-specific Self-Arranged AS in SCED

     60-Day QSE-specific Self-Arranged AS in SCED

    Args:
        sced_timestamp_from (Union[Unset, str]):
        sced_timestamp_to (Union[Unset, str]):
        repeat_hour_flag (Union[Unset, bool]):
        qse_name (Union[Unset, str]):
        regup_from (Union[Unset, float]):
        regup_to (Union[Unset, float]):
        regdn_from (Union[Unset, float]):
        regdn_to (Union[Unset, float]):
        nspin_from (Union[Unset, float]):
        nspin_to (Union[Unset, float]):
        nspnm_from (Union[Unset, float]):
        nspnm_to (Union[Unset, float]):
        rrspfr_from (Union[Unset, float]):
        rrspfr_to (Union[Unset, float]):
        rrsffr_from (Union[Unset, float]):
        rrsffr_to (Union[Unset, float]):
        rrsufr_from (Union[Unset, float]):
        rrsufr_to (Union[Unset, float]):
        ecrss_from (Union[Unset, float]):
        ecrss_to (Union[Unset, float]):
        ecrsm_from (Union[Unset, float]):
        ecrsm_to (Union[Unset, float]):
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
            sced_timestamp_from=sced_timestamp_from,
            sced_timestamp_to=sced_timestamp_to,
            repeat_hour_flag=repeat_hour_flag,
            qse_name=qse_name,
            regup_from=regup_from,
            regup_to=regup_to,
            regdn_from=regdn_from,
            regdn_to=regdn_to,
            nspin_from=nspin_from,
            nspin_to=nspin_to,
            nspnm_from=nspnm_from,
            nspnm_to=nspnm_to,
            rrspfr_from=rrspfr_from,
            rrspfr_to=rrspfr_to,
            rrsffr_from=rrsffr_from,
            rrsffr_to=rrsffr_to,
            rrsufr_from=rrsufr_from,
            rrsufr_to=rrsufr_to,
            ecrss_from=ecrss_from,
            ecrss_to=ecrss_to,
            ecrsm_from=ecrsm_from,
            ecrsm_to=ecrsm_to,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
