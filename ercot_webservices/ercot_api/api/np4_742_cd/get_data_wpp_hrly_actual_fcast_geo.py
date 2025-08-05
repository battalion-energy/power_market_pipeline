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
    gen_system_wide_from: Union[Unset, float] = UNSET,
    gen_system_wide_to: Union[Unset, float] = UNSET,
    cophsl_system_wide_from: Union[Unset, float] = UNSET,
    cophsl_system_wide_to: Union[Unset, float] = UNSET,
    stwpf_system_wide_from: Union[Unset, float] = UNSET,
    stwpf_system_wide_to: Union[Unset, float] = UNSET,
    wgrpp_system_wide_from: Union[Unset, float] = UNSET,
    wgrpp_system_wide_to: Union[Unset, float] = UNSET,
    gen_panhandle_from: Union[Unset, float] = UNSET,
    gen_panhandle_to: Union[Unset, float] = UNSET,
    cophsl_panhandle_from: Union[Unset, float] = UNSET,
    cophsl_panhandle_to: Union[Unset, float] = UNSET,
    stwpf_panhandle_from: Union[Unset, float] = UNSET,
    stwpf_panhandle_to: Union[Unset, float] = UNSET,
    wgrpp_panhandle_from: Union[Unset, float] = UNSET,
    wgrpp_panhandle_to: Union[Unset, float] = UNSET,
    gen_coastal_from: Union[Unset, float] = UNSET,
    gen_coastal_to: Union[Unset, float] = UNSET,
    cophsl_coastal_from: Union[Unset, float] = UNSET,
    cophsl_coastal_to: Union[Unset, float] = UNSET,
    stwpf_coastal_from: Union[Unset, float] = UNSET,
    stwpf_coastal_to: Union[Unset, float] = UNSET,
    wgrpp_coastal_from: Union[Unset, float] = UNSET,
    wgrpp_coastal_to: Union[Unset, float] = UNSET,
    gen_south_from: Union[Unset, float] = UNSET,
    gen_south_to: Union[Unset, float] = UNSET,
    cophsl_south_from: Union[Unset, float] = UNSET,
    cophsl_south_to: Union[Unset, float] = UNSET,
    stwpf_south_from: Union[Unset, float] = UNSET,
    stwpf_south_to: Union[Unset, float] = UNSET,
    wgrpp_south_from: Union[Unset, float] = UNSET,
    wgrpp_south_to: Union[Unset, float] = UNSET,
    gen_west_from: Union[Unset, float] = UNSET,
    gen_west_to: Union[Unset, float] = UNSET,
    cophsl_west_from: Union[Unset, float] = UNSET,
    cophsl_west_to: Union[Unset, float] = UNSET,
    stwpf_west_from: Union[Unset, float] = UNSET,
    stwpf_west_to: Union[Unset, float] = UNSET,
    hsl_system_wide_from: Union[Unset, float] = UNSET,
    hsl_system_wide_to: Union[Unset, float] = UNSET,
    wgrpp_west_from: Union[Unset, float] = UNSET,
    wgrpp_west_to: Union[Unset, float] = UNSET,
    gen_north_from: Union[Unset, float] = UNSET,
    gen_north_to: Union[Unset, float] = UNSET,
    cophsl_north_from: Union[Unset, float] = UNSET,
    cophsl_north_to: Union[Unset, float] = UNSET,
    stwpf_north_from: Union[Unset, float] = UNSET,
    stwpf_north_to: Union[Unset, float] = UNSET,
    wgrpp_north_from: Union[Unset, float] = UNSET,
    wgrpp_north_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
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

    params["deliveryDateFrom"] = delivery_date_from

    params["deliveryDateTo"] = delivery_date_to

    params["hourEndingFrom"] = hour_ending_from

    params["hourEndingTo"] = hour_ending_to

    params["genSystemWideFrom"] = gen_system_wide_from

    params["genSystemWideTo"] = gen_system_wide_to

    params["COPHSLSystemWideFrom"] = cophsl_system_wide_from

    params["COPHSLSystemWideTo"] = cophsl_system_wide_to

    params["STWPFSystemWideFrom"] = stwpf_system_wide_from

    params["STWPFSystemWideTo"] = stwpf_system_wide_to

    params["WGRPPSystemWideFrom"] = wgrpp_system_wide_from

    params["WGRPPSystemWideTo"] = wgrpp_system_wide_to

    params["genPanhandleFrom"] = gen_panhandle_from

    params["genPanhandleTo"] = gen_panhandle_to

    params["COPHSLPanhandleFrom"] = cophsl_panhandle_from

    params["COPHSLPanhandleTo"] = cophsl_panhandle_to

    params["STWPFPanhandleFrom"] = stwpf_panhandle_from

    params["STWPFPanhandleTo"] = stwpf_panhandle_to

    params["WGRPPPanhandleFrom"] = wgrpp_panhandle_from

    params["WGRPPPanhandleTo"] = wgrpp_panhandle_to

    params["genCoastalFrom"] = gen_coastal_from

    params["genCoastalTo"] = gen_coastal_to

    params["COPHSLCoastalFrom"] = cophsl_coastal_from

    params["COPHSLCoastalTo"] = cophsl_coastal_to

    params["STWPFCoastalFrom"] = stwpf_coastal_from

    params["STWPFCoastalTo"] = stwpf_coastal_to

    params["WGRPPCoastalFrom"] = wgrpp_coastal_from

    params["WGRPPCoastalTo"] = wgrpp_coastal_to

    params["genSouthFrom"] = gen_south_from

    params["genSouthTo"] = gen_south_to

    params["COPHSLSouthFrom"] = cophsl_south_from

    params["COPHSLSouthTo"] = cophsl_south_to

    params["STWPFSouthFrom"] = stwpf_south_from

    params["STWPFSouthTo"] = stwpf_south_to

    params["WGRPPSouthFrom"] = wgrpp_south_from

    params["WGRPPSouthTo"] = wgrpp_south_to

    params["genWestFrom"] = gen_west_from

    params["genWestTo"] = gen_west_to

    params["COPHSLWestFrom"] = cophsl_west_from

    params["COPHSLWestTo"] = cophsl_west_to

    params["STWPFWestFrom"] = stwpf_west_from

    params["STWPFWestTo"] = stwpf_west_to

    params["HSLSystemWideFrom"] = hsl_system_wide_from

    params["HSLSystemWideTo"] = hsl_system_wide_to

    params["WGRPPWestFrom"] = wgrpp_west_from

    params["WGRPPWestTo"] = wgrpp_west_to

    params["genNorthFrom"] = gen_north_from

    params["genNorthTo"] = gen_north_to

    params["COPHSLNorthFrom"] = cophsl_north_from

    params["COPHSLNorthTo"] = cophsl_north_to

    params["STWPFNorthFrom"] = stwpf_north_from

    params["STWPFNorthTo"] = stwpf_north_to

    params["WGRPPNorthFrom"] = wgrpp_north_from

    params["WGRPPNorthTo"] = wgrpp_north_to

    params["DSTFlag"] = dst_flag

    params["postedDatetimeFrom"] = posted_datetime_from

    params["postedDatetimeTo"] = posted_datetime_to

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np4-742-cd/wpp_hrly_actual_fcast_geo",
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
    gen_system_wide_from: Union[Unset, float] = UNSET,
    gen_system_wide_to: Union[Unset, float] = UNSET,
    cophsl_system_wide_from: Union[Unset, float] = UNSET,
    cophsl_system_wide_to: Union[Unset, float] = UNSET,
    stwpf_system_wide_from: Union[Unset, float] = UNSET,
    stwpf_system_wide_to: Union[Unset, float] = UNSET,
    wgrpp_system_wide_from: Union[Unset, float] = UNSET,
    wgrpp_system_wide_to: Union[Unset, float] = UNSET,
    gen_panhandle_from: Union[Unset, float] = UNSET,
    gen_panhandle_to: Union[Unset, float] = UNSET,
    cophsl_panhandle_from: Union[Unset, float] = UNSET,
    cophsl_panhandle_to: Union[Unset, float] = UNSET,
    stwpf_panhandle_from: Union[Unset, float] = UNSET,
    stwpf_panhandle_to: Union[Unset, float] = UNSET,
    wgrpp_panhandle_from: Union[Unset, float] = UNSET,
    wgrpp_panhandle_to: Union[Unset, float] = UNSET,
    gen_coastal_from: Union[Unset, float] = UNSET,
    gen_coastal_to: Union[Unset, float] = UNSET,
    cophsl_coastal_from: Union[Unset, float] = UNSET,
    cophsl_coastal_to: Union[Unset, float] = UNSET,
    stwpf_coastal_from: Union[Unset, float] = UNSET,
    stwpf_coastal_to: Union[Unset, float] = UNSET,
    wgrpp_coastal_from: Union[Unset, float] = UNSET,
    wgrpp_coastal_to: Union[Unset, float] = UNSET,
    gen_south_from: Union[Unset, float] = UNSET,
    gen_south_to: Union[Unset, float] = UNSET,
    cophsl_south_from: Union[Unset, float] = UNSET,
    cophsl_south_to: Union[Unset, float] = UNSET,
    stwpf_south_from: Union[Unset, float] = UNSET,
    stwpf_south_to: Union[Unset, float] = UNSET,
    wgrpp_south_from: Union[Unset, float] = UNSET,
    wgrpp_south_to: Union[Unset, float] = UNSET,
    gen_west_from: Union[Unset, float] = UNSET,
    gen_west_to: Union[Unset, float] = UNSET,
    cophsl_west_from: Union[Unset, float] = UNSET,
    cophsl_west_to: Union[Unset, float] = UNSET,
    stwpf_west_from: Union[Unset, float] = UNSET,
    stwpf_west_to: Union[Unset, float] = UNSET,
    hsl_system_wide_from: Union[Unset, float] = UNSET,
    hsl_system_wide_to: Union[Unset, float] = UNSET,
    wgrpp_west_from: Union[Unset, float] = UNSET,
    wgrpp_west_to: Union[Unset, float] = UNSET,
    gen_north_from: Union[Unset, float] = UNSET,
    gen_north_to: Union[Unset, float] = UNSET,
    cophsl_north_from: Union[Unset, float] = UNSET,
    cophsl_north_to: Union[Unset, float] = UNSET,
    stwpf_north_from: Union[Unset, float] = UNSET,
    stwpf_north_to: Union[Unset, float] = UNSET,
    wgrpp_north_from: Union[Unset, float] = UNSET,
    wgrpp_north_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """Wind Power Production - Hourly Averaged Actual and Forecasted Values by Geographical Region

     Wind Power Production - Hourly Averaged Actual and Forecasted Values by Geographical Region

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        gen_system_wide_from (Union[Unset, float]):
        gen_system_wide_to (Union[Unset, float]):
        cophsl_system_wide_from (Union[Unset, float]):
        cophsl_system_wide_to (Union[Unset, float]):
        stwpf_system_wide_from (Union[Unset, float]):
        stwpf_system_wide_to (Union[Unset, float]):
        wgrpp_system_wide_from (Union[Unset, float]):
        wgrpp_system_wide_to (Union[Unset, float]):
        gen_panhandle_from (Union[Unset, float]):
        gen_panhandle_to (Union[Unset, float]):
        cophsl_panhandle_from (Union[Unset, float]):
        cophsl_panhandle_to (Union[Unset, float]):
        stwpf_panhandle_from (Union[Unset, float]):
        stwpf_panhandle_to (Union[Unset, float]):
        wgrpp_panhandle_from (Union[Unset, float]):
        wgrpp_panhandle_to (Union[Unset, float]):
        gen_coastal_from (Union[Unset, float]):
        gen_coastal_to (Union[Unset, float]):
        cophsl_coastal_from (Union[Unset, float]):
        cophsl_coastal_to (Union[Unset, float]):
        stwpf_coastal_from (Union[Unset, float]):
        stwpf_coastal_to (Union[Unset, float]):
        wgrpp_coastal_from (Union[Unset, float]):
        wgrpp_coastal_to (Union[Unset, float]):
        gen_south_from (Union[Unset, float]):
        gen_south_to (Union[Unset, float]):
        cophsl_south_from (Union[Unset, float]):
        cophsl_south_to (Union[Unset, float]):
        stwpf_south_from (Union[Unset, float]):
        stwpf_south_to (Union[Unset, float]):
        wgrpp_south_from (Union[Unset, float]):
        wgrpp_south_to (Union[Unset, float]):
        gen_west_from (Union[Unset, float]):
        gen_west_to (Union[Unset, float]):
        cophsl_west_from (Union[Unset, float]):
        cophsl_west_to (Union[Unset, float]):
        stwpf_west_from (Union[Unset, float]):
        stwpf_west_to (Union[Unset, float]):
        hsl_system_wide_from (Union[Unset, float]):
        hsl_system_wide_to (Union[Unset, float]):
        wgrpp_west_from (Union[Unset, float]):
        wgrpp_west_to (Union[Unset, float]):
        gen_north_from (Union[Unset, float]):
        gen_north_to (Union[Unset, float]):
        cophsl_north_from (Union[Unset, float]):
        cophsl_north_to (Union[Unset, float]):
        stwpf_north_from (Union[Unset, float]):
        stwpf_north_to (Union[Unset, float]):
        wgrpp_north_from (Union[Unset, float]):
        wgrpp_north_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
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
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        gen_system_wide_from=gen_system_wide_from,
        gen_system_wide_to=gen_system_wide_to,
        cophsl_system_wide_from=cophsl_system_wide_from,
        cophsl_system_wide_to=cophsl_system_wide_to,
        stwpf_system_wide_from=stwpf_system_wide_from,
        stwpf_system_wide_to=stwpf_system_wide_to,
        wgrpp_system_wide_from=wgrpp_system_wide_from,
        wgrpp_system_wide_to=wgrpp_system_wide_to,
        gen_panhandle_from=gen_panhandle_from,
        gen_panhandle_to=gen_panhandle_to,
        cophsl_panhandle_from=cophsl_panhandle_from,
        cophsl_panhandle_to=cophsl_panhandle_to,
        stwpf_panhandle_from=stwpf_panhandle_from,
        stwpf_panhandle_to=stwpf_panhandle_to,
        wgrpp_panhandle_from=wgrpp_panhandle_from,
        wgrpp_panhandle_to=wgrpp_panhandle_to,
        gen_coastal_from=gen_coastal_from,
        gen_coastal_to=gen_coastal_to,
        cophsl_coastal_from=cophsl_coastal_from,
        cophsl_coastal_to=cophsl_coastal_to,
        stwpf_coastal_from=stwpf_coastal_from,
        stwpf_coastal_to=stwpf_coastal_to,
        wgrpp_coastal_from=wgrpp_coastal_from,
        wgrpp_coastal_to=wgrpp_coastal_to,
        gen_south_from=gen_south_from,
        gen_south_to=gen_south_to,
        cophsl_south_from=cophsl_south_from,
        cophsl_south_to=cophsl_south_to,
        stwpf_south_from=stwpf_south_from,
        stwpf_south_to=stwpf_south_to,
        wgrpp_south_from=wgrpp_south_from,
        wgrpp_south_to=wgrpp_south_to,
        gen_west_from=gen_west_from,
        gen_west_to=gen_west_to,
        cophsl_west_from=cophsl_west_from,
        cophsl_west_to=cophsl_west_to,
        stwpf_west_from=stwpf_west_from,
        stwpf_west_to=stwpf_west_to,
        hsl_system_wide_from=hsl_system_wide_from,
        hsl_system_wide_to=hsl_system_wide_to,
        wgrpp_west_from=wgrpp_west_from,
        wgrpp_west_to=wgrpp_west_to,
        gen_north_from=gen_north_from,
        gen_north_to=gen_north_to,
        cophsl_north_from=cophsl_north_from,
        cophsl_north_to=cophsl_north_to,
        stwpf_north_from=stwpf_north_from,
        stwpf_north_to=stwpf_north_to,
        wgrpp_north_from=wgrpp_north_from,
        wgrpp_north_to=wgrpp_north_to,
        dst_flag=dst_flag,
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
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    gen_system_wide_from: Union[Unset, float] = UNSET,
    gen_system_wide_to: Union[Unset, float] = UNSET,
    cophsl_system_wide_from: Union[Unset, float] = UNSET,
    cophsl_system_wide_to: Union[Unset, float] = UNSET,
    stwpf_system_wide_from: Union[Unset, float] = UNSET,
    stwpf_system_wide_to: Union[Unset, float] = UNSET,
    wgrpp_system_wide_from: Union[Unset, float] = UNSET,
    wgrpp_system_wide_to: Union[Unset, float] = UNSET,
    gen_panhandle_from: Union[Unset, float] = UNSET,
    gen_panhandle_to: Union[Unset, float] = UNSET,
    cophsl_panhandle_from: Union[Unset, float] = UNSET,
    cophsl_panhandle_to: Union[Unset, float] = UNSET,
    stwpf_panhandle_from: Union[Unset, float] = UNSET,
    stwpf_panhandle_to: Union[Unset, float] = UNSET,
    wgrpp_panhandle_from: Union[Unset, float] = UNSET,
    wgrpp_panhandle_to: Union[Unset, float] = UNSET,
    gen_coastal_from: Union[Unset, float] = UNSET,
    gen_coastal_to: Union[Unset, float] = UNSET,
    cophsl_coastal_from: Union[Unset, float] = UNSET,
    cophsl_coastal_to: Union[Unset, float] = UNSET,
    stwpf_coastal_from: Union[Unset, float] = UNSET,
    stwpf_coastal_to: Union[Unset, float] = UNSET,
    wgrpp_coastal_from: Union[Unset, float] = UNSET,
    wgrpp_coastal_to: Union[Unset, float] = UNSET,
    gen_south_from: Union[Unset, float] = UNSET,
    gen_south_to: Union[Unset, float] = UNSET,
    cophsl_south_from: Union[Unset, float] = UNSET,
    cophsl_south_to: Union[Unset, float] = UNSET,
    stwpf_south_from: Union[Unset, float] = UNSET,
    stwpf_south_to: Union[Unset, float] = UNSET,
    wgrpp_south_from: Union[Unset, float] = UNSET,
    wgrpp_south_to: Union[Unset, float] = UNSET,
    gen_west_from: Union[Unset, float] = UNSET,
    gen_west_to: Union[Unset, float] = UNSET,
    cophsl_west_from: Union[Unset, float] = UNSET,
    cophsl_west_to: Union[Unset, float] = UNSET,
    stwpf_west_from: Union[Unset, float] = UNSET,
    stwpf_west_to: Union[Unset, float] = UNSET,
    hsl_system_wide_from: Union[Unset, float] = UNSET,
    hsl_system_wide_to: Union[Unset, float] = UNSET,
    wgrpp_west_from: Union[Unset, float] = UNSET,
    wgrpp_west_to: Union[Unset, float] = UNSET,
    gen_north_from: Union[Unset, float] = UNSET,
    gen_north_to: Union[Unset, float] = UNSET,
    cophsl_north_from: Union[Unset, float] = UNSET,
    cophsl_north_to: Union[Unset, float] = UNSET,
    stwpf_north_from: Union[Unset, float] = UNSET,
    stwpf_north_to: Union[Unset, float] = UNSET,
    wgrpp_north_from: Union[Unset, float] = UNSET,
    wgrpp_north_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """Wind Power Production - Hourly Averaged Actual and Forecasted Values by Geographical Region

     Wind Power Production - Hourly Averaged Actual and Forecasted Values by Geographical Region

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        gen_system_wide_from (Union[Unset, float]):
        gen_system_wide_to (Union[Unset, float]):
        cophsl_system_wide_from (Union[Unset, float]):
        cophsl_system_wide_to (Union[Unset, float]):
        stwpf_system_wide_from (Union[Unset, float]):
        stwpf_system_wide_to (Union[Unset, float]):
        wgrpp_system_wide_from (Union[Unset, float]):
        wgrpp_system_wide_to (Union[Unset, float]):
        gen_panhandle_from (Union[Unset, float]):
        gen_panhandle_to (Union[Unset, float]):
        cophsl_panhandle_from (Union[Unset, float]):
        cophsl_panhandle_to (Union[Unset, float]):
        stwpf_panhandle_from (Union[Unset, float]):
        stwpf_panhandle_to (Union[Unset, float]):
        wgrpp_panhandle_from (Union[Unset, float]):
        wgrpp_panhandle_to (Union[Unset, float]):
        gen_coastal_from (Union[Unset, float]):
        gen_coastal_to (Union[Unset, float]):
        cophsl_coastal_from (Union[Unset, float]):
        cophsl_coastal_to (Union[Unset, float]):
        stwpf_coastal_from (Union[Unset, float]):
        stwpf_coastal_to (Union[Unset, float]):
        wgrpp_coastal_from (Union[Unset, float]):
        wgrpp_coastal_to (Union[Unset, float]):
        gen_south_from (Union[Unset, float]):
        gen_south_to (Union[Unset, float]):
        cophsl_south_from (Union[Unset, float]):
        cophsl_south_to (Union[Unset, float]):
        stwpf_south_from (Union[Unset, float]):
        stwpf_south_to (Union[Unset, float]):
        wgrpp_south_from (Union[Unset, float]):
        wgrpp_south_to (Union[Unset, float]):
        gen_west_from (Union[Unset, float]):
        gen_west_to (Union[Unset, float]):
        cophsl_west_from (Union[Unset, float]):
        cophsl_west_to (Union[Unset, float]):
        stwpf_west_from (Union[Unset, float]):
        stwpf_west_to (Union[Unset, float]):
        hsl_system_wide_from (Union[Unset, float]):
        hsl_system_wide_to (Union[Unset, float]):
        wgrpp_west_from (Union[Unset, float]):
        wgrpp_west_to (Union[Unset, float]):
        gen_north_from (Union[Unset, float]):
        gen_north_to (Union[Unset, float]):
        cophsl_north_from (Union[Unset, float]):
        cophsl_north_to (Union[Unset, float]):
        stwpf_north_from (Union[Unset, float]):
        stwpf_north_to (Union[Unset, float]):
        wgrpp_north_from (Union[Unset, float]):
        wgrpp_north_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
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
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        gen_system_wide_from=gen_system_wide_from,
        gen_system_wide_to=gen_system_wide_to,
        cophsl_system_wide_from=cophsl_system_wide_from,
        cophsl_system_wide_to=cophsl_system_wide_to,
        stwpf_system_wide_from=stwpf_system_wide_from,
        stwpf_system_wide_to=stwpf_system_wide_to,
        wgrpp_system_wide_from=wgrpp_system_wide_from,
        wgrpp_system_wide_to=wgrpp_system_wide_to,
        gen_panhandle_from=gen_panhandle_from,
        gen_panhandle_to=gen_panhandle_to,
        cophsl_panhandle_from=cophsl_panhandle_from,
        cophsl_panhandle_to=cophsl_panhandle_to,
        stwpf_panhandle_from=stwpf_panhandle_from,
        stwpf_panhandle_to=stwpf_panhandle_to,
        wgrpp_panhandle_from=wgrpp_panhandle_from,
        wgrpp_panhandle_to=wgrpp_panhandle_to,
        gen_coastal_from=gen_coastal_from,
        gen_coastal_to=gen_coastal_to,
        cophsl_coastal_from=cophsl_coastal_from,
        cophsl_coastal_to=cophsl_coastal_to,
        stwpf_coastal_from=stwpf_coastal_from,
        stwpf_coastal_to=stwpf_coastal_to,
        wgrpp_coastal_from=wgrpp_coastal_from,
        wgrpp_coastal_to=wgrpp_coastal_to,
        gen_south_from=gen_south_from,
        gen_south_to=gen_south_to,
        cophsl_south_from=cophsl_south_from,
        cophsl_south_to=cophsl_south_to,
        stwpf_south_from=stwpf_south_from,
        stwpf_south_to=stwpf_south_to,
        wgrpp_south_from=wgrpp_south_from,
        wgrpp_south_to=wgrpp_south_to,
        gen_west_from=gen_west_from,
        gen_west_to=gen_west_to,
        cophsl_west_from=cophsl_west_from,
        cophsl_west_to=cophsl_west_to,
        stwpf_west_from=stwpf_west_from,
        stwpf_west_to=stwpf_west_to,
        hsl_system_wide_from=hsl_system_wide_from,
        hsl_system_wide_to=hsl_system_wide_to,
        wgrpp_west_from=wgrpp_west_from,
        wgrpp_west_to=wgrpp_west_to,
        gen_north_from=gen_north_from,
        gen_north_to=gen_north_to,
        cophsl_north_from=cophsl_north_from,
        cophsl_north_to=cophsl_north_to,
        stwpf_north_from=stwpf_north_from,
        stwpf_north_to=stwpf_north_to,
        wgrpp_north_from=wgrpp_north_from,
        wgrpp_north_to=wgrpp_north_to,
        dst_flag=dst_flag,
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
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    gen_system_wide_from: Union[Unset, float] = UNSET,
    gen_system_wide_to: Union[Unset, float] = UNSET,
    cophsl_system_wide_from: Union[Unset, float] = UNSET,
    cophsl_system_wide_to: Union[Unset, float] = UNSET,
    stwpf_system_wide_from: Union[Unset, float] = UNSET,
    stwpf_system_wide_to: Union[Unset, float] = UNSET,
    wgrpp_system_wide_from: Union[Unset, float] = UNSET,
    wgrpp_system_wide_to: Union[Unset, float] = UNSET,
    gen_panhandle_from: Union[Unset, float] = UNSET,
    gen_panhandle_to: Union[Unset, float] = UNSET,
    cophsl_panhandle_from: Union[Unset, float] = UNSET,
    cophsl_panhandle_to: Union[Unset, float] = UNSET,
    stwpf_panhandle_from: Union[Unset, float] = UNSET,
    stwpf_panhandle_to: Union[Unset, float] = UNSET,
    wgrpp_panhandle_from: Union[Unset, float] = UNSET,
    wgrpp_panhandle_to: Union[Unset, float] = UNSET,
    gen_coastal_from: Union[Unset, float] = UNSET,
    gen_coastal_to: Union[Unset, float] = UNSET,
    cophsl_coastal_from: Union[Unset, float] = UNSET,
    cophsl_coastal_to: Union[Unset, float] = UNSET,
    stwpf_coastal_from: Union[Unset, float] = UNSET,
    stwpf_coastal_to: Union[Unset, float] = UNSET,
    wgrpp_coastal_from: Union[Unset, float] = UNSET,
    wgrpp_coastal_to: Union[Unset, float] = UNSET,
    gen_south_from: Union[Unset, float] = UNSET,
    gen_south_to: Union[Unset, float] = UNSET,
    cophsl_south_from: Union[Unset, float] = UNSET,
    cophsl_south_to: Union[Unset, float] = UNSET,
    stwpf_south_from: Union[Unset, float] = UNSET,
    stwpf_south_to: Union[Unset, float] = UNSET,
    wgrpp_south_from: Union[Unset, float] = UNSET,
    wgrpp_south_to: Union[Unset, float] = UNSET,
    gen_west_from: Union[Unset, float] = UNSET,
    gen_west_to: Union[Unset, float] = UNSET,
    cophsl_west_from: Union[Unset, float] = UNSET,
    cophsl_west_to: Union[Unset, float] = UNSET,
    stwpf_west_from: Union[Unset, float] = UNSET,
    stwpf_west_to: Union[Unset, float] = UNSET,
    hsl_system_wide_from: Union[Unset, float] = UNSET,
    hsl_system_wide_to: Union[Unset, float] = UNSET,
    wgrpp_west_from: Union[Unset, float] = UNSET,
    wgrpp_west_to: Union[Unset, float] = UNSET,
    gen_north_from: Union[Unset, float] = UNSET,
    gen_north_to: Union[Unset, float] = UNSET,
    cophsl_north_from: Union[Unset, float] = UNSET,
    cophsl_north_to: Union[Unset, float] = UNSET,
    stwpf_north_from: Union[Unset, float] = UNSET,
    stwpf_north_to: Union[Unset, float] = UNSET,
    wgrpp_north_from: Union[Unset, float] = UNSET,
    wgrpp_north_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """Wind Power Production - Hourly Averaged Actual and Forecasted Values by Geographical Region

     Wind Power Production - Hourly Averaged Actual and Forecasted Values by Geographical Region

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        gen_system_wide_from (Union[Unset, float]):
        gen_system_wide_to (Union[Unset, float]):
        cophsl_system_wide_from (Union[Unset, float]):
        cophsl_system_wide_to (Union[Unset, float]):
        stwpf_system_wide_from (Union[Unset, float]):
        stwpf_system_wide_to (Union[Unset, float]):
        wgrpp_system_wide_from (Union[Unset, float]):
        wgrpp_system_wide_to (Union[Unset, float]):
        gen_panhandle_from (Union[Unset, float]):
        gen_panhandle_to (Union[Unset, float]):
        cophsl_panhandle_from (Union[Unset, float]):
        cophsl_panhandle_to (Union[Unset, float]):
        stwpf_panhandle_from (Union[Unset, float]):
        stwpf_panhandle_to (Union[Unset, float]):
        wgrpp_panhandle_from (Union[Unset, float]):
        wgrpp_panhandle_to (Union[Unset, float]):
        gen_coastal_from (Union[Unset, float]):
        gen_coastal_to (Union[Unset, float]):
        cophsl_coastal_from (Union[Unset, float]):
        cophsl_coastal_to (Union[Unset, float]):
        stwpf_coastal_from (Union[Unset, float]):
        stwpf_coastal_to (Union[Unset, float]):
        wgrpp_coastal_from (Union[Unset, float]):
        wgrpp_coastal_to (Union[Unset, float]):
        gen_south_from (Union[Unset, float]):
        gen_south_to (Union[Unset, float]):
        cophsl_south_from (Union[Unset, float]):
        cophsl_south_to (Union[Unset, float]):
        stwpf_south_from (Union[Unset, float]):
        stwpf_south_to (Union[Unset, float]):
        wgrpp_south_from (Union[Unset, float]):
        wgrpp_south_to (Union[Unset, float]):
        gen_west_from (Union[Unset, float]):
        gen_west_to (Union[Unset, float]):
        cophsl_west_from (Union[Unset, float]):
        cophsl_west_to (Union[Unset, float]):
        stwpf_west_from (Union[Unset, float]):
        stwpf_west_to (Union[Unset, float]):
        hsl_system_wide_from (Union[Unset, float]):
        hsl_system_wide_to (Union[Unset, float]):
        wgrpp_west_from (Union[Unset, float]):
        wgrpp_west_to (Union[Unset, float]):
        gen_north_from (Union[Unset, float]):
        gen_north_to (Union[Unset, float]):
        cophsl_north_from (Union[Unset, float]):
        cophsl_north_to (Union[Unset, float]):
        stwpf_north_from (Union[Unset, float]):
        stwpf_north_to (Union[Unset, float]):
        wgrpp_north_from (Union[Unset, float]):
        wgrpp_north_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
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
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        hour_ending_from=hour_ending_from,
        hour_ending_to=hour_ending_to,
        gen_system_wide_from=gen_system_wide_from,
        gen_system_wide_to=gen_system_wide_to,
        cophsl_system_wide_from=cophsl_system_wide_from,
        cophsl_system_wide_to=cophsl_system_wide_to,
        stwpf_system_wide_from=stwpf_system_wide_from,
        stwpf_system_wide_to=stwpf_system_wide_to,
        wgrpp_system_wide_from=wgrpp_system_wide_from,
        wgrpp_system_wide_to=wgrpp_system_wide_to,
        gen_panhandle_from=gen_panhandle_from,
        gen_panhandle_to=gen_panhandle_to,
        cophsl_panhandle_from=cophsl_panhandle_from,
        cophsl_panhandle_to=cophsl_panhandle_to,
        stwpf_panhandle_from=stwpf_panhandle_from,
        stwpf_panhandle_to=stwpf_panhandle_to,
        wgrpp_panhandle_from=wgrpp_panhandle_from,
        wgrpp_panhandle_to=wgrpp_panhandle_to,
        gen_coastal_from=gen_coastal_from,
        gen_coastal_to=gen_coastal_to,
        cophsl_coastal_from=cophsl_coastal_from,
        cophsl_coastal_to=cophsl_coastal_to,
        stwpf_coastal_from=stwpf_coastal_from,
        stwpf_coastal_to=stwpf_coastal_to,
        wgrpp_coastal_from=wgrpp_coastal_from,
        wgrpp_coastal_to=wgrpp_coastal_to,
        gen_south_from=gen_south_from,
        gen_south_to=gen_south_to,
        cophsl_south_from=cophsl_south_from,
        cophsl_south_to=cophsl_south_to,
        stwpf_south_from=stwpf_south_from,
        stwpf_south_to=stwpf_south_to,
        wgrpp_south_from=wgrpp_south_from,
        wgrpp_south_to=wgrpp_south_to,
        gen_west_from=gen_west_from,
        gen_west_to=gen_west_to,
        cophsl_west_from=cophsl_west_from,
        cophsl_west_to=cophsl_west_to,
        stwpf_west_from=stwpf_west_from,
        stwpf_west_to=stwpf_west_to,
        hsl_system_wide_from=hsl_system_wide_from,
        hsl_system_wide_to=hsl_system_wide_to,
        wgrpp_west_from=wgrpp_west_from,
        wgrpp_west_to=wgrpp_west_to,
        gen_north_from=gen_north_from,
        gen_north_to=gen_north_to,
        cophsl_north_from=cophsl_north_from,
        cophsl_north_to=cophsl_north_to,
        stwpf_north_from=stwpf_north_from,
        stwpf_north_to=stwpf_north_to,
        wgrpp_north_from=wgrpp_north_from,
        wgrpp_north_to=wgrpp_north_to,
        dst_flag=dst_flag,
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
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending_from: Union[Unset, int] = UNSET,
    hour_ending_to: Union[Unset, int] = UNSET,
    gen_system_wide_from: Union[Unset, float] = UNSET,
    gen_system_wide_to: Union[Unset, float] = UNSET,
    cophsl_system_wide_from: Union[Unset, float] = UNSET,
    cophsl_system_wide_to: Union[Unset, float] = UNSET,
    stwpf_system_wide_from: Union[Unset, float] = UNSET,
    stwpf_system_wide_to: Union[Unset, float] = UNSET,
    wgrpp_system_wide_from: Union[Unset, float] = UNSET,
    wgrpp_system_wide_to: Union[Unset, float] = UNSET,
    gen_panhandle_from: Union[Unset, float] = UNSET,
    gen_panhandle_to: Union[Unset, float] = UNSET,
    cophsl_panhandle_from: Union[Unset, float] = UNSET,
    cophsl_panhandle_to: Union[Unset, float] = UNSET,
    stwpf_panhandle_from: Union[Unset, float] = UNSET,
    stwpf_panhandle_to: Union[Unset, float] = UNSET,
    wgrpp_panhandle_from: Union[Unset, float] = UNSET,
    wgrpp_panhandle_to: Union[Unset, float] = UNSET,
    gen_coastal_from: Union[Unset, float] = UNSET,
    gen_coastal_to: Union[Unset, float] = UNSET,
    cophsl_coastal_from: Union[Unset, float] = UNSET,
    cophsl_coastal_to: Union[Unset, float] = UNSET,
    stwpf_coastal_from: Union[Unset, float] = UNSET,
    stwpf_coastal_to: Union[Unset, float] = UNSET,
    wgrpp_coastal_from: Union[Unset, float] = UNSET,
    wgrpp_coastal_to: Union[Unset, float] = UNSET,
    gen_south_from: Union[Unset, float] = UNSET,
    gen_south_to: Union[Unset, float] = UNSET,
    cophsl_south_from: Union[Unset, float] = UNSET,
    cophsl_south_to: Union[Unset, float] = UNSET,
    stwpf_south_from: Union[Unset, float] = UNSET,
    stwpf_south_to: Union[Unset, float] = UNSET,
    wgrpp_south_from: Union[Unset, float] = UNSET,
    wgrpp_south_to: Union[Unset, float] = UNSET,
    gen_west_from: Union[Unset, float] = UNSET,
    gen_west_to: Union[Unset, float] = UNSET,
    cophsl_west_from: Union[Unset, float] = UNSET,
    cophsl_west_to: Union[Unset, float] = UNSET,
    stwpf_west_from: Union[Unset, float] = UNSET,
    stwpf_west_to: Union[Unset, float] = UNSET,
    hsl_system_wide_from: Union[Unset, float] = UNSET,
    hsl_system_wide_to: Union[Unset, float] = UNSET,
    wgrpp_west_from: Union[Unset, float] = UNSET,
    wgrpp_west_to: Union[Unset, float] = UNSET,
    gen_north_from: Union[Unset, float] = UNSET,
    gen_north_to: Union[Unset, float] = UNSET,
    cophsl_north_from: Union[Unset, float] = UNSET,
    cophsl_north_to: Union[Unset, float] = UNSET,
    stwpf_north_from: Union[Unset, float] = UNSET,
    stwpf_north_to: Union[Unset, float] = UNSET,
    wgrpp_north_from: Union[Unset, float] = UNSET,
    wgrpp_north_to: Union[Unset, float] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """Wind Power Production - Hourly Averaged Actual and Forecasted Values by Geographical Region

     Wind Power Production - Hourly Averaged Actual and Forecasted Values by Geographical Region

    Args:
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending_from (Union[Unset, int]):
        hour_ending_to (Union[Unset, int]):
        gen_system_wide_from (Union[Unset, float]):
        gen_system_wide_to (Union[Unset, float]):
        cophsl_system_wide_from (Union[Unset, float]):
        cophsl_system_wide_to (Union[Unset, float]):
        stwpf_system_wide_from (Union[Unset, float]):
        stwpf_system_wide_to (Union[Unset, float]):
        wgrpp_system_wide_from (Union[Unset, float]):
        wgrpp_system_wide_to (Union[Unset, float]):
        gen_panhandle_from (Union[Unset, float]):
        gen_panhandle_to (Union[Unset, float]):
        cophsl_panhandle_from (Union[Unset, float]):
        cophsl_panhandle_to (Union[Unset, float]):
        stwpf_panhandle_from (Union[Unset, float]):
        stwpf_panhandle_to (Union[Unset, float]):
        wgrpp_panhandle_from (Union[Unset, float]):
        wgrpp_panhandle_to (Union[Unset, float]):
        gen_coastal_from (Union[Unset, float]):
        gen_coastal_to (Union[Unset, float]):
        cophsl_coastal_from (Union[Unset, float]):
        cophsl_coastal_to (Union[Unset, float]):
        stwpf_coastal_from (Union[Unset, float]):
        stwpf_coastal_to (Union[Unset, float]):
        wgrpp_coastal_from (Union[Unset, float]):
        wgrpp_coastal_to (Union[Unset, float]):
        gen_south_from (Union[Unset, float]):
        gen_south_to (Union[Unset, float]):
        cophsl_south_from (Union[Unset, float]):
        cophsl_south_to (Union[Unset, float]):
        stwpf_south_from (Union[Unset, float]):
        stwpf_south_to (Union[Unset, float]):
        wgrpp_south_from (Union[Unset, float]):
        wgrpp_south_to (Union[Unset, float]):
        gen_west_from (Union[Unset, float]):
        gen_west_to (Union[Unset, float]):
        cophsl_west_from (Union[Unset, float]):
        cophsl_west_to (Union[Unset, float]):
        stwpf_west_from (Union[Unset, float]):
        stwpf_west_to (Union[Unset, float]):
        hsl_system_wide_from (Union[Unset, float]):
        hsl_system_wide_to (Union[Unset, float]):
        wgrpp_west_from (Union[Unset, float]):
        wgrpp_west_to (Union[Unset, float]):
        gen_north_from (Union[Unset, float]):
        gen_north_to (Union[Unset, float]):
        cophsl_north_from (Union[Unset, float]):
        cophsl_north_to (Union[Unset, float]):
        stwpf_north_from (Union[Unset, float]):
        stwpf_north_to (Union[Unset, float]):
        wgrpp_north_from (Union[Unset, float]):
        wgrpp_north_to (Union[Unset, float]):
        dst_flag (Union[Unset, bool]):
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
            delivery_date_from=delivery_date_from,
            delivery_date_to=delivery_date_to,
            hour_ending_from=hour_ending_from,
            hour_ending_to=hour_ending_to,
            gen_system_wide_from=gen_system_wide_from,
            gen_system_wide_to=gen_system_wide_to,
            cophsl_system_wide_from=cophsl_system_wide_from,
            cophsl_system_wide_to=cophsl_system_wide_to,
            stwpf_system_wide_from=stwpf_system_wide_from,
            stwpf_system_wide_to=stwpf_system_wide_to,
            wgrpp_system_wide_from=wgrpp_system_wide_from,
            wgrpp_system_wide_to=wgrpp_system_wide_to,
            gen_panhandle_from=gen_panhandle_from,
            gen_panhandle_to=gen_panhandle_to,
            cophsl_panhandle_from=cophsl_panhandle_from,
            cophsl_panhandle_to=cophsl_panhandle_to,
            stwpf_panhandle_from=stwpf_panhandle_from,
            stwpf_panhandle_to=stwpf_panhandle_to,
            wgrpp_panhandle_from=wgrpp_panhandle_from,
            wgrpp_panhandle_to=wgrpp_panhandle_to,
            gen_coastal_from=gen_coastal_from,
            gen_coastal_to=gen_coastal_to,
            cophsl_coastal_from=cophsl_coastal_from,
            cophsl_coastal_to=cophsl_coastal_to,
            stwpf_coastal_from=stwpf_coastal_from,
            stwpf_coastal_to=stwpf_coastal_to,
            wgrpp_coastal_from=wgrpp_coastal_from,
            wgrpp_coastal_to=wgrpp_coastal_to,
            gen_south_from=gen_south_from,
            gen_south_to=gen_south_to,
            cophsl_south_from=cophsl_south_from,
            cophsl_south_to=cophsl_south_to,
            stwpf_south_from=stwpf_south_from,
            stwpf_south_to=stwpf_south_to,
            wgrpp_south_from=wgrpp_south_from,
            wgrpp_south_to=wgrpp_south_to,
            gen_west_from=gen_west_from,
            gen_west_to=gen_west_to,
            cophsl_west_from=cophsl_west_from,
            cophsl_west_to=cophsl_west_to,
            stwpf_west_from=stwpf_west_from,
            stwpf_west_to=stwpf_west_to,
            hsl_system_wide_from=hsl_system_wide_from,
            hsl_system_wide_to=hsl_system_wide_to,
            wgrpp_west_from=wgrpp_west_from,
            wgrpp_west_to=wgrpp_west_to,
            gen_north_from=gen_north_from,
            gen_north_to=gen_north_to,
            cophsl_north_from=cophsl_north_from,
            cophsl_north_to=cophsl_north_to,
            stwpf_north_from=stwpf_north_from,
            stwpf_north_to=stwpf_north_to,
            wgrpp_north_from=wgrpp_north_from,
            wgrpp_north_to=wgrpp_north_to,
            dst_flag=dst_flag,
            posted_datetime_from=posted_datetime_from,
            posted_datetime_to=posted_datetime_to,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
