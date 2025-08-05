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
    sum_base_point_non_irr_from: Union[Unset, float] = UNSET,
    sum_base_point_non_irr_to: Union[Unset, float] = UNSET,
    sum_hasl_non_irr_from: Union[Unset, float] = UNSET,
    sum_hasl_non_irr_to: Union[Unset, float] = UNSET,
    sum_lasl_non_irr_from: Union[Unset, float] = UNSET,
    sum_lasl_non_irr_to: Union[Unset, float] = UNSET,
    sum_base_point_wgr_from: Union[Unset, float] = UNSET,
    sum_base_point_wgr_to: Union[Unset, float] = UNSET,
    sum_haslwgr_from: Union[Unset, float] = UNSET,
    sum_haslwgr_to: Union[Unset, float] = UNSET,
    sum_laslwgr_from: Union[Unset, float] = UNSET,
    sum_laslwgr_to: Union[Unset, float] = UNSET,
    sum_base_point_pvgr_from: Union[Unset, float] = UNSET,
    sum_base_point_pvgr_to: Union[Unset, float] = UNSET,
    sum_haslpvgr_from: Union[Unset, float] = UNSET,
    sum_haslpvgr_to: Union[Unset, float] = UNSET,
    sum_laslpvgr_from: Union[Unset, float] = UNSET,
    sum_laslpvgr_to: Union[Unset, float] = UNSET,
    sum_base_point_remres_from: Union[Unset, float] = UNSET,
    sum_base_point_remres_to: Union[Unset, float] = UNSET,
    sum_haslremres_from: Union[Unset, float] = UNSET,
    sum_haslremres_to: Union[Unset, float] = UNSET,
    sum_laslremres_from: Union[Unset, float] = UNSET,
    sum_laslremres_to: Union[Unset, float] = UNSET,
    sum_gen_telem_mw_from: Union[Unset, float] = UNSET,
    sum_gen_telem_mw_to: Union[Unset, float] = UNSET,
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

    params["sumBasePointNonIRRFrom"] = sum_base_point_non_irr_from

    params["sumBasePointNonIRRTo"] = sum_base_point_non_irr_to

    params["sumHASLNonIRRFrom"] = sum_hasl_non_irr_from

    params["sumHASLNonIRRTo"] = sum_hasl_non_irr_to

    params["sumLASLNonIRRFrom"] = sum_lasl_non_irr_from

    params["sumLASLNonIRRTo"] = sum_lasl_non_irr_to

    params["sumBasePointWGRFrom"] = sum_base_point_wgr_from

    params["sumBasePointWGRTo"] = sum_base_point_wgr_to

    params["sumHASLWGRFrom"] = sum_haslwgr_from

    params["sumHASLWGRTo"] = sum_haslwgr_to

    params["sumLASLWGRFrom"] = sum_laslwgr_from

    params["sumLASLWGRTo"] = sum_laslwgr_to

    params["sumBasePointPVGRFrom"] = sum_base_point_pvgr_from

    params["sumBasePointPVGRTo"] = sum_base_point_pvgr_to

    params["sumHASLPVGRFrom"] = sum_haslpvgr_from

    params["sumHASLPVGRTo"] = sum_haslpvgr_to

    params["sumLASLPVGRFrom"] = sum_laslpvgr_from

    params["sumLASLPVGRTo"] = sum_laslpvgr_to

    params["sumBasePointREMRESFrom"] = sum_base_point_remres_from

    params["sumBasePointREMRESTo"] = sum_base_point_remres_to

    params["sumHASLREMRESFrom"] = sum_haslremres_from

    params["sumHASLREMRESTo"] = sum_haslremres_to

    params["sumLASLREMRESFrom"] = sum_laslremres_from

    params["sumLASLREMRESTo"] = sum_laslremres_to

    params["sumGenTelemMWFrom"] = sum_gen_telem_mw_from

    params["sumGenTelemMWTo"] = sum_gen_telem_mw_to

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np3-910-er/2d_agg_gen_summary",
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
    sum_base_point_non_irr_from: Union[Unset, float] = UNSET,
    sum_base_point_non_irr_to: Union[Unset, float] = UNSET,
    sum_hasl_non_irr_from: Union[Unset, float] = UNSET,
    sum_hasl_non_irr_to: Union[Unset, float] = UNSET,
    sum_lasl_non_irr_from: Union[Unset, float] = UNSET,
    sum_lasl_non_irr_to: Union[Unset, float] = UNSET,
    sum_base_point_wgr_from: Union[Unset, float] = UNSET,
    sum_base_point_wgr_to: Union[Unset, float] = UNSET,
    sum_haslwgr_from: Union[Unset, float] = UNSET,
    sum_haslwgr_to: Union[Unset, float] = UNSET,
    sum_laslwgr_from: Union[Unset, float] = UNSET,
    sum_laslwgr_to: Union[Unset, float] = UNSET,
    sum_base_point_pvgr_from: Union[Unset, float] = UNSET,
    sum_base_point_pvgr_to: Union[Unset, float] = UNSET,
    sum_haslpvgr_from: Union[Unset, float] = UNSET,
    sum_haslpvgr_to: Union[Unset, float] = UNSET,
    sum_laslpvgr_from: Union[Unset, float] = UNSET,
    sum_laslpvgr_to: Union[Unset, float] = UNSET,
    sum_base_point_remres_from: Union[Unset, float] = UNSET,
    sum_base_point_remres_to: Union[Unset, float] = UNSET,
    sum_haslremres_from: Union[Unset, float] = UNSET,
    sum_haslremres_to: Union[Unset, float] = UNSET,
    sum_laslremres_from: Union[Unset, float] = UNSET,
    sum_laslremres_to: Union[Unset, float] = UNSET,
    sum_gen_telem_mw_from: Union[Unset, float] = UNSET,
    sum_gen_telem_mw_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """2-Day Aggregated Generation Summary

     2-Day Aggregated Generation Summary

    Args:
        sced_timestamp_from (Union[Unset, str]):
        sced_timestamp_to (Union[Unset, str]):
        repeat_hour_flag (Union[Unset, bool]):
        sum_base_point_non_irr_from (Union[Unset, float]):
        sum_base_point_non_irr_to (Union[Unset, float]):
        sum_hasl_non_irr_from (Union[Unset, float]):
        sum_hasl_non_irr_to (Union[Unset, float]):
        sum_lasl_non_irr_from (Union[Unset, float]):
        sum_lasl_non_irr_to (Union[Unset, float]):
        sum_base_point_wgr_from (Union[Unset, float]):
        sum_base_point_wgr_to (Union[Unset, float]):
        sum_haslwgr_from (Union[Unset, float]):
        sum_haslwgr_to (Union[Unset, float]):
        sum_laslwgr_from (Union[Unset, float]):
        sum_laslwgr_to (Union[Unset, float]):
        sum_base_point_pvgr_from (Union[Unset, float]):
        sum_base_point_pvgr_to (Union[Unset, float]):
        sum_haslpvgr_from (Union[Unset, float]):
        sum_haslpvgr_to (Union[Unset, float]):
        sum_laslpvgr_from (Union[Unset, float]):
        sum_laslpvgr_to (Union[Unset, float]):
        sum_base_point_remres_from (Union[Unset, float]):
        sum_base_point_remres_to (Union[Unset, float]):
        sum_haslremres_from (Union[Unset, float]):
        sum_haslremres_to (Union[Unset, float]):
        sum_laslremres_from (Union[Unset, float]):
        sum_laslremres_to (Union[Unset, float]):
        sum_gen_telem_mw_from (Union[Unset, float]):
        sum_gen_telem_mw_to (Union[Unset, float]):
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
        sum_base_point_non_irr_from=sum_base_point_non_irr_from,
        sum_base_point_non_irr_to=sum_base_point_non_irr_to,
        sum_hasl_non_irr_from=sum_hasl_non_irr_from,
        sum_hasl_non_irr_to=sum_hasl_non_irr_to,
        sum_lasl_non_irr_from=sum_lasl_non_irr_from,
        sum_lasl_non_irr_to=sum_lasl_non_irr_to,
        sum_base_point_wgr_from=sum_base_point_wgr_from,
        sum_base_point_wgr_to=sum_base_point_wgr_to,
        sum_haslwgr_from=sum_haslwgr_from,
        sum_haslwgr_to=sum_haslwgr_to,
        sum_laslwgr_from=sum_laslwgr_from,
        sum_laslwgr_to=sum_laslwgr_to,
        sum_base_point_pvgr_from=sum_base_point_pvgr_from,
        sum_base_point_pvgr_to=sum_base_point_pvgr_to,
        sum_haslpvgr_from=sum_haslpvgr_from,
        sum_haslpvgr_to=sum_haslpvgr_to,
        sum_laslpvgr_from=sum_laslpvgr_from,
        sum_laslpvgr_to=sum_laslpvgr_to,
        sum_base_point_remres_from=sum_base_point_remres_from,
        sum_base_point_remres_to=sum_base_point_remres_to,
        sum_haslremres_from=sum_haslremres_from,
        sum_haslremres_to=sum_haslremres_to,
        sum_laslremres_from=sum_laslremres_from,
        sum_laslremres_to=sum_laslremres_to,
        sum_gen_telem_mw_from=sum_gen_telem_mw_from,
        sum_gen_telem_mw_to=sum_gen_telem_mw_to,
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
    sum_base_point_non_irr_from: Union[Unset, float] = UNSET,
    sum_base_point_non_irr_to: Union[Unset, float] = UNSET,
    sum_hasl_non_irr_from: Union[Unset, float] = UNSET,
    sum_hasl_non_irr_to: Union[Unset, float] = UNSET,
    sum_lasl_non_irr_from: Union[Unset, float] = UNSET,
    sum_lasl_non_irr_to: Union[Unset, float] = UNSET,
    sum_base_point_wgr_from: Union[Unset, float] = UNSET,
    sum_base_point_wgr_to: Union[Unset, float] = UNSET,
    sum_haslwgr_from: Union[Unset, float] = UNSET,
    sum_haslwgr_to: Union[Unset, float] = UNSET,
    sum_laslwgr_from: Union[Unset, float] = UNSET,
    sum_laslwgr_to: Union[Unset, float] = UNSET,
    sum_base_point_pvgr_from: Union[Unset, float] = UNSET,
    sum_base_point_pvgr_to: Union[Unset, float] = UNSET,
    sum_haslpvgr_from: Union[Unset, float] = UNSET,
    sum_haslpvgr_to: Union[Unset, float] = UNSET,
    sum_laslpvgr_from: Union[Unset, float] = UNSET,
    sum_laslpvgr_to: Union[Unset, float] = UNSET,
    sum_base_point_remres_from: Union[Unset, float] = UNSET,
    sum_base_point_remres_to: Union[Unset, float] = UNSET,
    sum_haslremres_from: Union[Unset, float] = UNSET,
    sum_haslremres_to: Union[Unset, float] = UNSET,
    sum_laslremres_from: Union[Unset, float] = UNSET,
    sum_laslremres_to: Union[Unset, float] = UNSET,
    sum_gen_telem_mw_from: Union[Unset, float] = UNSET,
    sum_gen_telem_mw_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """2-Day Aggregated Generation Summary

     2-Day Aggregated Generation Summary

    Args:
        sced_timestamp_from (Union[Unset, str]):
        sced_timestamp_to (Union[Unset, str]):
        repeat_hour_flag (Union[Unset, bool]):
        sum_base_point_non_irr_from (Union[Unset, float]):
        sum_base_point_non_irr_to (Union[Unset, float]):
        sum_hasl_non_irr_from (Union[Unset, float]):
        sum_hasl_non_irr_to (Union[Unset, float]):
        sum_lasl_non_irr_from (Union[Unset, float]):
        sum_lasl_non_irr_to (Union[Unset, float]):
        sum_base_point_wgr_from (Union[Unset, float]):
        sum_base_point_wgr_to (Union[Unset, float]):
        sum_haslwgr_from (Union[Unset, float]):
        sum_haslwgr_to (Union[Unset, float]):
        sum_laslwgr_from (Union[Unset, float]):
        sum_laslwgr_to (Union[Unset, float]):
        sum_base_point_pvgr_from (Union[Unset, float]):
        sum_base_point_pvgr_to (Union[Unset, float]):
        sum_haslpvgr_from (Union[Unset, float]):
        sum_haslpvgr_to (Union[Unset, float]):
        sum_laslpvgr_from (Union[Unset, float]):
        sum_laslpvgr_to (Union[Unset, float]):
        sum_base_point_remres_from (Union[Unset, float]):
        sum_base_point_remres_to (Union[Unset, float]):
        sum_haslremres_from (Union[Unset, float]):
        sum_haslremres_to (Union[Unset, float]):
        sum_laslremres_from (Union[Unset, float]):
        sum_laslremres_to (Union[Unset, float]):
        sum_gen_telem_mw_from (Union[Unset, float]):
        sum_gen_telem_mw_to (Union[Unset, float]):
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
        sum_base_point_non_irr_from=sum_base_point_non_irr_from,
        sum_base_point_non_irr_to=sum_base_point_non_irr_to,
        sum_hasl_non_irr_from=sum_hasl_non_irr_from,
        sum_hasl_non_irr_to=sum_hasl_non_irr_to,
        sum_lasl_non_irr_from=sum_lasl_non_irr_from,
        sum_lasl_non_irr_to=sum_lasl_non_irr_to,
        sum_base_point_wgr_from=sum_base_point_wgr_from,
        sum_base_point_wgr_to=sum_base_point_wgr_to,
        sum_haslwgr_from=sum_haslwgr_from,
        sum_haslwgr_to=sum_haslwgr_to,
        sum_laslwgr_from=sum_laslwgr_from,
        sum_laslwgr_to=sum_laslwgr_to,
        sum_base_point_pvgr_from=sum_base_point_pvgr_from,
        sum_base_point_pvgr_to=sum_base_point_pvgr_to,
        sum_haslpvgr_from=sum_haslpvgr_from,
        sum_haslpvgr_to=sum_haslpvgr_to,
        sum_laslpvgr_from=sum_laslpvgr_from,
        sum_laslpvgr_to=sum_laslpvgr_to,
        sum_base_point_remres_from=sum_base_point_remres_from,
        sum_base_point_remres_to=sum_base_point_remres_to,
        sum_haslremres_from=sum_haslremres_from,
        sum_haslremres_to=sum_haslremres_to,
        sum_laslremres_from=sum_laslremres_from,
        sum_laslremres_to=sum_laslremres_to,
        sum_gen_telem_mw_from=sum_gen_telem_mw_from,
        sum_gen_telem_mw_to=sum_gen_telem_mw_to,
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
    sum_base_point_non_irr_from: Union[Unset, float] = UNSET,
    sum_base_point_non_irr_to: Union[Unset, float] = UNSET,
    sum_hasl_non_irr_from: Union[Unset, float] = UNSET,
    sum_hasl_non_irr_to: Union[Unset, float] = UNSET,
    sum_lasl_non_irr_from: Union[Unset, float] = UNSET,
    sum_lasl_non_irr_to: Union[Unset, float] = UNSET,
    sum_base_point_wgr_from: Union[Unset, float] = UNSET,
    sum_base_point_wgr_to: Union[Unset, float] = UNSET,
    sum_haslwgr_from: Union[Unset, float] = UNSET,
    sum_haslwgr_to: Union[Unset, float] = UNSET,
    sum_laslwgr_from: Union[Unset, float] = UNSET,
    sum_laslwgr_to: Union[Unset, float] = UNSET,
    sum_base_point_pvgr_from: Union[Unset, float] = UNSET,
    sum_base_point_pvgr_to: Union[Unset, float] = UNSET,
    sum_haslpvgr_from: Union[Unset, float] = UNSET,
    sum_haslpvgr_to: Union[Unset, float] = UNSET,
    sum_laslpvgr_from: Union[Unset, float] = UNSET,
    sum_laslpvgr_to: Union[Unset, float] = UNSET,
    sum_base_point_remres_from: Union[Unset, float] = UNSET,
    sum_base_point_remres_to: Union[Unset, float] = UNSET,
    sum_haslremres_from: Union[Unset, float] = UNSET,
    sum_haslremres_to: Union[Unset, float] = UNSET,
    sum_laslremres_from: Union[Unset, float] = UNSET,
    sum_laslremres_to: Union[Unset, float] = UNSET,
    sum_gen_telem_mw_from: Union[Unset, float] = UNSET,
    sum_gen_telem_mw_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """2-Day Aggregated Generation Summary

     2-Day Aggregated Generation Summary

    Args:
        sced_timestamp_from (Union[Unset, str]):
        sced_timestamp_to (Union[Unset, str]):
        repeat_hour_flag (Union[Unset, bool]):
        sum_base_point_non_irr_from (Union[Unset, float]):
        sum_base_point_non_irr_to (Union[Unset, float]):
        sum_hasl_non_irr_from (Union[Unset, float]):
        sum_hasl_non_irr_to (Union[Unset, float]):
        sum_lasl_non_irr_from (Union[Unset, float]):
        sum_lasl_non_irr_to (Union[Unset, float]):
        sum_base_point_wgr_from (Union[Unset, float]):
        sum_base_point_wgr_to (Union[Unset, float]):
        sum_haslwgr_from (Union[Unset, float]):
        sum_haslwgr_to (Union[Unset, float]):
        sum_laslwgr_from (Union[Unset, float]):
        sum_laslwgr_to (Union[Unset, float]):
        sum_base_point_pvgr_from (Union[Unset, float]):
        sum_base_point_pvgr_to (Union[Unset, float]):
        sum_haslpvgr_from (Union[Unset, float]):
        sum_haslpvgr_to (Union[Unset, float]):
        sum_laslpvgr_from (Union[Unset, float]):
        sum_laslpvgr_to (Union[Unset, float]):
        sum_base_point_remres_from (Union[Unset, float]):
        sum_base_point_remres_to (Union[Unset, float]):
        sum_haslremres_from (Union[Unset, float]):
        sum_haslremres_to (Union[Unset, float]):
        sum_laslremres_from (Union[Unset, float]):
        sum_laslremres_to (Union[Unset, float]):
        sum_gen_telem_mw_from (Union[Unset, float]):
        sum_gen_telem_mw_to (Union[Unset, float]):
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
        sum_base_point_non_irr_from=sum_base_point_non_irr_from,
        sum_base_point_non_irr_to=sum_base_point_non_irr_to,
        sum_hasl_non_irr_from=sum_hasl_non_irr_from,
        sum_hasl_non_irr_to=sum_hasl_non_irr_to,
        sum_lasl_non_irr_from=sum_lasl_non_irr_from,
        sum_lasl_non_irr_to=sum_lasl_non_irr_to,
        sum_base_point_wgr_from=sum_base_point_wgr_from,
        sum_base_point_wgr_to=sum_base_point_wgr_to,
        sum_haslwgr_from=sum_haslwgr_from,
        sum_haslwgr_to=sum_haslwgr_to,
        sum_laslwgr_from=sum_laslwgr_from,
        sum_laslwgr_to=sum_laslwgr_to,
        sum_base_point_pvgr_from=sum_base_point_pvgr_from,
        sum_base_point_pvgr_to=sum_base_point_pvgr_to,
        sum_haslpvgr_from=sum_haslpvgr_from,
        sum_haslpvgr_to=sum_haslpvgr_to,
        sum_laslpvgr_from=sum_laslpvgr_from,
        sum_laslpvgr_to=sum_laslpvgr_to,
        sum_base_point_remres_from=sum_base_point_remres_from,
        sum_base_point_remres_to=sum_base_point_remres_to,
        sum_haslremres_from=sum_haslremres_from,
        sum_haslremres_to=sum_haslremres_to,
        sum_laslremres_from=sum_laslremres_from,
        sum_laslremres_to=sum_laslremres_to,
        sum_gen_telem_mw_from=sum_gen_telem_mw_from,
        sum_gen_telem_mw_to=sum_gen_telem_mw_to,
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
    sum_base_point_non_irr_from: Union[Unset, float] = UNSET,
    sum_base_point_non_irr_to: Union[Unset, float] = UNSET,
    sum_hasl_non_irr_from: Union[Unset, float] = UNSET,
    sum_hasl_non_irr_to: Union[Unset, float] = UNSET,
    sum_lasl_non_irr_from: Union[Unset, float] = UNSET,
    sum_lasl_non_irr_to: Union[Unset, float] = UNSET,
    sum_base_point_wgr_from: Union[Unset, float] = UNSET,
    sum_base_point_wgr_to: Union[Unset, float] = UNSET,
    sum_haslwgr_from: Union[Unset, float] = UNSET,
    sum_haslwgr_to: Union[Unset, float] = UNSET,
    sum_laslwgr_from: Union[Unset, float] = UNSET,
    sum_laslwgr_to: Union[Unset, float] = UNSET,
    sum_base_point_pvgr_from: Union[Unset, float] = UNSET,
    sum_base_point_pvgr_to: Union[Unset, float] = UNSET,
    sum_haslpvgr_from: Union[Unset, float] = UNSET,
    sum_haslpvgr_to: Union[Unset, float] = UNSET,
    sum_laslpvgr_from: Union[Unset, float] = UNSET,
    sum_laslpvgr_to: Union[Unset, float] = UNSET,
    sum_base_point_remres_from: Union[Unset, float] = UNSET,
    sum_base_point_remres_to: Union[Unset, float] = UNSET,
    sum_haslremres_from: Union[Unset, float] = UNSET,
    sum_haslremres_to: Union[Unset, float] = UNSET,
    sum_laslremres_from: Union[Unset, float] = UNSET,
    sum_laslremres_to: Union[Unset, float] = UNSET,
    sum_gen_telem_mw_from: Union[Unset, float] = UNSET,
    sum_gen_telem_mw_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """2-Day Aggregated Generation Summary

     2-Day Aggregated Generation Summary

    Args:
        sced_timestamp_from (Union[Unset, str]):
        sced_timestamp_to (Union[Unset, str]):
        repeat_hour_flag (Union[Unset, bool]):
        sum_base_point_non_irr_from (Union[Unset, float]):
        sum_base_point_non_irr_to (Union[Unset, float]):
        sum_hasl_non_irr_from (Union[Unset, float]):
        sum_hasl_non_irr_to (Union[Unset, float]):
        sum_lasl_non_irr_from (Union[Unset, float]):
        sum_lasl_non_irr_to (Union[Unset, float]):
        sum_base_point_wgr_from (Union[Unset, float]):
        sum_base_point_wgr_to (Union[Unset, float]):
        sum_haslwgr_from (Union[Unset, float]):
        sum_haslwgr_to (Union[Unset, float]):
        sum_laslwgr_from (Union[Unset, float]):
        sum_laslwgr_to (Union[Unset, float]):
        sum_base_point_pvgr_from (Union[Unset, float]):
        sum_base_point_pvgr_to (Union[Unset, float]):
        sum_haslpvgr_from (Union[Unset, float]):
        sum_haslpvgr_to (Union[Unset, float]):
        sum_laslpvgr_from (Union[Unset, float]):
        sum_laslpvgr_to (Union[Unset, float]):
        sum_base_point_remres_from (Union[Unset, float]):
        sum_base_point_remres_to (Union[Unset, float]):
        sum_haslremres_from (Union[Unset, float]):
        sum_haslremres_to (Union[Unset, float]):
        sum_laslremres_from (Union[Unset, float]):
        sum_laslremres_to (Union[Unset, float]):
        sum_gen_telem_mw_from (Union[Unset, float]):
        sum_gen_telem_mw_to (Union[Unset, float]):
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
            sum_base_point_non_irr_from=sum_base_point_non_irr_from,
            sum_base_point_non_irr_to=sum_base_point_non_irr_to,
            sum_hasl_non_irr_from=sum_hasl_non_irr_from,
            sum_hasl_non_irr_to=sum_hasl_non_irr_to,
            sum_lasl_non_irr_from=sum_lasl_non_irr_from,
            sum_lasl_non_irr_to=sum_lasl_non_irr_to,
            sum_base_point_wgr_from=sum_base_point_wgr_from,
            sum_base_point_wgr_to=sum_base_point_wgr_to,
            sum_haslwgr_from=sum_haslwgr_from,
            sum_haslwgr_to=sum_haslwgr_to,
            sum_laslwgr_from=sum_laslwgr_from,
            sum_laslwgr_to=sum_laslwgr_to,
            sum_base_point_pvgr_from=sum_base_point_pvgr_from,
            sum_base_point_pvgr_to=sum_base_point_pvgr_to,
            sum_haslpvgr_from=sum_haslpvgr_from,
            sum_haslpvgr_to=sum_haslpvgr_to,
            sum_laslpvgr_from=sum_laslpvgr_from,
            sum_laslpvgr_to=sum_laslpvgr_to,
            sum_base_point_remres_from=sum_base_point_remres_from,
            sum_base_point_remres_to=sum_base_point_remres_to,
            sum_haslremres_from=sum_haslremres_from,
            sum_haslremres_to=sum_haslremres_to,
            sum_laslremres_from=sum_laslremres_from,
            sum_laslremres_to=sum_laslremres_to,
            sum_gen_telem_mw_from=sum_gen_telem_mw_from,
            sum_gen_telem_mw_to=sum_gen_telem_mw_to,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
