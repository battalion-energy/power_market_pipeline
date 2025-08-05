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
    sum_telem_gen_mw_from: Union[Unset, float] = UNSET,
    sum_telem_gen_mw_to: Union[Unset, float] = UNSET,
    sum_telem_d_ctie_mw_from: Union[Unset, float] = UNSET,
    sum_telem_d_ctie_mw_to: Union[Unset, float] = UNSET,
    agg_load_summary_from: Union[Unset, float] = UNSET,
    agg_load_summary_to: Union[Unset, float] = UNSET,
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

    params["sumTelemGenMWFrom"] = sum_telem_gen_mw_from

    params["sumTelemGenMWTo"] = sum_telem_gen_mw_to

    params["sumTelemDCtieMWFrom"] = sum_telem_d_ctie_mw_from

    params["sumTelemDCtieMWTo"] = sum_telem_d_ctie_mw_to

    params["aggLoadSummaryFrom"] = agg_load_summary_from

    params["aggLoadSummaryTo"] = agg_load_summary_to

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np3-910-er/2d_agg_load_summary_north",
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
    sum_telem_gen_mw_from: Union[Unset, float] = UNSET,
    sum_telem_gen_mw_to: Union[Unset, float] = UNSET,
    sum_telem_d_ctie_mw_from: Union[Unset, float] = UNSET,
    sum_telem_d_ctie_mw_to: Union[Unset, float] = UNSET,
    agg_load_summary_from: Union[Unset, float] = UNSET,
    agg_load_summary_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """2-Day Aggregated Load Summary North

     2-Day Aggregated Load Summary North

    Args:
        sced_timestamp_from (Union[Unset, str]):
        sced_timestamp_to (Union[Unset, str]):
        repeat_hour_flag (Union[Unset, bool]):
        sum_telem_gen_mw_from (Union[Unset, float]):
        sum_telem_gen_mw_to (Union[Unset, float]):
        sum_telem_d_ctie_mw_from (Union[Unset, float]):
        sum_telem_d_ctie_mw_to (Union[Unset, float]):
        agg_load_summary_from (Union[Unset, float]):
        agg_load_summary_to (Union[Unset, float]):
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
        sum_telem_gen_mw_from=sum_telem_gen_mw_from,
        sum_telem_gen_mw_to=sum_telem_gen_mw_to,
        sum_telem_d_ctie_mw_from=sum_telem_d_ctie_mw_from,
        sum_telem_d_ctie_mw_to=sum_telem_d_ctie_mw_to,
        agg_load_summary_from=agg_load_summary_from,
        agg_load_summary_to=agg_load_summary_to,
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
    sum_telem_gen_mw_from: Union[Unset, float] = UNSET,
    sum_telem_gen_mw_to: Union[Unset, float] = UNSET,
    sum_telem_d_ctie_mw_from: Union[Unset, float] = UNSET,
    sum_telem_d_ctie_mw_to: Union[Unset, float] = UNSET,
    agg_load_summary_from: Union[Unset, float] = UNSET,
    agg_load_summary_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """2-Day Aggregated Load Summary North

     2-Day Aggregated Load Summary North

    Args:
        sced_timestamp_from (Union[Unset, str]):
        sced_timestamp_to (Union[Unset, str]):
        repeat_hour_flag (Union[Unset, bool]):
        sum_telem_gen_mw_from (Union[Unset, float]):
        sum_telem_gen_mw_to (Union[Unset, float]):
        sum_telem_d_ctie_mw_from (Union[Unset, float]):
        sum_telem_d_ctie_mw_to (Union[Unset, float]):
        agg_load_summary_from (Union[Unset, float]):
        agg_load_summary_to (Union[Unset, float]):
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
        sum_telem_gen_mw_from=sum_telem_gen_mw_from,
        sum_telem_gen_mw_to=sum_telem_gen_mw_to,
        sum_telem_d_ctie_mw_from=sum_telem_d_ctie_mw_from,
        sum_telem_d_ctie_mw_to=sum_telem_d_ctie_mw_to,
        agg_load_summary_from=agg_load_summary_from,
        agg_load_summary_to=agg_load_summary_to,
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
    sum_telem_gen_mw_from: Union[Unset, float] = UNSET,
    sum_telem_gen_mw_to: Union[Unset, float] = UNSET,
    sum_telem_d_ctie_mw_from: Union[Unset, float] = UNSET,
    sum_telem_d_ctie_mw_to: Union[Unset, float] = UNSET,
    agg_load_summary_from: Union[Unset, float] = UNSET,
    agg_load_summary_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """2-Day Aggregated Load Summary North

     2-Day Aggregated Load Summary North

    Args:
        sced_timestamp_from (Union[Unset, str]):
        sced_timestamp_to (Union[Unset, str]):
        repeat_hour_flag (Union[Unset, bool]):
        sum_telem_gen_mw_from (Union[Unset, float]):
        sum_telem_gen_mw_to (Union[Unset, float]):
        sum_telem_d_ctie_mw_from (Union[Unset, float]):
        sum_telem_d_ctie_mw_to (Union[Unset, float]):
        agg_load_summary_from (Union[Unset, float]):
        agg_load_summary_to (Union[Unset, float]):
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
        sum_telem_gen_mw_from=sum_telem_gen_mw_from,
        sum_telem_gen_mw_to=sum_telem_gen_mw_to,
        sum_telem_d_ctie_mw_from=sum_telem_d_ctie_mw_from,
        sum_telem_d_ctie_mw_to=sum_telem_d_ctie_mw_to,
        agg_load_summary_from=agg_load_summary_from,
        agg_load_summary_to=agg_load_summary_to,
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
    sum_telem_gen_mw_from: Union[Unset, float] = UNSET,
    sum_telem_gen_mw_to: Union[Unset, float] = UNSET,
    sum_telem_d_ctie_mw_from: Union[Unset, float] = UNSET,
    sum_telem_d_ctie_mw_to: Union[Unset, float] = UNSET,
    agg_load_summary_from: Union[Unset, float] = UNSET,
    agg_load_summary_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """2-Day Aggregated Load Summary North

     2-Day Aggregated Load Summary North

    Args:
        sced_timestamp_from (Union[Unset, str]):
        sced_timestamp_to (Union[Unset, str]):
        repeat_hour_flag (Union[Unset, bool]):
        sum_telem_gen_mw_from (Union[Unset, float]):
        sum_telem_gen_mw_to (Union[Unset, float]):
        sum_telem_d_ctie_mw_from (Union[Unset, float]):
        sum_telem_d_ctie_mw_to (Union[Unset, float]):
        agg_load_summary_from (Union[Unset, float]):
        agg_load_summary_to (Union[Unset, float]):
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
            sum_telem_gen_mw_from=sum_telem_gen_mw_from,
            sum_telem_gen_mw_to=sum_telem_gen_mw_to,
            sum_telem_d_ctie_mw_from=sum_telem_d_ctie_mw_from,
            sum_telem_d_ctie_mw_to=sum_telem_d_ctie_mw_to,
            agg_load_summary_from=agg_load_summary_from,
            agg_load_summary_to=agg_load_summary_to,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
