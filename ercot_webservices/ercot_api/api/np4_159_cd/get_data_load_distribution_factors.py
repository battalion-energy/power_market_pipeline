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
    ldf_date_from: Union[Unset, str] = UNSET,
    ldf_date_to: Union[Unset, str] = UNSET,
    ldf_hour: Union[Unset, str] = UNSET,
    substation: Union[Unset, str] = UNSET,
    distribution_factor_from: Union[Unset, float] = UNSET,
    distribution_factor_to: Union[Unset, float] = UNSET,
    load_id: Union[Unset, str] = UNSET,
    mvar_distribution_factor_from: Union[Unset, float] = UNSET,
    mvar_distribution_factor_to: Union[Unset, float] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    mrid_load: Union[Unset, str] = UNSET,
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

    params["LDFDateFrom"] = ldf_date_from

    params["LDFDateTo"] = ldf_date_to

    params["LDFHour"] = ldf_hour

    params["substation"] = substation

    params["distributionFactorFrom"] = distribution_factor_from

    params["distributionFactorTo"] = distribution_factor_to

    params["loadId"] = load_id

    params["MVARDistributionFactorFrom"] = mvar_distribution_factor_from

    params["MVARDistributionFactorTo"] = mvar_distribution_factor_to

    params["postedDatetimeFrom"] = posted_datetime_from

    params["postedDatetimeTo"] = posted_datetime_to

    params["MRIDLoad"] = mrid_load

    params["DSTFlag"] = dst_flag

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np4-159-cd/load_distribution_factors",
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
    ldf_date_from: Union[Unset, str] = UNSET,
    ldf_date_to: Union[Unset, str] = UNSET,
    ldf_hour: Union[Unset, str] = UNSET,
    substation: Union[Unset, str] = UNSET,
    distribution_factor_from: Union[Unset, float] = UNSET,
    distribution_factor_to: Union[Unset, float] = UNSET,
    load_id: Union[Unset, str] = UNSET,
    mvar_distribution_factor_from: Union[Unset, float] = UNSET,
    mvar_distribution_factor_to: Union[Unset, float] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    mrid_load: Union[Unset, str] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """Load Distribution Factors

     Load Distribution Factors

    Args:
        ldf_date_from (Union[Unset, str]):
        ldf_date_to (Union[Unset, str]):
        ldf_hour (Union[Unset, str]):
        substation (Union[Unset, str]):
        distribution_factor_from (Union[Unset, float]):
        distribution_factor_to (Union[Unset, float]):
        load_id (Union[Unset, str]):
        mvar_distribution_factor_from (Union[Unset, float]):
        mvar_distribution_factor_to (Union[Unset, float]):
        posted_datetime_from (Union[Unset, str]):
        posted_datetime_to (Union[Unset, str]):
        mrid_load (Union[Unset, str]):
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
        ldf_date_from=ldf_date_from,
        ldf_date_to=ldf_date_to,
        ldf_hour=ldf_hour,
        substation=substation,
        distribution_factor_from=distribution_factor_from,
        distribution_factor_to=distribution_factor_to,
        load_id=load_id,
        mvar_distribution_factor_from=mvar_distribution_factor_from,
        mvar_distribution_factor_to=mvar_distribution_factor_to,
        posted_datetime_from=posted_datetime_from,
        posted_datetime_to=posted_datetime_to,
        mrid_load=mrid_load,
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
    ldf_date_from: Union[Unset, str] = UNSET,
    ldf_date_to: Union[Unset, str] = UNSET,
    ldf_hour: Union[Unset, str] = UNSET,
    substation: Union[Unset, str] = UNSET,
    distribution_factor_from: Union[Unset, float] = UNSET,
    distribution_factor_to: Union[Unset, float] = UNSET,
    load_id: Union[Unset, str] = UNSET,
    mvar_distribution_factor_from: Union[Unset, float] = UNSET,
    mvar_distribution_factor_to: Union[Unset, float] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    mrid_load: Union[Unset, str] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """Load Distribution Factors

     Load Distribution Factors

    Args:
        ldf_date_from (Union[Unset, str]):
        ldf_date_to (Union[Unset, str]):
        ldf_hour (Union[Unset, str]):
        substation (Union[Unset, str]):
        distribution_factor_from (Union[Unset, float]):
        distribution_factor_to (Union[Unset, float]):
        load_id (Union[Unset, str]):
        mvar_distribution_factor_from (Union[Unset, float]):
        mvar_distribution_factor_to (Union[Unset, float]):
        posted_datetime_from (Union[Unset, str]):
        posted_datetime_to (Union[Unset, str]):
        mrid_load (Union[Unset, str]):
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
        ldf_date_from=ldf_date_from,
        ldf_date_to=ldf_date_to,
        ldf_hour=ldf_hour,
        substation=substation,
        distribution_factor_from=distribution_factor_from,
        distribution_factor_to=distribution_factor_to,
        load_id=load_id,
        mvar_distribution_factor_from=mvar_distribution_factor_from,
        mvar_distribution_factor_to=mvar_distribution_factor_to,
        posted_datetime_from=posted_datetime_from,
        posted_datetime_to=posted_datetime_to,
        mrid_load=mrid_load,
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
    ldf_date_from: Union[Unset, str] = UNSET,
    ldf_date_to: Union[Unset, str] = UNSET,
    ldf_hour: Union[Unset, str] = UNSET,
    substation: Union[Unset, str] = UNSET,
    distribution_factor_from: Union[Unset, float] = UNSET,
    distribution_factor_to: Union[Unset, float] = UNSET,
    load_id: Union[Unset, str] = UNSET,
    mvar_distribution_factor_from: Union[Unset, float] = UNSET,
    mvar_distribution_factor_to: Union[Unset, float] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    mrid_load: Union[Unset, str] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """Load Distribution Factors

     Load Distribution Factors

    Args:
        ldf_date_from (Union[Unset, str]):
        ldf_date_to (Union[Unset, str]):
        ldf_hour (Union[Unset, str]):
        substation (Union[Unset, str]):
        distribution_factor_from (Union[Unset, float]):
        distribution_factor_to (Union[Unset, float]):
        load_id (Union[Unset, str]):
        mvar_distribution_factor_from (Union[Unset, float]):
        mvar_distribution_factor_to (Union[Unset, float]):
        posted_datetime_from (Union[Unset, str]):
        posted_datetime_to (Union[Unset, str]):
        mrid_load (Union[Unset, str]):
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
        ldf_date_from=ldf_date_from,
        ldf_date_to=ldf_date_to,
        ldf_hour=ldf_hour,
        substation=substation,
        distribution_factor_from=distribution_factor_from,
        distribution_factor_to=distribution_factor_to,
        load_id=load_id,
        mvar_distribution_factor_from=mvar_distribution_factor_from,
        mvar_distribution_factor_to=mvar_distribution_factor_to,
        posted_datetime_from=posted_datetime_from,
        posted_datetime_to=posted_datetime_to,
        mrid_load=mrid_load,
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
    ldf_date_from: Union[Unset, str] = UNSET,
    ldf_date_to: Union[Unset, str] = UNSET,
    ldf_hour: Union[Unset, str] = UNSET,
    substation: Union[Unset, str] = UNSET,
    distribution_factor_from: Union[Unset, float] = UNSET,
    distribution_factor_to: Union[Unset, float] = UNSET,
    load_id: Union[Unset, str] = UNSET,
    mvar_distribution_factor_from: Union[Unset, float] = UNSET,
    mvar_distribution_factor_to: Union[Unset, float] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    mrid_load: Union[Unset, str] = UNSET,
    dst_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """Load Distribution Factors

     Load Distribution Factors

    Args:
        ldf_date_from (Union[Unset, str]):
        ldf_date_to (Union[Unset, str]):
        ldf_hour (Union[Unset, str]):
        substation (Union[Unset, str]):
        distribution_factor_from (Union[Unset, float]):
        distribution_factor_to (Union[Unset, float]):
        load_id (Union[Unset, str]):
        mvar_distribution_factor_from (Union[Unset, float]):
        mvar_distribution_factor_to (Union[Unset, float]):
        posted_datetime_from (Union[Unset, str]):
        posted_datetime_to (Union[Unset, str]):
        mrid_load (Union[Unset, str]):
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
            ldf_date_from=ldf_date_from,
            ldf_date_to=ldf_date_to,
            ldf_hour=ldf_hour,
            substation=substation,
            distribution_factor_from=distribution_factor_from,
            distribution_factor_to=distribution_factor_to,
            load_id=load_id,
            mvar_distribution_factor_from=mvar_distribution_factor_from,
            mvar_distribution_factor_to=mvar_distribution_factor_to,
            posted_datetime_from=posted_datetime_from,
            posted_datetime_to=posted_datetime_to,
            mrid_load=mrid_load,
            dst_flag=dst_flag,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
