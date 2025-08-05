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
    dst_flag: Union[Unset, bool] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending: Union[Unset, str] = UNSET,
    coast_from: Union[Unset, float] = UNSET,
    coast_to: Union[Unset, float] = UNSET,
    east_from: Union[Unset, float] = UNSET,
    east_to: Union[Unset, float] = UNSET,
    far_west_from: Union[Unset, float] = UNSET,
    far_west_to: Union[Unset, float] = UNSET,
    north_from: Union[Unset, float] = UNSET,
    north_to: Union[Unset, float] = UNSET,
    north_central_from: Union[Unset, float] = UNSET,
    north_central_to: Union[Unset, float] = UNSET,
    south_central_from: Union[Unset, float] = UNSET,
    south_central_to: Union[Unset, float] = UNSET,
    southern_from: Union[Unset, float] = UNSET,
    southern_to: Union[Unset, float] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    west_from: Union[Unset, float] = UNSET,
    west_to: Union[Unset, float] = UNSET,
    system_total_from: Union[Unset, float] = UNSET,
    system_total_to: Union[Unset, float] = UNSET,
    model: Union[Unset, str] = UNSET,
    in_use_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["Ocp-Apim-Subscription-Key"] = ocp_apim_subscription_key

    params: dict[str, Any] = {}

    params["DSTFlag"] = dst_flag

    params["deliveryDateFrom"] = delivery_date_from

    params["deliveryDateTo"] = delivery_date_to

    params["hourEnding"] = hour_ending

    params["coastFrom"] = coast_from

    params["coastTo"] = coast_to

    params["eastFrom"] = east_from

    params["eastTo"] = east_to

    params["farWestFrom"] = far_west_from

    params["farWestTo"] = far_west_to

    params["northFrom"] = north_from

    params["northTo"] = north_to

    params["northCentralFrom"] = north_central_from

    params["northCentralTo"] = north_central_to

    params["southCentralFrom"] = south_central_from

    params["southCentralTo"] = south_central_to

    params["southernFrom"] = southern_from

    params["southernTo"] = southern_to

    params["postedDatetimeFrom"] = posted_datetime_from

    params["postedDatetimeTo"] = posted_datetime_to

    params["westFrom"] = west_from

    params["westTo"] = west_to

    params["systemTotalFrom"] = system_total_from

    params["systemTotalTo"] = system_total_to

    params["model"] = model

    params["inUseFlag"] = in_use_flag

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np3-565-cd/lf_by_model_weather_zone",
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
    dst_flag: Union[Unset, bool] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending: Union[Unset, str] = UNSET,
    coast_from: Union[Unset, float] = UNSET,
    coast_to: Union[Unset, float] = UNSET,
    east_from: Union[Unset, float] = UNSET,
    east_to: Union[Unset, float] = UNSET,
    far_west_from: Union[Unset, float] = UNSET,
    far_west_to: Union[Unset, float] = UNSET,
    north_from: Union[Unset, float] = UNSET,
    north_to: Union[Unset, float] = UNSET,
    north_central_from: Union[Unset, float] = UNSET,
    north_central_to: Union[Unset, float] = UNSET,
    south_central_from: Union[Unset, float] = UNSET,
    south_central_to: Union[Unset, float] = UNSET,
    southern_from: Union[Unset, float] = UNSET,
    southern_to: Union[Unset, float] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    west_from: Union[Unset, float] = UNSET,
    west_to: Union[Unset, float] = UNSET,
    system_total_from: Union[Unset, float] = UNSET,
    system_total_to: Union[Unset, float] = UNSET,
    model: Union[Unset, str] = UNSET,
    in_use_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """Seven-Day Load Forecast by Model and Weather Zone

     Seven-Day Load Forecast by Model and Weather Zone

    Args:
        dst_flag (Union[Unset, bool]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending (Union[Unset, str]):
        coast_from (Union[Unset, float]):
        coast_to (Union[Unset, float]):
        east_from (Union[Unset, float]):
        east_to (Union[Unset, float]):
        far_west_from (Union[Unset, float]):
        far_west_to (Union[Unset, float]):
        north_from (Union[Unset, float]):
        north_to (Union[Unset, float]):
        north_central_from (Union[Unset, float]):
        north_central_to (Union[Unset, float]):
        south_central_from (Union[Unset, float]):
        south_central_to (Union[Unset, float]):
        southern_from (Union[Unset, float]):
        southern_to (Union[Unset, float]):
        posted_datetime_from (Union[Unset, str]):
        posted_datetime_to (Union[Unset, str]):
        west_from (Union[Unset, float]):
        west_to (Union[Unset, float]):
        system_total_from (Union[Unset, float]):
        system_total_to (Union[Unset, float]):
        model (Union[Unset, str]):
        in_use_flag (Union[Unset, bool]):
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
        dst_flag=dst_flag,
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        hour_ending=hour_ending,
        coast_from=coast_from,
        coast_to=coast_to,
        east_from=east_from,
        east_to=east_to,
        far_west_from=far_west_from,
        far_west_to=far_west_to,
        north_from=north_from,
        north_to=north_to,
        north_central_from=north_central_from,
        north_central_to=north_central_to,
        south_central_from=south_central_from,
        south_central_to=south_central_to,
        southern_from=southern_from,
        southern_to=southern_to,
        posted_datetime_from=posted_datetime_from,
        posted_datetime_to=posted_datetime_to,
        west_from=west_from,
        west_to=west_to,
        system_total_from=system_total_from,
        system_total_to=system_total_to,
        model=model,
        in_use_flag=in_use_flag,
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
    dst_flag: Union[Unset, bool] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending: Union[Unset, str] = UNSET,
    coast_from: Union[Unset, float] = UNSET,
    coast_to: Union[Unset, float] = UNSET,
    east_from: Union[Unset, float] = UNSET,
    east_to: Union[Unset, float] = UNSET,
    far_west_from: Union[Unset, float] = UNSET,
    far_west_to: Union[Unset, float] = UNSET,
    north_from: Union[Unset, float] = UNSET,
    north_to: Union[Unset, float] = UNSET,
    north_central_from: Union[Unset, float] = UNSET,
    north_central_to: Union[Unset, float] = UNSET,
    south_central_from: Union[Unset, float] = UNSET,
    south_central_to: Union[Unset, float] = UNSET,
    southern_from: Union[Unset, float] = UNSET,
    southern_to: Union[Unset, float] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    west_from: Union[Unset, float] = UNSET,
    west_to: Union[Unset, float] = UNSET,
    system_total_from: Union[Unset, float] = UNSET,
    system_total_to: Union[Unset, float] = UNSET,
    model: Union[Unset, str] = UNSET,
    in_use_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """Seven-Day Load Forecast by Model and Weather Zone

     Seven-Day Load Forecast by Model and Weather Zone

    Args:
        dst_flag (Union[Unset, bool]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending (Union[Unset, str]):
        coast_from (Union[Unset, float]):
        coast_to (Union[Unset, float]):
        east_from (Union[Unset, float]):
        east_to (Union[Unset, float]):
        far_west_from (Union[Unset, float]):
        far_west_to (Union[Unset, float]):
        north_from (Union[Unset, float]):
        north_to (Union[Unset, float]):
        north_central_from (Union[Unset, float]):
        north_central_to (Union[Unset, float]):
        south_central_from (Union[Unset, float]):
        south_central_to (Union[Unset, float]):
        southern_from (Union[Unset, float]):
        southern_to (Union[Unset, float]):
        posted_datetime_from (Union[Unset, str]):
        posted_datetime_to (Union[Unset, str]):
        west_from (Union[Unset, float]):
        west_to (Union[Unset, float]):
        system_total_from (Union[Unset, float]):
        system_total_to (Union[Unset, float]):
        model (Union[Unset, str]):
        in_use_flag (Union[Unset, bool]):
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
        dst_flag=dst_flag,
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        hour_ending=hour_ending,
        coast_from=coast_from,
        coast_to=coast_to,
        east_from=east_from,
        east_to=east_to,
        far_west_from=far_west_from,
        far_west_to=far_west_to,
        north_from=north_from,
        north_to=north_to,
        north_central_from=north_central_from,
        north_central_to=north_central_to,
        south_central_from=south_central_from,
        south_central_to=south_central_to,
        southern_from=southern_from,
        southern_to=southern_to,
        posted_datetime_from=posted_datetime_from,
        posted_datetime_to=posted_datetime_to,
        west_from=west_from,
        west_to=west_to,
        system_total_from=system_total_from,
        system_total_to=system_total_to,
        model=model,
        in_use_flag=in_use_flag,
        page=page,
        size=size,
        sort=sort,
        dir_=dir_,
        ocp_apim_subscription_key=ocp_apim_subscription_key,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    dst_flag: Union[Unset, bool] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending: Union[Unset, str] = UNSET,
    coast_from: Union[Unset, float] = UNSET,
    coast_to: Union[Unset, float] = UNSET,
    east_from: Union[Unset, float] = UNSET,
    east_to: Union[Unset, float] = UNSET,
    far_west_from: Union[Unset, float] = UNSET,
    far_west_to: Union[Unset, float] = UNSET,
    north_from: Union[Unset, float] = UNSET,
    north_to: Union[Unset, float] = UNSET,
    north_central_from: Union[Unset, float] = UNSET,
    north_central_to: Union[Unset, float] = UNSET,
    south_central_from: Union[Unset, float] = UNSET,
    south_central_to: Union[Unset, float] = UNSET,
    southern_from: Union[Unset, float] = UNSET,
    southern_to: Union[Unset, float] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    west_from: Union[Unset, float] = UNSET,
    west_to: Union[Unset, float] = UNSET,
    system_total_from: Union[Unset, float] = UNSET,
    system_total_to: Union[Unset, float] = UNSET,
    model: Union[Unset, str] = UNSET,
    in_use_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """Seven-Day Load Forecast by Model and Weather Zone

     Seven-Day Load Forecast by Model and Weather Zone

    Args:
        dst_flag (Union[Unset, bool]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending (Union[Unset, str]):
        coast_from (Union[Unset, float]):
        coast_to (Union[Unset, float]):
        east_from (Union[Unset, float]):
        east_to (Union[Unset, float]):
        far_west_from (Union[Unset, float]):
        far_west_to (Union[Unset, float]):
        north_from (Union[Unset, float]):
        north_to (Union[Unset, float]):
        north_central_from (Union[Unset, float]):
        north_central_to (Union[Unset, float]):
        south_central_from (Union[Unset, float]):
        south_central_to (Union[Unset, float]):
        southern_from (Union[Unset, float]):
        southern_to (Union[Unset, float]):
        posted_datetime_from (Union[Unset, str]):
        posted_datetime_to (Union[Unset, str]):
        west_from (Union[Unset, float]):
        west_to (Union[Unset, float]):
        system_total_from (Union[Unset, float]):
        system_total_to (Union[Unset, float]):
        model (Union[Unset, str]):
        in_use_flag (Union[Unset, bool]):
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
        dst_flag=dst_flag,
        delivery_date_from=delivery_date_from,
        delivery_date_to=delivery_date_to,
        hour_ending=hour_ending,
        coast_from=coast_from,
        coast_to=coast_to,
        east_from=east_from,
        east_to=east_to,
        far_west_from=far_west_from,
        far_west_to=far_west_to,
        north_from=north_from,
        north_to=north_to,
        north_central_from=north_central_from,
        north_central_to=north_central_to,
        south_central_from=south_central_from,
        south_central_to=south_central_to,
        southern_from=southern_from,
        southern_to=southern_to,
        posted_datetime_from=posted_datetime_from,
        posted_datetime_to=posted_datetime_to,
        west_from=west_from,
        west_to=west_to,
        system_total_from=system_total_from,
        system_total_to=system_total_to,
        model=model,
        in_use_flag=in_use_flag,
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
    dst_flag: Union[Unset, bool] = UNSET,
    delivery_date_from: Union[Unset, str] = UNSET,
    delivery_date_to: Union[Unset, str] = UNSET,
    hour_ending: Union[Unset, str] = UNSET,
    coast_from: Union[Unset, float] = UNSET,
    coast_to: Union[Unset, float] = UNSET,
    east_from: Union[Unset, float] = UNSET,
    east_to: Union[Unset, float] = UNSET,
    far_west_from: Union[Unset, float] = UNSET,
    far_west_to: Union[Unset, float] = UNSET,
    north_from: Union[Unset, float] = UNSET,
    north_to: Union[Unset, float] = UNSET,
    north_central_from: Union[Unset, float] = UNSET,
    north_central_to: Union[Unset, float] = UNSET,
    south_central_from: Union[Unset, float] = UNSET,
    south_central_to: Union[Unset, float] = UNSET,
    southern_from: Union[Unset, float] = UNSET,
    southern_to: Union[Unset, float] = UNSET,
    posted_datetime_from: Union[Unset, str] = UNSET,
    posted_datetime_to: Union[Unset, str] = UNSET,
    west_from: Union[Unset, float] = UNSET,
    west_to: Union[Unset, float] = UNSET,
    system_total_from: Union[Unset, float] = UNSET,
    system_total_to: Union[Unset, float] = UNSET,
    model: Union[Unset, str] = UNSET,
    in_use_flag: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """Seven-Day Load Forecast by Model and Weather Zone

     Seven-Day Load Forecast by Model and Weather Zone

    Args:
        dst_flag (Union[Unset, bool]):
        delivery_date_from (Union[Unset, str]):
        delivery_date_to (Union[Unset, str]):
        hour_ending (Union[Unset, str]):
        coast_from (Union[Unset, float]):
        coast_to (Union[Unset, float]):
        east_from (Union[Unset, float]):
        east_to (Union[Unset, float]):
        far_west_from (Union[Unset, float]):
        far_west_to (Union[Unset, float]):
        north_from (Union[Unset, float]):
        north_to (Union[Unset, float]):
        north_central_from (Union[Unset, float]):
        north_central_to (Union[Unset, float]):
        south_central_from (Union[Unset, float]):
        south_central_to (Union[Unset, float]):
        southern_from (Union[Unset, float]):
        southern_to (Union[Unset, float]):
        posted_datetime_from (Union[Unset, str]):
        posted_datetime_to (Union[Unset, str]):
        west_from (Union[Unset, float]):
        west_to (Union[Unset, float]):
        system_total_from (Union[Unset, float]):
        system_total_to (Union[Unset, float]):
        model (Union[Unset, str]):
        in_use_flag (Union[Unset, bool]):
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
            dst_flag=dst_flag,
            delivery_date_from=delivery_date_from,
            delivery_date_to=delivery_date_to,
            hour_ending=hour_ending,
            coast_from=coast_from,
            coast_to=coast_to,
            east_from=east_from,
            east_to=east_to,
            far_west_from=far_west_from,
            far_west_to=far_west_to,
            north_from=north_from,
            north_to=north_to,
            north_central_from=north_central_from,
            north_central_to=north_central_to,
            south_central_from=south_central_from,
            south_central_to=south_central_to,
            southern_from=southern_from,
            southern_to=southern_to,
            posted_datetime_from=posted_datetime_from,
            posted_datetime_to=posted_datetime_to,
            west_from=west_from,
            west_to=west_to,
            system_total_from=system_total_from,
            system_total_to=system_total_to,
            model=model,
            in_use_flag=in_use_flag,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
