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
    from_station: Union[Unset, str] = UNSET,
    to_station: Union[Unset, str] = UNSET,
    from_stationk_v_from: Union[Unset, float] = UNSET,
    from_stationk_v_to: Union[Unset, float] = UNSET,
    to_stationk_v_from: Union[Unset, float] = UNSET,
    to_stationk_v_to: Union[Unset, float] = UNSET,
    cct_status: Union[Unset, str] = UNSET,
    sced_timestamp_from: Union[Unset, str] = UNSET,
    sced_timestamp_to: Union[Unset, str] = UNSET,
    repeated_hour_flag: Union[Unset, bool] = UNSET,
    constraint_id_from: Union[Unset, int] = UNSET,
    constraint_id_to: Union[Unset, int] = UNSET,
    constraint_name: Union[Unset, str] = UNSET,
    contingency_name: Union[Unset, str] = UNSET,
    shadow_price_from: Union[Unset, float] = UNSET,
    shadow_price_to: Union[Unset, float] = UNSET,
    max_shadow_price_from: Union[Unset, float] = UNSET,
    max_shadow_price_to: Union[Unset, float] = UNSET,
    limit_from: Union[Unset, float] = UNSET,
    limit_to: Union[Unset, float] = UNSET,
    value_from: Union[Unset, float] = UNSET,
    value_to: Union[Unset, float] = UNSET,
    violated_mw_from: Union[Unset, float] = UNSET,
    violated_mw_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["Ocp-Apim-Subscription-Key"] = ocp_apim_subscription_key

    params: dict[str, Any] = {}

    params["fromStation"] = from_station

    params["toStation"] = to_station

    params["fromStationkVFrom"] = from_stationk_v_from

    params["fromStationkVTo"] = from_stationk_v_to

    params["toStationkVFrom"] = to_stationk_v_from

    params["toStationkVTo"] = to_stationk_v_to

    params["CCTStatus"] = cct_status

    params["SCEDTimestampFrom"] = sced_timestamp_from

    params["SCEDTimestampTo"] = sced_timestamp_to

    params["repeatedHourFlag"] = repeated_hour_flag

    params["constraintIDFrom"] = constraint_id_from

    params["constraintIDTo"] = constraint_id_to

    params["constraintName"] = constraint_name

    params["contingencyName"] = contingency_name

    params["shadowPriceFrom"] = shadow_price_from

    params["shadowPriceTo"] = shadow_price_to

    params["maxShadowPriceFrom"] = max_shadow_price_from

    params["maxShadowPriceTo"] = max_shadow_price_to

    params["limitFrom"] = limit_from

    params["limitTo"] = limit_to

    params["valueFrom"] = value_from

    params["valueTo"] = value_to

    params["violatedMWFrom"] = violated_mw_from

    params["violatedMWTo"] = violated_mw_to

    params["page"] = page

    params["size"] = size

    params["sort"] = sort

    params["dir"] = dir_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/np6-86-cd/shdw_prices_bnd_trns_const",
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
    from_station: Union[Unset, str] = UNSET,
    to_station: Union[Unset, str] = UNSET,
    from_stationk_v_from: Union[Unset, float] = UNSET,
    from_stationk_v_to: Union[Unset, float] = UNSET,
    to_stationk_v_from: Union[Unset, float] = UNSET,
    to_stationk_v_to: Union[Unset, float] = UNSET,
    cct_status: Union[Unset, str] = UNSET,
    sced_timestamp_from: Union[Unset, str] = UNSET,
    sced_timestamp_to: Union[Unset, str] = UNSET,
    repeated_hour_flag: Union[Unset, bool] = UNSET,
    constraint_id_from: Union[Unset, int] = UNSET,
    constraint_id_to: Union[Unset, int] = UNSET,
    constraint_name: Union[Unset, str] = UNSET,
    contingency_name: Union[Unset, str] = UNSET,
    shadow_price_from: Union[Unset, float] = UNSET,
    shadow_price_to: Union[Unset, float] = UNSET,
    max_shadow_price_from: Union[Unset, float] = UNSET,
    max_shadow_price_to: Union[Unset, float] = UNSET,
    limit_from: Union[Unset, float] = UNSET,
    limit_to: Union[Unset, float] = UNSET,
    value_from: Union[Unset, float] = UNSET,
    value_to: Union[Unset, float] = UNSET,
    violated_mw_from: Union[Unset, float] = UNSET,
    violated_mw_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """SCED Shadow Prices and Binding Transmission Constraints

     SCED Shadow Prices and Binding Transmission Constraints

    Args:
        from_station (Union[Unset, str]):
        to_station (Union[Unset, str]):
        from_stationk_v_from (Union[Unset, float]):
        from_stationk_v_to (Union[Unset, float]):
        to_stationk_v_from (Union[Unset, float]):
        to_stationk_v_to (Union[Unset, float]):
        cct_status (Union[Unset, str]):
        sced_timestamp_from (Union[Unset, str]):
        sced_timestamp_to (Union[Unset, str]):
        repeated_hour_flag (Union[Unset, bool]):
        constraint_id_from (Union[Unset, int]):
        constraint_id_to (Union[Unset, int]):
        constraint_name (Union[Unset, str]):
        contingency_name (Union[Unset, str]):
        shadow_price_from (Union[Unset, float]):
        shadow_price_to (Union[Unset, float]):
        max_shadow_price_from (Union[Unset, float]):
        max_shadow_price_to (Union[Unset, float]):
        limit_from (Union[Unset, float]):
        limit_to (Union[Unset, float]):
        value_from (Union[Unset, float]):
        value_to (Union[Unset, float]):
        violated_mw_from (Union[Unset, float]):
        violated_mw_to (Union[Unset, float]):
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
        from_station=from_station,
        to_station=to_station,
        from_stationk_v_from=from_stationk_v_from,
        from_stationk_v_to=from_stationk_v_to,
        to_stationk_v_from=to_stationk_v_from,
        to_stationk_v_to=to_stationk_v_to,
        cct_status=cct_status,
        sced_timestamp_from=sced_timestamp_from,
        sced_timestamp_to=sced_timestamp_to,
        repeated_hour_flag=repeated_hour_flag,
        constraint_id_from=constraint_id_from,
        constraint_id_to=constraint_id_to,
        constraint_name=constraint_name,
        contingency_name=contingency_name,
        shadow_price_from=shadow_price_from,
        shadow_price_to=shadow_price_to,
        max_shadow_price_from=max_shadow_price_from,
        max_shadow_price_to=max_shadow_price_to,
        limit_from=limit_from,
        limit_to=limit_to,
        value_from=value_from,
        value_to=value_to,
        violated_mw_from=violated_mw_from,
        violated_mw_to=violated_mw_to,
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
    from_station: Union[Unset, str] = UNSET,
    to_station: Union[Unset, str] = UNSET,
    from_stationk_v_from: Union[Unset, float] = UNSET,
    from_stationk_v_to: Union[Unset, float] = UNSET,
    to_stationk_v_from: Union[Unset, float] = UNSET,
    to_stationk_v_to: Union[Unset, float] = UNSET,
    cct_status: Union[Unset, str] = UNSET,
    sced_timestamp_from: Union[Unset, str] = UNSET,
    sced_timestamp_to: Union[Unset, str] = UNSET,
    repeated_hour_flag: Union[Unset, bool] = UNSET,
    constraint_id_from: Union[Unset, int] = UNSET,
    constraint_id_to: Union[Unset, int] = UNSET,
    constraint_name: Union[Unset, str] = UNSET,
    contingency_name: Union[Unset, str] = UNSET,
    shadow_price_from: Union[Unset, float] = UNSET,
    shadow_price_to: Union[Unset, float] = UNSET,
    max_shadow_price_from: Union[Unset, float] = UNSET,
    max_shadow_price_to: Union[Unset, float] = UNSET,
    limit_from: Union[Unset, float] = UNSET,
    limit_to: Union[Unset, float] = UNSET,
    value_from: Union[Unset, float] = UNSET,
    value_to: Union[Unset, float] = UNSET,
    violated_mw_from: Union[Unset, float] = UNSET,
    violated_mw_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """SCED Shadow Prices and Binding Transmission Constraints

     SCED Shadow Prices and Binding Transmission Constraints

    Args:
        from_station (Union[Unset, str]):
        to_station (Union[Unset, str]):
        from_stationk_v_from (Union[Unset, float]):
        from_stationk_v_to (Union[Unset, float]):
        to_stationk_v_from (Union[Unset, float]):
        to_stationk_v_to (Union[Unset, float]):
        cct_status (Union[Unset, str]):
        sced_timestamp_from (Union[Unset, str]):
        sced_timestamp_to (Union[Unset, str]):
        repeated_hour_flag (Union[Unset, bool]):
        constraint_id_from (Union[Unset, int]):
        constraint_id_to (Union[Unset, int]):
        constraint_name (Union[Unset, str]):
        contingency_name (Union[Unset, str]):
        shadow_price_from (Union[Unset, float]):
        shadow_price_to (Union[Unset, float]):
        max_shadow_price_from (Union[Unset, float]):
        max_shadow_price_to (Union[Unset, float]):
        limit_from (Union[Unset, float]):
        limit_to (Union[Unset, float]):
        value_from (Union[Unset, float]):
        value_to (Union[Unset, float]):
        violated_mw_from (Union[Unset, float]):
        violated_mw_to (Union[Unset, float]):
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
        from_station=from_station,
        to_station=to_station,
        from_stationk_v_from=from_stationk_v_from,
        from_stationk_v_to=from_stationk_v_to,
        to_stationk_v_from=to_stationk_v_from,
        to_stationk_v_to=to_stationk_v_to,
        cct_status=cct_status,
        sced_timestamp_from=sced_timestamp_from,
        sced_timestamp_to=sced_timestamp_to,
        repeated_hour_flag=repeated_hour_flag,
        constraint_id_from=constraint_id_from,
        constraint_id_to=constraint_id_to,
        constraint_name=constraint_name,
        contingency_name=contingency_name,
        shadow_price_from=shadow_price_from,
        shadow_price_to=shadow_price_to,
        max_shadow_price_from=max_shadow_price_from,
        max_shadow_price_to=max_shadow_price_to,
        limit_from=limit_from,
        limit_to=limit_to,
        value_from=value_from,
        value_to=value_to,
        violated_mw_from=violated_mw_from,
        violated_mw_to=violated_mw_to,
        page=page,
        size=size,
        sort=sort,
        dir_=dir_,
        ocp_apim_subscription_key=ocp_apim_subscription_key,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    from_station: Union[Unset, str] = UNSET,
    to_station: Union[Unset, str] = UNSET,
    from_stationk_v_from: Union[Unset, float] = UNSET,
    from_stationk_v_to: Union[Unset, float] = UNSET,
    to_stationk_v_from: Union[Unset, float] = UNSET,
    to_stationk_v_to: Union[Unset, float] = UNSET,
    cct_status: Union[Unset, str] = UNSET,
    sced_timestamp_from: Union[Unset, str] = UNSET,
    sced_timestamp_to: Union[Unset, str] = UNSET,
    repeated_hour_flag: Union[Unset, bool] = UNSET,
    constraint_id_from: Union[Unset, int] = UNSET,
    constraint_id_to: Union[Unset, int] = UNSET,
    constraint_name: Union[Unset, str] = UNSET,
    contingency_name: Union[Unset, str] = UNSET,
    shadow_price_from: Union[Unset, float] = UNSET,
    shadow_price_to: Union[Unset, float] = UNSET,
    max_shadow_price_from: Union[Unset, float] = UNSET,
    max_shadow_price_to: Union[Unset, float] = UNSET,
    limit_from: Union[Unset, float] = UNSET,
    limit_to: Union[Unset, float] = UNSET,
    value_from: Union[Unset, float] = UNSET,
    value_to: Union[Unset, float] = UNSET,
    violated_mw_from: Union[Unset, float] = UNSET,
    violated_mw_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Response[Union[Exception_, Report]]:
    """SCED Shadow Prices and Binding Transmission Constraints

     SCED Shadow Prices and Binding Transmission Constraints

    Args:
        from_station (Union[Unset, str]):
        to_station (Union[Unset, str]):
        from_stationk_v_from (Union[Unset, float]):
        from_stationk_v_to (Union[Unset, float]):
        to_stationk_v_from (Union[Unset, float]):
        to_stationk_v_to (Union[Unset, float]):
        cct_status (Union[Unset, str]):
        sced_timestamp_from (Union[Unset, str]):
        sced_timestamp_to (Union[Unset, str]):
        repeated_hour_flag (Union[Unset, bool]):
        constraint_id_from (Union[Unset, int]):
        constraint_id_to (Union[Unset, int]):
        constraint_name (Union[Unset, str]):
        contingency_name (Union[Unset, str]):
        shadow_price_from (Union[Unset, float]):
        shadow_price_to (Union[Unset, float]):
        max_shadow_price_from (Union[Unset, float]):
        max_shadow_price_to (Union[Unset, float]):
        limit_from (Union[Unset, float]):
        limit_to (Union[Unset, float]):
        value_from (Union[Unset, float]):
        value_to (Union[Unset, float]):
        violated_mw_from (Union[Unset, float]):
        violated_mw_to (Union[Unset, float]):
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
        from_station=from_station,
        to_station=to_station,
        from_stationk_v_from=from_stationk_v_from,
        from_stationk_v_to=from_stationk_v_to,
        to_stationk_v_from=to_stationk_v_from,
        to_stationk_v_to=to_stationk_v_to,
        cct_status=cct_status,
        sced_timestamp_from=sced_timestamp_from,
        sced_timestamp_to=sced_timestamp_to,
        repeated_hour_flag=repeated_hour_flag,
        constraint_id_from=constraint_id_from,
        constraint_id_to=constraint_id_to,
        constraint_name=constraint_name,
        contingency_name=contingency_name,
        shadow_price_from=shadow_price_from,
        shadow_price_to=shadow_price_to,
        max_shadow_price_from=max_shadow_price_from,
        max_shadow_price_to=max_shadow_price_to,
        limit_from=limit_from,
        limit_to=limit_to,
        value_from=value_from,
        value_to=value_to,
        violated_mw_from=violated_mw_from,
        violated_mw_to=violated_mw_to,
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
    from_station: Union[Unset, str] = UNSET,
    to_station: Union[Unset, str] = UNSET,
    from_stationk_v_from: Union[Unset, float] = UNSET,
    from_stationk_v_to: Union[Unset, float] = UNSET,
    to_stationk_v_from: Union[Unset, float] = UNSET,
    to_stationk_v_to: Union[Unset, float] = UNSET,
    cct_status: Union[Unset, str] = UNSET,
    sced_timestamp_from: Union[Unset, str] = UNSET,
    sced_timestamp_to: Union[Unset, str] = UNSET,
    repeated_hour_flag: Union[Unset, bool] = UNSET,
    constraint_id_from: Union[Unset, int] = UNSET,
    constraint_id_to: Union[Unset, int] = UNSET,
    constraint_name: Union[Unset, str] = UNSET,
    contingency_name: Union[Unset, str] = UNSET,
    shadow_price_from: Union[Unset, float] = UNSET,
    shadow_price_to: Union[Unset, float] = UNSET,
    max_shadow_price_from: Union[Unset, float] = UNSET,
    max_shadow_price_to: Union[Unset, float] = UNSET,
    limit_from: Union[Unset, float] = UNSET,
    limit_to: Union[Unset, float] = UNSET,
    value_from: Union[Unset, float] = UNSET,
    value_to: Union[Unset, float] = UNSET,
    violated_mw_from: Union[Unset, float] = UNSET,
    violated_mw_to: Union[Unset, float] = UNSET,
    page: Union[Unset, int] = UNSET,
    size: Union[Unset, int] = UNSET,
    sort: Union[Unset, str] = UNSET,
    dir_: Union[Unset, str] = UNSET,
    ocp_apim_subscription_key: str,
) -> Optional[Union[Exception_, Report]]:
    """SCED Shadow Prices and Binding Transmission Constraints

     SCED Shadow Prices and Binding Transmission Constraints

    Args:
        from_station (Union[Unset, str]):
        to_station (Union[Unset, str]):
        from_stationk_v_from (Union[Unset, float]):
        from_stationk_v_to (Union[Unset, float]):
        to_stationk_v_from (Union[Unset, float]):
        to_stationk_v_to (Union[Unset, float]):
        cct_status (Union[Unset, str]):
        sced_timestamp_from (Union[Unset, str]):
        sced_timestamp_to (Union[Unset, str]):
        repeated_hour_flag (Union[Unset, bool]):
        constraint_id_from (Union[Unset, int]):
        constraint_id_to (Union[Unset, int]):
        constraint_name (Union[Unset, str]):
        contingency_name (Union[Unset, str]):
        shadow_price_from (Union[Unset, float]):
        shadow_price_to (Union[Unset, float]):
        max_shadow_price_from (Union[Unset, float]):
        max_shadow_price_to (Union[Unset, float]):
        limit_from (Union[Unset, float]):
        limit_to (Union[Unset, float]):
        value_from (Union[Unset, float]):
        value_to (Union[Unset, float]):
        violated_mw_from (Union[Unset, float]):
        violated_mw_to (Union[Unset, float]):
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
            from_station=from_station,
            to_station=to_station,
            from_stationk_v_from=from_stationk_v_from,
            from_stationk_v_to=from_stationk_v_to,
            to_stationk_v_from=to_stationk_v_from,
            to_stationk_v_to=to_stationk_v_to,
            cct_status=cct_status,
            sced_timestamp_from=sced_timestamp_from,
            sced_timestamp_to=sced_timestamp_to,
            repeated_hour_flag=repeated_hour_flag,
            constraint_id_from=constraint_id_from,
            constraint_id_to=constraint_id_to,
            constraint_name=constraint_name,
            contingency_name=contingency_name,
            shadow_price_from=shadow_price_from,
            shadow_price_to=shadow_price_to,
            max_shadow_price_from=max_shadow_price_from,
            max_shadow_price_to=max_shadow_price_to,
            limit_from=limit_from,
            limit_to=limit_to,
            value_from=value_from,
            value_to=value_to,
            violated_mw_from=violated_mw_from,
            violated_mw_to=violated_mw_to,
            page=page,
            size=size,
            sort=sort,
            dir_=dir_,
            ocp_apim_subscription_key=ocp_apim_subscription_key,
        )
    ).parsed
