import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.exception_data import ExceptionData


T = TypeVar("T", bound="Exception_")


@_attrs_define
class Exception_:
    """Represents any exception encountered while access API endpoints.

    Attributes:
        timestamp (Union[Unset, datetime.datetime]):
        code (Union[Unset, int]):
        status (Union[Unset, str]):
        message (Union[Unset, str]):
        data (Union[Unset, ExceptionData]):
    """

    timestamp: Union[Unset, datetime.datetime] = UNSET
    code: Union[Unset, int] = UNSET
    status: Union[Unset, str] = UNSET
    message: Union[Unset, str] = UNSET
    data: Union[Unset, "ExceptionData"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        timestamp: Union[Unset, str] = UNSET
        if not isinstance(self.timestamp, Unset):
            timestamp = self.timestamp.isoformat()

        code = self.code

        status = self.status

        message = self.message

        data: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.data, Unset):
            data = self.data.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if timestamp is not UNSET:
            field_dict["timestamp"] = timestamp
        if code is not UNSET:
            field_dict["code"] = code
        if status is not UNSET:
            field_dict["status"] = status
        if message is not UNSET:
            field_dict["message"] = message
        if data is not UNSET:
            field_dict["data"] = data

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.exception_data import ExceptionData

        d = dict(src_dict)
        _timestamp = d.pop("timestamp", UNSET)
        timestamp: Union[Unset, datetime.datetime]
        if isinstance(_timestamp, Unset):
            timestamp = UNSET
        else:
            timestamp = isoparse(_timestamp)

        code = d.pop("code", UNSET)

        status = d.pop("status", UNSET)

        message = d.pop("message", UNSET)

        _data = d.pop("data", UNSET)
        data: Union[Unset, ExceptionData]
        if isinstance(_data, Unset):
            data = UNSET
        else:
            data = ExceptionData.from_dict(_data)

        exception = cls(
            timestamp=timestamp,
            code=code,
            status=status,
            message=message,
            data=data,
        )

        exception.additional_properties = d
        return exception

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
