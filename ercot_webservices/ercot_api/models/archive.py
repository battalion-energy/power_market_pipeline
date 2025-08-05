import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.link import Link


T = TypeVar("T", bound="Archive")


@_attrs_define
class Archive:
    """Represents an EMIL Product archive file.

    Attributes:
        doc_id (Union[Unset, int]):
        friendly_name (Union[Unset, str]):
        post_datetime (Union[Unset, datetime.datetime]):
        links (Union[Unset, list['Link']]):
    """

    doc_id: Union[Unset, int] = UNSET
    friendly_name: Union[Unset, str] = UNSET
    post_datetime: Union[Unset, datetime.datetime] = UNSET
    links: Union[Unset, list["Link"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        doc_id = self.doc_id

        friendly_name = self.friendly_name

        post_datetime: Union[Unset, str] = UNSET
        if not isinstance(self.post_datetime, Unset):
            post_datetime = self.post_datetime.isoformat()

        links: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.links, Unset):
            links = []
            for links_item_data in self.links:
                links_item = links_item_data.to_dict()
                links.append(links_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if doc_id is not UNSET:
            field_dict["docId"] = doc_id
        if friendly_name is not UNSET:
            field_dict["friendlyName"] = friendly_name
        if post_datetime is not UNSET:
            field_dict["postDatetime"] = post_datetime
        if links is not UNSET:
            field_dict["links"] = links

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.link import Link

        d = dict(src_dict)
        doc_id = d.pop("docId", UNSET)

        friendly_name = d.pop("friendlyName", UNSET)

        _post_datetime = d.pop("postDatetime", UNSET)
        post_datetime: Union[Unset, datetime.datetime]
        if isinstance(_post_datetime, Unset):
            post_datetime = UNSET
        else:
            post_datetime = isoparse(_post_datetime)

        links = []
        _links = d.pop("links", UNSET)
        for links_item_data in _links or []:
            links_item = Link.from_dict(links_item_data)

            links.append(links_item)

        archive = cls(
            doc_id=doc_id,
            friendly_name=friendly_name,
            post_datetime=post_datetime,
            links=links,
        )

        archive.additional_properties = d
        return archive

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
