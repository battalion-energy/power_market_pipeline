from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.field import Field
    from ..models.link import Link
    from ..models.report_data import ReportData
    from ..models.report_metadata import ReportMetadata
    from ..models.result_metadata import ResultMetadata


T = TypeVar("T", bound="Report")


@_attrs_define
class Report:
    """Represents report data search results from any EMIL artifact report query.

    Attributes:
        field_meta (Union[Unset, ResultMetadata]): Represents record and paging summary for the specified query results.
        report (Union[Unset, ReportMetadata]): Represents general metadata for the specified report artifact, including
            the parent EMIL Product identifier.
        fields (Union[Unset, list['Field']]):
        data (Union[Unset, ReportData]):
        links (Union[Unset, list['Link']]):
    """

    field_meta: Union[Unset, "ResultMetadata"] = UNSET
    report: Union[Unset, "ReportMetadata"] = UNSET
    fields: Union[Unset, list["Field"]] = UNSET
    data: Union[Unset, "ReportData"] = UNSET
    links: Union[Unset, list["Link"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        field_meta: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.field_meta, Unset):
            field_meta = self.field_meta.to_dict()

        report: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.report, Unset):
            report = self.report.to_dict()

        fields: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.fields, Unset):
            fields = []
            for fields_item_data in self.fields:
                fields_item = fields_item_data.to_dict()
                fields.append(fields_item)

        data: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.data, Unset):
            data = self.data.to_dict()

        links: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.links, Unset):
            links = []
            for links_item_data in self.links:
                links_item = links_item_data.to_dict()
                links.append(links_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if field_meta is not UNSET:
            field_dict["_meta"] = field_meta
        if report is not UNSET:
            field_dict["report"] = report
        if fields is not UNSET:
            field_dict["fields"] = fields
        if data is not UNSET:
            field_dict["data"] = data
        if links is not UNSET:
            field_dict["links"] = links

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.field import Field
        from ..models.link import Link
        from ..models.report_data import ReportData
        from ..models.report_metadata import ReportMetadata
        from ..models.result_metadata import ResultMetadata

        d = dict(src_dict)
        _field_meta = d.pop("_meta", UNSET)
        field_meta: Union[Unset, ResultMetadata]
        if isinstance(_field_meta, Unset):
            field_meta = UNSET
        else:
            field_meta = ResultMetadata.from_dict(_field_meta)

        _report = d.pop("report", UNSET)
        report: Union[Unset, ReportMetadata]
        if isinstance(_report, Unset):
            report = UNSET
        else:
            report = ReportMetadata.from_dict(_report)

        fields = []
        _fields = d.pop("fields", UNSET)
        for fields_item_data in _fields or []:
            fields_item = Field.from_dict(fields_item_data)

            fields.append(fields_item)

        _data = d.pop("data", UNSET)
        data: Union[Unset, ReportData]
        if isinstance(_data, Unset):
            data = UNSET
        else:
            data = ReportData.from_dict(_data)

        links = []
        _links = d.pop("links", UNSET)
        for links_item_data in _links or []:
            links_item = Link.from_dict(links_item_data)

            links.append(links_item)

        report = cls(
            field_meta=field_meta,
            report=report,
            fields=fields,
            data=data,
            links=links,
        )

        report.additional_properties = d
        return report

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
