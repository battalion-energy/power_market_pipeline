from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.query_metadata import QueryMetadata


T = TypeVar("T", bound="ProductHistoryMetadata")


@_attrs_define
class ProductHistoryMetadata:
    """Represents record and paging summary for the specified query results.

    Attributes:
        total_records (Union[Unset, int]):
        page_size (Union[Unset, int]):
        total_pages (Union[Unset, int]):
        current_page (Union[Unset, int]):
        query (Union[Unset, QueryMetadata]): Represents query parameter summary for the specified query results.
    """

    total_records: Union[Unset, int] = UNSET
    page_size: Union[Unset, int] = UNSET
    total_pages: Union[Unset, int] = UNSET
    current_page: Union[Unset, int] = UNSET
    query: Union[Unset, "QueryMetadata"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        total_records = self.total_records

        page_size = self.page_size

        total_pages = self.total_pages

        current_page = self.current_page

        query: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.query, Unset):
            query = self.query.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if total_records is not UNSET:
            field_dict["totalRecords"] = total_records
        if page_size is not UNSET:
            field_dict["pageSize"] = page_size
        if total_pages is not UNSET:
            field_dict["totalPages"] = total_pages
        if current_page is not UNSET:
            field_dict["currentPage"] = current_page
        if query is not UNSET:
            field_dict["query"] = query

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.query_metadata import QueryMetadata

        d = dict(src_dict)
        total_records = d.pop("totalRecords", UNSET)

        page_size = d.pop("pageSize", UNSET)

        total_pages = d.pop("totalPages", UNSET)

        current_page = d.pop("currentPage", UNSET)

        _query = d.pop("query", UNSET)
        query: Union[Unset, QueryMetadata]
        if isinstance(_query, Unset):
            query = UNSET
        else:
            query = QueryMetadata.from_dict(_query)

        product_history_metadata = cls(
            total_records=total_records,
            page_size=page_size,
            total_pages=total_pages,
            current_page=current_page,
            query=query,
        )

        product_history_metadata.additional_properties = d
        return product_history_metadata

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
