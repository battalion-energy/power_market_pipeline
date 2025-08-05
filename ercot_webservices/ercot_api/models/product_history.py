from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.archive import Archive
    from ..models.link import Link
    from ..models.product_history_metadata import ProductHistoryMetadata
    from ..models.result_metadata import ResultMetadata


T = TypeVar("T", bound="ProductHistory")


@_attrs_define
class ProductHistory:
    """Each archive represents a single EMIL Product archive (zip file).

    Attributes:
        field_meta (Union[Unset, ResultMetadata]): Represents record and paging summary for the specified query results.
        product (Union[Unset, ProductHistoryMetadata]): Represents record and paging summary for the specified query
            results.
        archives (Union[Unset, list['Archive']]):
        bundles (Union[Unset, list['Archive']]):
        links (Union[Unset, list['Link']]):
    """

    field_meta: Union[Unset, "ResultMetadata"] = UNSET
    product: Union[Unset, "ProductHistoryMetadata"] = UNSET
    archives: Union[Unset, list["Archive"]] = UNSET
    bundles: Union[Unset, list["Archive"]] = UNSET
    links: Union[Unset, list["Link"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        field_meta: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.field_meta, Unset):
            field_meta = self.field_meta.to_dict()

        product: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.product, Unset):
            product = self.product.to_dict()

        archives: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.archives, Unset):
            archives = []
            for archives_item_data in self.archives:
                archives_item = archives_item_data.to_dict()
                archives.append(archives_item)

        bundles: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.bundles, Unset):
            bundles = []
            for bundles_item_data in self.bundles:
                bundles_item = bundles_item_data.to_dict()
                bundles.append(bundles_item)

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
        if product is not UNSET:
            field_dict["product"] = product
        if archives is not UNSET:
            field_dict["archives"] = archives
        if bundles is not UNSET:
            field_dict["bundles"] = bundles
        if links is not UNSET:
            field_dict["links"] = links

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.archive import Archive
        from ..models.link import Link
        from ..models.product_history_metadata import ProductHistoryMetadata
        from ..models.result_metadata import ResultMetadata

        d = dict(src_dict)
        _field_meta = d.pop("_meta", UNSET)
        field_meta: Union[Unset, ResultMetadata]
        if isinstance(_field_meta, Unset):
            field_meta = UNSET
        else:
            field_meta = ResultMetadata.from_dict(_field_meta)

        _product = d.pop("product", UNSET)
        product: Union[Unset, ProductHistoryMetadata]
        if isinstance(_product, Unset):
            product = UNSET
        else:
            product = ProductHistoryMetadata.from_dict(_product)

        archives = []
        _archives = d.pop("archives", UNSET)
        for archives_item_data in _archives or []:
            archives_item = Archive.from_dict(archives_item_data)

            archives.append(archives_item)

        bundles = []
        _bundles = d.pop("bundles", UNSET)
        for bundles_item_data in _bundles or []:
            bundles_item = Archive.from_dict(bundles_item_data)

            bundles.append(bundles_item)

        links = []
        _links = d.pop("links", UNSET)
        for links_item_data in _links or []:
            links_item = Link.from_dict(links_item_data)

            links.append(links_item)

        product_history = cls(
            field_meta=field_meta,
            product=product,
            archives=archives,
            bundles=bundles,
            links=links,
        )

        product_history.additional_properties = d
        return product_history

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
