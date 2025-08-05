import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.product_content_type import ProductContentType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.artifact import Artifact
    from ..models.link import Link
    from ..models.product_protocol_rules import ProductProtocolRules


T = TypeVar("T", bound="Product")


@_attrs_define
class Product:
    """Represents an EMIL Product, which includes its metadata along with all Artifact information.

    Attributes:
        emil_id (Union[Unset, str]):
        name (Union[Unset, str]):
        description (Union[Unset, str]):
        status (Union[Unset, str]):
        report_type_id (Union[Unset, int]):
        audience (Union[Unset, str]):
        generation_frequency (Union[Unset, str]):
        security_classification (Union[Unset, str]):
        last_updated (Union[Unset, datetime.datetime]):
        first_run (Union[Unset, datetime.datetime]):
        eceii (Union[Unset, str]):
        channel (Union[Unset, str]):
        user_guide (Union[Unset, str]):
        posting_type (Union[Unset, str]):
        market (Union[Unset, str]):
        extract_subscriber (Union[Unset, str]):
        xsd_name (Union[Unset, str]):
        mis_posting_location (Union[Unset, str]):
        certificate_role (Union[Unset, str]):
        file_type (Union[Unset, str]):
        ddl_name (Union[Unset, str]):
        mis_display_duration (Union[Unset, int]):
        archive_duration (Union[Unset, int]):
        notification_type (Union[Unset, str]):
        content_type (Union[Unset, ProductContentType]):
        download_limit (Union[Unset, int]):
        last_post_datetime (Union[Unset, datetime.datetime]):
        bundle (Union[Unset, int]):
        protocol_rules (Union[Unset, ProductProtocolRules]):
        links (Union[Unset, list['Link']]):
        artifacts (Union[Unset, list['Artifact']]):
    """

    emil_id: Union[Unset, str] = UNSET
    name: Union[Unset, str] = UNSET
    description: Union[Unset, str] = UNSET
    status: Union[Unset, str] = UNSET
    report_type_id: Union[Unset, int] = UNSET
    audience: Union[Unset, str] = UNSET
    generation_frequency: Union[Unset, str] = UNSET
    security_classification: Union[Unset, str] = UNSET
    last_updated: Union[Unset, datetime.datetime] = UNSET
    first_run: Union[Unset, datetime.datetime] = UNSET
    eceii: Union[Unset, str] = UNSET
    channel: Union[Unset, str] = UNSET
    user_guide: Union[Unset, str] = UNSET
    posting_type: Union[Unset, str] = UNSET
    market: Union[Unset, str] = UNSET
    extract_subscriber: Union[Unset, str] = UNSET
    xsd_name: Union[Unset, str] = UNSET
    mis_posting_location: Union[Unset, str] = UNSET
    certificate_role: Union[Unset, str] = UNSET
    file_type: Union[Unset, str] = UNSET
    ddl_name: Union[Unset, str] = UNSET
    mis_display_duration: Union[Unset, int] = UNSET
    archive_duration: Union[Unset, int] = UNSET
    notification_type: Union[Unset, str] = UNSET
    content_type: Union[Unset, ProductContentType] = UNSET
    download_limit: Union[Unset, int] = UNSET
    last_post_datetime: Union[Unset, datetime.datetime] = UNSET
    bundle: Union[Unset, int] = UNSET
    protocol_rules: Union[Unset, "ProductProtocolRules"] = UNSET
    links: Union[Unset, list["Link"]] = UNSET
    artifacts: Union[Unset, list["Artifact"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        emil_id = self.emil_id

        name = self.name

        description = self.description

        status = self.status

        report_type_id = self.report_type_id

        audience = self.audience

        generation_frequency = self.generation_frequency

        security_classification = self.security_classification

        last_updated: Union[Unset, str] = UNSET
        if not isinstance(self.last_updated, Unset):
            last_updated = self.last_updated.isoformat()

        first_run: Union[Unset, str] = UNSET
        if not isinstance(self.first_run, Unset):
            first_run = self.first_run.isoformat()

        eceii = self.eceii

        channel = self.channel

        user_guide = self.user_guide

        posting_type = self.posting_type

        market = self.market

        extract_subscriber = self.extract_subscriber

        xsd_name = self.xsd_name

        mis_posting_location = self.mis_posting_location

        certificate_role = self.certificate_role

        file_type = self.file_type

        ddl_name = self.ddl_name

        mis_display_duration = self.mis_display_duration

        archive_duration = self.archive_duration

        notification_type = self.notification_type

        content_type: Union[Unset, str] = UNSET
        if not isinstance(self.content_type, Unset):
            content_type = self.content_type.value

        download_limit = self.download_limit

        last_post_datetime: Union[Unset, str] = UNSET
        if not isinstance(self.last_post_datetime, Unset):
            last_post_datetime = self.last_post_datetime.isoformat()

        bundle = self.bundle

        protocol_rules: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.protocol_rules, Unset):
            protocol_rules = self.protocol_rules.to_dict()

        links: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.links, Unset):
            links = []
            for links_item_data in self.links:
                links_item = links_item_data.to_dict()
                links.append(links_item)

        artifacts: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.artifacts, Unset):
            artifacts = []
            for artifacts_item_data in self.artifacts:
                artifacts_item = artifacts_item_data.to_dict()
                artifacts.append(artifacts_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if emil_id is not UNSET:
            field_dict["emilId"] = emil_id
        if name is not UNSET:
            field_dict["name"] = name
        if description is not UNSET:
            field_dict["description"] = description
        if status is not UNSET:
            field_dict["status"] = status
        if report_type_id is not UNSET:
            field_dict["reportTypeId"] = report_type_id
        if audience is not UNSET:
            field_dict["audience"] = audience
        if generation_frequency is not UNSET:
            field_dict["generationFrequency"] = generation_frequency
        if security_classification is not UNSET:
            field_dict["securityClassification"] = security_classification
        if last_updated is not UNSET:
            field_dict["lastUpdated"] = last_updated
        if first_run is not UNSET:
            field_dict["firstRun"] = first_run
        if eceii is not UNSET:
            field_dict["eceii"] = eceii
        if channel is not UNSET:
            field_dict["channel"] = channel
        if user_guide is not UNSET:
            field_dict["userGuide"] = user_guide
        if posting_type is not UNSET:
            field_dict["postingType"] = posting_type
        if market is not UNSET:
            field_dict["market"] = market
        if extract_subscriber is not UNSET:
            field_dict["extractSubscriber"] = extract_subscriber
        if xsd_name is not UNSET:
            field_dict["xsdName"] = xsd_name
        if mis_posting_location is not UNSET:
            field_dict["misPostingLocation"] = mis_posting_location
        if certificate_role is not UNSET:
            field_dict["certificateRole"] = certificate_role
        if file_type is not UNSET:
            field_dict["fileType"] = file_type
        if ddl_name is not UNSET:
            field_dict["ddlName"] = ddl_name
        if mis_display_duration is not UNSET:
            field_dict["misDisplayDuration"] = mis_display_duration
        if archive_duration is not UNSET:
            field_dict["archiveDuration"] = archive_duration
        if notification_type is not UNSET:
            field_dict["notificationType"] = notification_type
        if content_type is not UNSET:
            field_dict["contentType"] = content_type
        if download_limit is not UNSET:
            field_dict["downloadLimit"] = download_limit
        if last_post_datetime is not UNSET:
            field_dict["lastPostDatetime"] = last_post_datetime
        if bundle is not UNSET:
            field_dict["bundle"] = bundle
        if protocol_rules is not UNSET:
            field_dict["protocolRules"] = protocol_rules
        if links is not UNSET:
            field_dict["links"] = links
        if artifacts is not UNSET:
            field_dict["artifacts"] = artifacts

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.artifact import Artifact
        from ..models.link import Link
        from ..models.product_protocol_rules import ProductProtocolRules

        d = dict(src_dict)
        emil_id = d.pop("emilId", UNSET)

        name = d.pop("name", UNSET)

        description = d.pop("description", UNSET)

        status = d.pop("status", UNSET)

        report_type_id = d.pop("reportTypeId", UNSET)

        audience = d.pop("audience", UNSET)

        generation_frequency = d.pop("generationFrequency", UNSET)

        security_classification = d.pop("securityClassification", UNSET)

        _last_updated = d.pop("lastUpdated", UNSET)
        last_updated: Union[Unset, datetime.datetime]
        if isinstance(_last_updated, Unset):
            last_updated = UNSET
        else:
            last_updated = isoparse(_last_updated)

        _first_run = d.pop("firstRun", UNSET)
        first_run: Union[Unset, datetime.datetime]
        if isinstance(_first_run, Unset):
            first_run = UNSET
        else:
            first_run = isoparse(_first_run)

        eceii = d.pop("eceii", UNSET)

        channel = d.pop("channel", UNSET)

        user_guide = d.pop("userGuide", UNSET)

        posting_type = d.pop("postingType", UNSET)

        market = d.pop("market", UNSET)

        extract_subscriber = d.pop("extractSubscriber", UNSET)

        xsd_name = d.pop("xsdName", UNSET)

        mis_posting_location = d.pop("misPostingLocation", UNSET)

        certificate_role = d.pop("certificateRole", UNSET)

        file_type = d.pop("fileType", UNSET)

        ddl_name = d.pop("ddlName", UNSET)

        mis_display_duration = d.pop("misDisplayDuration", UNSET)

        archive_duration = d.pop("archiveDuration", UNSET)

        notification_type = d.pop("notificationType", UNSET)

        _content_type = d.pop("contentType", UNSET)
        content_type: Union[Unset, ProductContentType]
        if isinstance(_content_type, Unset):
            content_type = UNSET
        else:
            content_type = ProductContentType(_content_type)

        download_limit = d.pop("downloadLimit", UNSET)

        _last_post_datetime = d.pop("lastPostDatetime", UNSET)
        last_post_datetime: Union[Unset, datetime.datetime]
        if isinstance(_last_post_datetime, Unset):
            last_post_datetime = UNSET
        else:
            last_post_datetime = isoparse(_last_post_datetime)

        bundle = d.pop("bundle", UNSET)

        _protocol_rules = d.pop("protocolRules", UNSET)
        protocol_rules: Union[Unset, ProductProtocolRules]
        if isinstance(_protocol_rules, Unset):
            protocol_rules = UNSET
        else:
            protocol_rules = ProductProtocolRules.from_dict(_protocol_rules)

        links = []
        _links = d.pop("links", UNSET)
        for links_item_data in _links or []:
            links_item = Link.from_dict(links_item_data)

            links.append(links_item)

        artifacts = []
        _artifacts = d.pop("artifacts", UNSET)
        for artifacts_item_data in _artifacts or []:
            artifacts_item = Artifact.from_dict(artifacts_item_data)

            artifacts.append(artifacts_item)

        product = cls(
            emil_id=emil_id,
            name=name,
            description=description,
            status=status,
            report_type_id=report_type_id,
            audience=audience,
            generation_frequency=generation_frequency,
            security_classification=security_classification,
            last_updated=last_updated,
            first_run=first_run,
            eceii=eceii,
            channel=channel,
            user_guide=user_guide,
            posting_type=posting_type,
            market=market,
            extract_subscriber=extract_subscriber,
            xsd_name=xsd_name,
            mis_posting_location=mis_posting_location,
            certificate_role=certificate_role,
            file_type=file_type,
            ddl_name=ddl_name,
            mis_display_duration=mis_display_duration,
            archive_duration=archive_duration,
            notification_type=notification_type,
            content_type=content_type,
            download_limit=download_limit,
            last_post_datetime=last_post_datetime,
            bundle=bundle,
            protocol_rules=protocol_rules,
            links=links,
            artifacts=artifacts,
        )

        product.additional_properties = d
        return product

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
