from __future__ import annotations

from enum import Enum
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, model_validator


# -----------------------------
# Core node taxonomy
# -----------------------------

class EntityType(str, Enum):
    COMPANY = "company"
    PROJECT = "project"
    FACILITY = "facility"
    COUNTRY = "country"
    REGION = "region"
    COMMODITY = "commodity"
    PRODUCT = "product"
    POLICY = "policy"
    REGULATION = "regulation"
    EVENT = "event"


class CompanyRole(str, Enum):
    MINER = "miner"
    REFINER = "refiner"
    SEPARATOR = "separator"
    TRADER = "trader"
    MAGNET_MANUFACTURER = "magnet_manufacturer"
    OEM = "oem"
    RECYCLER = "recycler"
    LOGISTICS_PROVIDER = "logistics_provider"
    UNKNOWN = "unknown"


class FacilityRole(str, Enum):
    MINE = "mine"
    CONCENTRATOR = "concentrator"
    SEPARATOR = "separator"
    REFINERY = "refinery"
    SMELTER = "smelter"
    MAGNET_PLANT = "magnet_plant"
    RECYCLING_PLANT = "recycling_plant"
    PORT = "port"
    WAREHOUSE = "warehouse"
    UNKNOWN = "unknown"


class CommodityForm(str, Enum):
    ORE = "ore"
    CONCENTRATE = "concentrate"
    OXIDE = "oxide"
    METAL = "metal"
    ALLOY = "alloy"
    MAGNET = "magnet"
    UNKNOWN = "unknown"


class EventType(str, Enum):
    EXPORT_RESTRICTION = "export_restriction"
    IMPORT_RESTRICTION = "import_restriction"
    SANCTION = "sanction"
    TARIFF_CHANGE = "tariff_change"
    MINE_CLOSURE = "mine_closure"
    FACILITY_OUTAGE = "facility_outage"
    PRODUCTION_CUT = "production_cut"
    SHIPPING_DELAY = "shipping_delay"
    STRIKE = "strike"
    ACCIDENT = "accident"
    ENVIRONMENTAL_ACTION = "environmental_action"
    PERMIT_CHANGE = "permit_change"
    PRICE_SHOCK = "price_shock"
    DEMAND_SHOCK = "demand_shock"
    ACQUISITION = "acquisition"
    INVESTMENT = "investment"
    UNKNOWN = "unknown"


# -----------------------------
# Core relation taxonomy
# -----------------------------

class RelationType(str, Enum):
    LOCATED_IN = "located_in"
    OWNS = "owns"
    OPERATES = "operates"
    DEVELOPS = "develops"
    PRODUCES = "produces"
    PROCESSES = "processes"
    REFINES = "refines"
    SHIPS_VIA = "ships_via"
    EXPORTS_TO = "exports_to"
    IMPORTS_FROM = "imports_from"
    SUPPLIES = "supplies"
    SELLS_TO = "sells_to"
    DEPENDS_ON = "depends_on"
    USES_INPUT = "uses_input"
    HAS_UPSTREAM_EXPOSURE_TO = "has_upstream_exposure_to"
    HAS_DOWNSTREAM_EXPOSURE_TO = "has_downstream_exposure_to"
    REGULATES = "regulates"
    RESTRICTS = "restricts"
    AFFECTS = "affects"
    DISRUPTS = "disrupts"
    INVOLVES_COMMODITY = "involves_commodity"
    OCCURS_AT = "occurs_at"
    OCCURS_IN = "occurs_in"


# -----------------------------
# Base schemas for extraction
# -----------------------------

class Provenance(BaseModel):
    source_id: str = Field(..., description="Document or record identifier")
    source_type: Literal["news", "filing", "trade_data", "policy", "market_data", "web"]
    source_url: Optional[str] = None
    snippet: Optional[str] = Field(default=None, description="Short supporting span")
    confidence: float = Field(ge=0.0, le=1.0, default=0.75)
    observed_at: Optional[str] = Field(default=None, description="When the system observed this evidence")


class TemporalSpan(BaseModel):
    valid_from: Optional[str] = Field(default=None, description="When the fact/event became true")
    valid_to: Optional[str] = Field(default=None, description="When the fact/event stopped being true")
    event_date: Optional[str] = Field(default=None, description="Primary event date if applicable")


class Entity(BaseModel):
    entity_id: str
    canonical_name: str
    entity_type: EntityType
    aliases: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    country_code: Optional[str] = None

    # Type-specific optional fields
    company_role: Optional[CompanyRole] = None
    facility_role: Optional[FacilityRole] = None
    commodity_form: Optional[CommodityForm] = None
    event_type: Optional[EventType] = None

    provenance: List[Provenance] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_type_specific_fields(self) -> "Entity":
        if self.entity_type == EntityType.EVENT and self.event_type is None:
            raise ValueError("event_type is required when entity_type='event'")
        return self


class Relation(BaseModel):
    relation_id: str
    subject_id: str
    relation_type: RelationType
    object_id: str
    temporal: TemporalSpan = Field(default_factory=TemporalSpan)
    provenance: List[Provenance] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.75)


class ExtractionRecord(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)


# -----------------------------
# Allowed relation signatures
# Used to constrain LLM extraction
# -----------------------------

ALLOWED_RELATION_SIGNATURES = {
    RelationType.LOCATED_IN: {
        EntityType.COMPANY,
        EntityType.PROJECT,
        EntityType.FACILITY,
    },
    RelationType.OWNS: {EntityType.COMPANY},
    RelationType.OPERATES: {EntityType.COMPANY},
    RelationType.DEVELOPS: {EntityType.COMPANY},
    RelationType.PRODUCES: {
        EntityType.COMPANY,
        EntityType.FACILITY,
        EntityType.PROJECT,
    },
    RelationType.PROCESSES: {
        EntityType.COMPANY,
        EntityType.FACILITY,
    },
    RelationType.REFINES: {EntityType.COMPANY, EntityType.FACILITY},
    RelationType.SUPPLIES: {
        EntityType.COMPANY,
        EntityType.FACILITY,
    },
    RelationType.SELLS_TO: {EntityType.COMPANY},
    RelationType.DEPENDS_ON: {EntityType.COMPANY, EntityType.FACILITY, EntityType.PROJECT},
    RelationType.USES_INPUT: {EntityType.COMPANY, EntityType.FACILITY},
    RelationType.EXPORTS_TO: {EntityType.COUNTRY, EntityType.COMPANY},
    RelationType.IMPORTS_FROM: {EntityType.COUNTRY, EntityType.COMPANY},
    RelationType.SHIPS_VIA: {EntityType.COMPANY, EntityType.FACILITY},
    RelationType.REGULATES: {EntityType.POLICY, EntityType.REGULATION},
    RelationType.RESTRICTS: {EntityType.POLICY, EntityType.REGULATION, EntityType.EVENT},
    RelationType.AFFECTS: {EntityType.EVENT, EntityType.POLICY, EntityType.REGULATION},
    RelationType.DISRUPTS: {EntityType.EVENT},
    RelationType.INVOLVES_COMMODITY: {EntityType.EVENT, EntityType.POLICY, EntityType.REGULATION},
    RelationType.OCCURS_AT: {EntityType.EVENT},
    RelationType.OCCURS_IN: {EntityType.EVENT},
    RelationType.HAS_UPSTREAM_EXPOSURE_TO: {EntityType.COMPANY},
    RelationType.HAS_DOWNSTREAM_EXPOSURE_TO: {EntityType.COMPANY},
}


ALLOWED_OBJECT_TYPES = {
    RelationType.LOCATED_IN: {EntityType.COUNTRY, EntityType.REGION},
    RelationType.OWNS: {EntityType.COMPANY, EntityType.PROJECT, EntityType.FACILITY},
    RelationType.OPERATES: {EntityType.FACILITY},
    RelationType.DEVELOPS: {EntityType.PROJECT, EntityType.FACILITY},
    RelationType.PRODUCES: {EntityType.COMMODITY, EntityType.PRODUCT},
    RelationType.PROCESSES: {EntityType.COMMODITY, EntityType.PRODUCT},
    RelationType.REFINES: {EntityType.COMMODITY, EntityType.PRODUCT},
    RelationType.SUPPLIES: {EntityType.COMPANY, EntityType.FACILITY},
    RelationType.SELLS_TO: {EntityType.COMPANY},
    RelationType.DEPENDS_ON: {EntityType.COMPANY, EntityType.FACILITY, EntityType.COUNTRY, EntityType.COMMODITY, EntityType.PRODUCT},
    RelationType.USES_INPUT: {EntityType.COMMODITY, EntityType.PRODUCT},
    RelationType.EXPORTS_TO: {EntityType.COUNTRY, EntityType.COMPANY},
    RelationType.IMPORTS_FROM: {EntityType.COUNTRY, EntityType.COMPANY},
    RelationType.SHIPS_VIA: {EntityType.FACILITY, EntityType.REGION, EntityType.COUNTRY},
    RelationType.REGULATES: {EntityType.COMPANY, EntityType.COMMODITY, EntityType.PROJECT, EntityType.COUNTRY, EntityType.FACILITY},
    RelationType.RESTRICTS: {EntityType.COMMODITY, EntityType.COUNTRY, EntityType.COMPANY, EntityType.PROJECT, EntityType.FACILITY},
    RelationType.AFFECTS: {EntityType.COMPANY, EntityType.FACILITY, EntityType.COUNTRY, EntityType.COMMODITY, EntityType.PROJECT, EntityType.PRODUCT},
    RelationType.DISRUPTS: {EntityType.COMPANY, EntityType.FACILITY, EntityType.PROJECT, EntityType.COMMODITY, EntityType.PRODUCT},
    RelationType.INVOLVES_COMMODITY: {EntityType.COMMODITY, EntityType.PRODUCT},
    RelationType.OCCURS_AT: {EntityType.FACILITY, EntityType.PROJECT},
    RelationType.OCCURS_IN: {EntityType.COUNTRY, EntityType.REGION},
    RelationType.HAS_UPSTREAM_EXPOSURE_TO: {EntityType.COMPANY, EntityType.FACILITY, EntityType.COUNTRY, EntityType.COMMODITY, EntityType.PRODUCT},
    RelationType.HAS_DOWNSTREAM_EXPOSURE_TO: {EntityType.COMPANY, EntityType.FACILITY, EntityType.COUNTRY, EntityType.COMMODITY, EntityType.PRODUCT},
}


def relation_signature_is_valid(subject_type: EntityType, relation_type: RelationType, object_type: EntityType) -> bool:
    return (
        subject_type in ALLOWED_RELATION_SIGNATURES.get(relation_type, set())
        and object_type in ALLOWED_OBJECT_TYPES.get(relation_type, set())
    )


# -----------------------------
# Example extraction payload
# -----------------------------

EXAMPLE = ExtractionRecord(
    entities=[
        Entity(
            entity_id="company_lynas",
            canonical_name="Lynas Rare Earths",
            entity_type=EntityType.COMPANY,
            aliases=["Lynas"],
        ),
        Entity(
            entity_id="country_myanmar",
            canonical_name="Myanmar",
            entity_type=EntityType.COUNTRY,
        ),
        Entity(
            entity_id="commodity_ndpr_oxide",
            canonical_name="NdPr oxide",
            entity_type=EntityType.COMMODITY,
            commodity_form=CommodityForm.OXIDE,
        ),
        Entity(
            entity_id="event_export_restriction_2026_04",
            canonical_name="Myanmar rare-earth export restriction",
            entity_type=EntityType.EVENT,
            event_type=EventType.EXPORT_RESTRICTION,
        ),
    ],
    relations=[
        Relation(
            relation_id="r1",
            subject_id="event_export_restriction_2026_04",
            relation_type=RelationType.OCCURS_IN,
            object_id="country_myanmar",
            temporal=TemporalSpan(event_date="2026-04-30"),
            confidence=0.93,
        ),
        Relation(
            relation_id="r2",
            subject_id="event_export_restriction_2026_04",
            relation_type=RelationType.INVOLVES_COMMODITY,
            object_id="commodity_ndpr_oxide",
            temporal=TemporalSpan(event_date="2026-04-30"),
            confidence=0.88,
        ),
    ],
)


if __name__ == "__main__":
    print(EXAMPLE.model_dump_json(indent=2))
