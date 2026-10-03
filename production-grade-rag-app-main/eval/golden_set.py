"""
Golden evaluation set for the Oracle WMS Cloud 26B documentation RAG app.

NOTE — why a vendor-specific file lives in an otherwise domain-agnostic engine:
this is a deliberate ("solomonic") exception. The product is a general-purpose
RAG engine, but it needs exactly one concrete golden set to measure against, and
its first use case is supply-chain research — specifically the WMS product. A
golden set is inherently corpus-specific *test data*, not engine code, so the
vendor naming is confined to this file and kept out of app/.

Every question is grounded in a REAL ingested section: the reference answer is
drawn from that section's text, and `gold_section_id` points at it. The corpus
spans two 26B books — the Online Help (`owmol`, how-to) and the WMS REST API
Guide (`owmre`, API reference) — so questions cover both how-to and API usage.

Balanced ~5/5/5 across three query types so eval scores can be sliced by type:
  - definitional   : "what is X" conceptual questions
  - configuration  : "how do I set up / configure X" procedural questions
  - api_reference  : REST API entity/operation questions
"""

from dataclasses import dataclass
from typing import Literal, Optional

RELEASE = "26b"

QueryType = Literal["definitional", "configuration", "api_reference"]


@dataclass(frozen=True)
class GoldenExample:
    id: str
    question: str
    query_type: QueryType
    reference_answer: str
    gold_section_id: str  # release/source/slug of the section that answers it
    release: str = RELEASE


GOLDEN_SET: list[GoldenExample] = [
    # ---------- definitional (owmol concepts) ----------
    GoldenExample(
        id="def-01",
        question="What is a wave in Oracle WMS Cloud?",
        query_type="definitional",
        reference_answer=(
            "A wave is run to allocate orders against allocatable inventory in "
            "the warehouse. Each wave creates an allocation and a series of tasks "
            "(such as picking tasks) for the selected orders."
        ),
        gold_section_id="26b/owmol/allocation-units-of-measurement-uom",
    ),
    GoldenExample(
        id="def-02",
        question="What is a task in Oracle WMS Cloud?",
        query_type="definitional",
        reference_answer=(
            "A task is a picking instruction generated with each allocation at the "
            "end of a wave. It gives the operator details such as the outbound LPN, "
            "the SKU, the quantity, and the picking location."
        ),
        gold_section_id="26b/owmol/task-management",
    ),
    GoldenExample(
        id="def-03",
        question="What is zone picking in Oracle WMS Cloud?",
        query_type="definitional",
        reference_answer=(
            "Zone picking is an order-picking method that divides items into "
            "multiple zones, where each employee is trained to pick within an "
            "assigned zone."
        ),
        gold_section_id="26b/owmol/executing-pick-zone-tasks-in-the-rf",
    ),
    GoldenExample(
        id="def-04",
        question="What is a cycle count in Oracle WMS Cloud?",
        query_type="definitional",
        reference_answer=(
            "A cycle count is an inventory-auditing procedure in which a small "
            "subset of inventory, in a specific location, is counted on a "
            "specified day."
        ),
        gold_section_id="26b/owmol/how-to-check-for-a-flagged-cycle-count-location",
    ),
    GoldenExample(
        id="def-05",
        question="What is cross-docking in Oracle WMS Cloud?",
        query_type="definitional",
        reference_answer=(
            "Cross-docking is the ability to cross-dock inventory during the "
            "receiving process, so received goods move outbound to fulfill orders "
            "without being stored. It supports receiving cross-dock against an "
            "existing order or an order automatically created for the LPN."
        ),
        gold_section_id="26b/owmol/cross-dock-management",
    ),

    # ---------- configuration (owmol how-to) ----------
    GoldenExample(
        id="cfg-01",
        question=(
            "How do you assign a putaway type to an inbound shipment (ASN) detail "
            "that differs from the item's default?"
        ),
        query_type="configuration",
        reference_answer=(
            "On the Inbound Shipments screen, select the shipment's ASN detail "
            "record and assign the putaway type there; this overrides the item's "
            "default putaway type from the Item Master for that ASN."
        ),
        gold_section_id="26b/owmol/assigning-putaway-types-in-inbound-shipment-detail-records",
    ),
    GoldenExample(
        id="cfg-02",
        question="How do you configure calling directed putaway with MHE?",
        query_type="configuration",
        reference_answer=(
            "In the MHE Route Instruction Configuration UI, create or select a "
            "Route Instruction rule, choose the LPN Type and LPN Status, and select "
            "Putaway from the Module drop-down."
        ),
        gold_section_id="26b/owmol/configure-calling-directed-putaway",
    ),
    GoldenExample(
        id="cfg-03",
        question=(
            "How do you configure auto-printing of outbound documents such as the "
            "Bill of Lading?"
        ),
        query_type="configuration",
        reference_answer=(
            "Auto-printing of documents (for example the Bill of Lading or "
            "Commercial Invoice) when an outbound load is shipped is set up in the "
            "Output Interface Configuration UI."
        ),
        gold_section_id="26b/owmol/configure-auto-print-for-documents-in-output-interface",
    ),
    GoldenExample(
        id="cfg-04",
        question="How do you assign a group to a user in Oracle WMS Cloud?",
        query_type="configuration",
        reference_answer=(
            "On the Users screen, select the user, click Groups, and use the "
            "Create button to add the group(s) to the user."
        ),
        gold_section_id="26b/owmol/assigning-groups-to-users",
    ),
    GoldenExample(
        id="cfg-05",
        question="How do you create a putaway type in Oracle WMS Cloud?",
        query_type="configuration",
        reference_answer=(
            "Putaway types group similar products by how they need to be stored "
            "(for example an 'ELECTRONICS' type for smartphones and tablets). "
            "Creating them is the first step in setting up system-directed putaway."
        ),
        gold_section_id="26b/owmol/creating-putaway-types",
    ),

    # ---------- api_reference (owmre REST API Guide) ----------
    GoldenExample(
        id="api-01",
        question="How do you authenticate to the WMS REST API (lgfapi)?",
        query_type="api_reference",
        reference_answer=(
            "Because each request is stateless, every request must carry "
            "authentication. lgfapi supports BasicAuth (classic username and "
            "password) and OAuth2 (token-based authorization)."
        ),
        gold_section_id="26b/owmre/login-and-authentication",
    ),
    GoldenExample(
        id="api-02",
        question="How do you create an entity resource via the WMS REST API?",
        query_type="api_reference",
        reference_answer=(
            "Send an HTTP POST request with the new resource's initial data in the "
            "request body; the requesting user must have the required permission. "
            "Only a limited set of entity resources can be created/linked this way."
        ),
        gold_section_id="26b/owmre/creating-a-resource-post",
    ),
    GoldenExample(
        id="api-03",
        question=(
            "How do you retrieve the sales orders allocated to a container via the "
            "WMS REST API?"
        ),
        query_type="api_reference",
        reference_answer=(
            "Send a GET to …/wms/lgfapi/v10/entity/container/{id}/orders/, which "
            "returns a paginated representation of the order_hdr entities for the "
            "sales orders allocated against that inbound or outbound container."
        ),
        gold_section_id="26b/owmre/get-sales-orders",
    ),
    GoldenExample(
        id="api-04",
        question="What does the Bulk Update Inventory Attributes API do?",
        query_type="api_reference",
        reference_answer=(
            "It updates the inventory attributes of one or more inventory objects "
            "(inventory in a Received or Located IBLPN, or in an Active Location). "
            "Inventory-history adjustment records are written for the changes."
        ),
        gold_section_id="26b/owmre/bulk-update-inventory-attributes",
    ),
    GoldenExample(
        id="api-05",
        question="What does the Movement Request API allow?",
        query_type="api_reference",
        reference_answer=(
            "It lets ERP and manufacturing applications order specific serial "
            "numbers against a particular movement request line, loading the "
            "movement-request stage tables in JSON format."
        ),
        gold_section_id="26b/owmre/movement-request",
    ),
]


def by_query_type(query_type: QueryType) -> list[GoldenExample]:
    return [ex for ex in GOLDEN_SET if ex.query_type == query_type]
