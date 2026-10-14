"""Per-category repair instructions shown to the model after a rejection."""

from arc_agi_3.contracts.generation import ErrorCategory

CATEGORY_INSTRUCTIONS: dict[ErrorCategory, str] = {
    "no_json_found": (
        "Return ONLY one valid JSON object matching the schema below. Do not "
        "include analysis, markdown, or prose."
    ),
    "malformed_json": (
        "A JSON object was started but did not parse. Close every brace/quote "
        "and remove trailing commas or comments."
    ),
    "schema_validation": (
        "One or more fields have the wrong type, are missing, or are not "
        "permitted by the schema. Repair only the structured object."
    ),
    "root_type_confusion": (
        "The root object must be exactly {target_type}, not one of its "
        "nested field types (an action, experiment, tool request, or "
        "hypothesis belongs only inside its own named field, never at the "
        "top level). Every required top-level field named in ERROR_DETAIL "
        "must be present. Return one complete {target_type} object, not a "
        "fragment of one."
    ),
    "logical_contract": (
        "The object is valid JSON but violates a cross-field rule (see "
        "ERROR_DETAIL). Adjust only the fields the rule names."
    ),
    "tool_contract_violation": (
        "A tool_requests entry does not match that tool's argument schema "
        "(see ERROR_DETAIL for the exact tool and fields; the schema is in "
        "TOOL_CONTRACTS). Repair only that entry's arguments."
    ),
    "belief_contract_violation": (
        "A hypothesis_proposals or hypothesis_updates entry conflicts with "
        "the current HYPOTHESIS_LEDGER (see ERROR_DETAIL). A proposal must "
        "use a fresh ID at version 1; an update must target an existing ID "
        "with its current expected_version. Move the entry to the other "
        "list or correct its ID/version - do not change your underlying "
        "claim or confidence."
    ),
    "unknown_backend_failure": (
        "The previous generation failed before producing output. Return a "
        "minimal valid object, matching the schema below, consistent with "
        "your prior intent."
    ),
}
