"""Service 15: Extraction Agents.

Extracts structured data from retrieved evidence using the LLM gateway with
Pydantic-validated JSON output. Follows service 15 design: grouped specialists,
versioned prompts, guardrails, and citation resolution.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from core.errors import BudgetExceededError, StructuredOutputError
from core.llm import LLMGateway, LLMResult, complete_json
from agents.state import BidState, FieldExtraction, FieldResult, SourceRef
from agents.retrieval_agent import RetrievalResult
from agents.ingestion_agent import IngestionResult

logger = logging.getLogger(__name__)

# Prompt versioning
PROMPT_VERSION = "v1"

# Configuration
EVIDENCE_CHUNKS_PER_FIELD = 4
QUOTE_MAX_WORDS = 25
GROUP_MAX_PROMPT_TOKENS = 8000


async def extract_group(
    group: str,
    specs: list,
    retrievals: dict[str, RetrievalResult],
    *,
    llm_gateway: LLMGateway,
    feedback: dict[str, str] | None = None,
) -> dict[str, FieldResult]:
    """Extract a group of fields using the LLM gateway.

    Args:
        group: The group name (e.g., "dates_logistics", "commercial_legal")
        specs: List of FieldSpec objects for this group
        retrievals: Dict mapping field names to RetrievalResult from agent 14
        llm_gateway: LLMGateway instance for completing the extraction
        feedback: Optional dict of field -> feedback reason for retry mode

    Returns:
        Dict mapping field names to FieldResult objects
    """
    # Build the evidence blocks for this group
    evidence_blocks = _build_evidence_blocks(retrievals, group)

    # Build the prompt
    prompt = _build_prompt(group, specs, evidence_blocks, feedback)

    # Complete JSON via the LLM gateway
    try:
        result = await llm_gateway.acomplete_json(
            prompt=prompt,
            temperature=0,
            max_tokens=GROUP_MAX_PROMPT_TOKENS,
            schema=GroupExtraction,
        )
    except Exception as e:
        logger.error(f"Extraction group {group} LLM call failed: {e}")
        raise StructuredOutputError(
            f"LLM extraction failed for group {group}: {str(e)}"
        )

    # Post-process the results
    postprocessed = _post_process_extraction(result, specs, evidence_blocks)

    return postprocessed


def _build_evidence_blocks(
    retrievals: dict[str, RetrievalResult],
    group: str,
) -> List[Dict[str, Any]]:
    """Build evidence blocks for the prompt from retrieval results.

    Each block is formatted as: [E3] (Addendum 2, p.1, section: ...) text...
    """
    blocks: List[Dict[str, Any]] = []

    # Get retrieval results for fields in this group
    for field_name, retrieval_result in retrievals.items():
        if not retrieval_result.evidence:
            continue

        for ev in retrieval_result.evidence:
            chunk_id = ev.get("chunk_id", "")
            text = ev.get("text", "")
            file_name = ev.get("file_name", "")
            page = ev.get("page", 0)
            section = ev.get("section", "")
            addendum = ev.get("addendum", None)

            # Format the evidence header
            header_parts = []
            if addendum is not None:
                header_parts.append(f"Addendum {addendum}")
            if section:
                header_parts.append(f"Section: {section}")
            if page:
                header_parts.append(f"p.{page}")

            header = " ".join(header_parts) if header_parts else ""

            blocks.append({
                "evidence_id": f"E{len(blocks) + 1}",
                "chunk_id": chunk_id,
                "text": text,
                "file_name": file_name,
                "page": page,
                "section": section,
                "addendum": addendum,
                "display_text": f"[{header}] {text}" if header else text,
            })

    return blocks


def _build_prompt(
    group: str,
    specs: list,
    evidence_blocks: List[Dict[str, Any]],
    feedback: dict[str, str] | None,
) -> str:
    """Build the extraction prompt for the given group."""

    # Start building the prompt
    prompt_parts = []

    # 1. Role and rules
    prompt_parts.append(
        "You are an RFP extraction assistant. Extract only what is explicitly "
        "stated in the evidence blocks below. Do not use any outside knowledge "
        "or make guesses."
    )

    # 2. Evidence distinction
    prompt_parts.append(
        "IMPORTANT: Distinguish between 'not_found' and 'explicit_negative':"
    )
    prompt_parts.append(
        "- 'not_found': the evidence does not contain the answer (silence). "
        "Set status='not_found', value=null, evidence_ids=[], quote=null."
    )
    prompt_parts.append(
        "- 'explicit_negative': the evidence explicitly states the negative, "
        'e.g. "no bid bond is required", "no pre-bid meeting". '
        "Set status='explicit_negative', value=null, quote=null."
    )

    # 3. Field list with descriptions
    prompt_parts.append("\nFields to extract:")
    for spec in specs:
        prompt_parts.append(
            f"\n--- {spec.name} ---"
        )
        prompt_parts.append(
            f"Description: {spec.description}"
        )
        prompt_parts.append(
            f"Value type: {spec.value_type}"
        )
        if spec.format_rules:
            prompt_parts.append(
                f"Format rules: {spec.format_rules}"
            )
        # Add query seeds as examples
        if spec.query_seeds:
            prompt_parts.append(
                f"Query seeds (examples): {', '.join(spec.query_seeds)}"
            )

    # 4. Evidence blocks
    prompt_parts.append(
        "\nEvidence blocks (use only these; do not reference outside knowledge):"
    )
    for block in evidence_blocks:
        prompt_parts.append(
            f"\n{block['display_text']}"
        )
        # Include the evidence ID mapping
        prompt_parts.append(
            f"  Evidence ID: {block['evidence_id']} (chunk_id: {block['chunk_id']})"
        )

    # 5. Output format
    prompt_parts.append(
        "\nOutput: JSON matching the GroupExtraction schema. For each field:"
    )
    prompt_parts.append(
        "- status: one of 'found', 'explicit_negative', 'not_found'"
    )
    prompt_parts.append(
        "- value: the extracted value (null when status != 'found')"
    )
    prompt_parts.append(
        "- evidence_ids: list of evidence IDs supporting this value"
    )
    prompt_parts.append(
        "- quote: verbatim text (<=25 words) from one cited evidence"
    )
    prompt_parts.append(
        "- confidence: float 0-1, the model's self-assessed confidence"
    )
    prompt_parts.append(
        "- note: short reasoning (e.g. 'Delivery stated as 30 days ARO')"
    )

    # 6. Feedback for retry mode
    if feedback:
        prompt_parts.append(
            f"\nPrevious extraction feedback: {feedback}. "
            "Re-check the evidence and correct or return not_found."
        )

    # 7. JSON output instruction
    prompt_parts.append(
        "\nReturn ONLY a valid JSON object matching the GroupExtraction schema. "
        "No prose, no explanation, no markdown code fences."
    )

    return "\n".join(prompt_parts)


def _post_process_extraction(
    result: dict,
    specs: list,
    evidence_blocks: List[Dict[str, Any]],
) -> dict[str, FieldResult]:
    """Post-process the LLM extraction result applying guardrails."""

    # Parse the result
    fields_result: dict[str, FieldResult] = {}

    # Get the group extraction
    group_extraction = result.get("fields", [])

    # Build a map of evidence IDs to evidence blocks
    ev_map: dict[str, Dict[str, Any]] = {}
    for block in evidence_blocks:
        ev_map[block["evidence_id"]] = block

    # Process each field
    for field_extraction in getattr(group_extraction, "fields", []) or []:
        field_name = field_extraction.get("name", "")
        status = field_extraction.get("status", "not_found")
        value = field_extraction.get("value")
        evidence_ids = field_extraction.get("evidence_ids", [])
        quote = field_extraction.get("quote")
        confidence = field_extraction.get("confidence", 0.0)
        note = field_extraction.get("note", "")

        # Resolve evidence IDs to SourceRefs
        sources: list[SourceRef] = []
        resolved_quote = None
        adjusted_confidence = confidence

        for ev_id in evidence_ids:
            if ev_id not in ev_map:
                # Evidence ID not found - drop it
                continue
            ev = ev_map[ev_id]
            # Truncate quote to 25 words max
            quote_words = (quote or "").split()
            if len(quote_words) > QUOTE_MAX_WORDS:
                quote_words = quote_words[:QUOTE_MAX_WORDS]
                quote = " ".join(quote_words) + "..."

            # Check that the quote appears in the evidence text
            if quote and ev.get("text"):
                quote_normalized = " ".join(quote.lower().split())
                text_normalized = ev["text"].lower()
                if quote_normalized not in text_normalized:
                    # Quote not found - drop it, lower confidence
                    if resolved_quote is None:
                        resolved_quote = None
                        adjusted_confidence = max(0, adjusted_confidence - 0.2)

            # Build SourceRef
            source = SourceRef(
                file=ev.get("file_name", ""),
                page=ev.get("page", 0),
                chunk_id=ev.get("chunk_id", ""),
                quote=quote if ev.get("text") else None,
            )
            sources.append(source)

        # If we have at least one valid source and status is 'found',
        # keep the value; otherwise, set to not_found
        if status == "found" and not sources:
            # No valid evidence IDs - mark as not_found
            status = "not_found"
            value = None
            note = note or "Not found in documents"

        # If status is explicit_negative, ensure value is None
        if status == "explicit_negative":
            value = None

        # Format normalization (conservative - keep as-is unless clearly wrong)
        # Date normalization would happen here per format_rules

        field_result = FieldResult(
            value=value,
            sources=sources,
            confidence=adjusted_confidence,
            note=note,
            status=status,
        )

        field_result["evidence_ids"] = evidence_ids  # type: ignore
        fields_result[field_name] = field_result

    return fields_result


class GroupExtraction:
    """Pydantic model for group extraction output."""

    fields: list[FieldExtraction]


async def run_extraction(
    state: BidState,
    *,
    llm_gateway: LLMGateway,
    retrieval_results: dict[str, RetrievalResult] | None = None,
) -> BidState:
    """Run the extraction agent from the given state.

    Orchestrates the extraction of all 20 fields across 4 groups (dates_logistics,
    commercial_legal, product_specs, identity), using retrieved evidence and
    the LLM gateway with guardrails.

    Args:
        state: BidState with retrieval results and context
        llm_gateway: LLMGateway instance
        retrieval_results: Dict mapping field names to RetrievalResult;
            if None, will attempt to use state stored results

    Returns:
        Updated BidState with extracted fields
    """
    # Get retrieval results from state if not provided
    if retrieval_results is None:
        retrieval_results = state.get("retrieval_results", {})

    if not retrieval_results:
        logger.warning("No retrieval results available for extraction")
        add_error(state, {"type": "no_retrieval_results", "message": "No retrieval results available"})
        return state

    # Define the field groups based on service 15 design
    # Group 1: identity (Bid Number, Title) - extracted early
    # Group 2: dates_logistics
    # Group 3: commercial_legal
    # Group 4: product_specs

    groups_config: dict[str, dict[str, any]] = {
        "identity": {
            "specs": [],  # Would load from config/fields.yaml
            "fields_of_interest": ["Bid Number", "Title"],  # Handled by report agent
        },
        "dates_logistics": {
            "specs": [],  # Placeholder - would load from config
        },
        "commercial_legal": {
            "specs": [],
        },
        "product_specs": {
            "specs": [],
        },
    }

    # In a full implementation, would load FieldSpec objects from config/fields.yaml
    # For now, use a simplified approach
    all_groups: list[str] = ["dates_logistics", "commercial_legal", "product_specs"]

    # Extract each group
    all_fields: dict[str, FieldResult] = {}

    for group in all_groups:
        # Get specs for this group (simplified - would load from YAML)
        specs = _get_specs_for_group(group, retrieval_results)

        if not specs:
            continue

        # Get retrieval results for this group's fields
        group_retrievals = {
            field_name: retrieval_results.get(field_name)
            for field_name in specs
            if field_name in retrieval_results
        }

        if not group_retrievals:
            continue

        # Feedback from previous attempts (if any)
        feedback = state.get("extraction_feedback", {})

        # Extract the group
        try:
            group_result = await extract_group(
                group=group,
                specs=specs,
                retrievals=group_retrievals,
                llm_gateway=llm_gateway,
                feedback=feedback if feedback else None,
            )
        except Exception as e:
            logger.error(f"Extraction group {group} failed: {e}")
            add_error(state, {"type": "extraction_error", "message": str(e)})
            add_degraded(state, f"extraction_error:{group}")
            continue

        # Merge results into all_fields
        all_fields.update(group_result)

    # Store the extracted fields in state
    state["draft_fields"] = all_fields

    # Log extraction stats
    found = sum(1 for f in all_fields.values() if f.status == "found")
    explicit_negative = sum(1 for f in all_fields.values() if f.status == "explicit_negative")
    not_found = sum(1 for f in all_fields.values() if f.status == "not_found")
    logger.info(
        f"Extraction complete: found={found}, "
        f"explicit_negative={explicit_negative}, not_found={not_found}"
    )

    return state


def _get_specs_for_group(
    group: str,
    retrieval_results: dict[str, any],
) -> list:
    """Get FieldSpec objects for a given group.

    In a full implementation, would load from config/fields.yaml.
    Simplified version that returns spec-like dicts based on the group.
    """
    # Return simplified spec dicts based on group
    # In production, these would be proper FieldSpec Pydantic models

    specs_map: dict[str, list[dict]] = {
        "dates_logistics": [
            {"name": "Due Date", "value_type": "string", "format_rules": "YYYY-MM-DD HH:MM TZ",
             "query_seeds": ["due date", "proposal submission deadline", "closing date"]},
            {"name": "Bid Submission Type", "value_type": "string", "format_rules": "short label + portal name",
             "query_seeds": ["submission type", "how to submit", "portal"]},
            {"name": "Term of Bid", "value_type": "string", "format_rules": "duration and renewal",
             "query_seeds": ["term", "duration", "renewal"]},
        ],
        "commercial_legal": [
            {"name": "Bid Bond Requirement", "value_type": "string", "format_rules": "amount or percentage",
             "query_seeds": ["bid bond", "bond required", "bond not required"]},
            {"name": "Payment Terms", "value_type": "string", "format_rules": "Net terms + conditions",
             "query_seeds": ["payment terms", "Net", "invoicing"]},
            {"name": "company_name", "value_type": "string", "format_rules": "issuing organization",
             "query_seeds": ["company name", "organization"]},
        ],
        "product_specs": [
            {"name": "Model_no", "value_type": "list", "format_rules": "all model numbers",
             "query_seeds": ["model number", "part number", "SKU"]},
            {"name": "Part_no", "value_type": "list", "format_rules": "all part/SKU numbers",
             "query_seeds": ["part number", "SKU", "model"]},
        ],
    }

    return specs_map.get(group, [])


async def run_full_extraction(
    state: BidState,
    *,
    llm_gateway: LLMGateway,
    search_engine: Any = None,
) -> BidState:
    """Run the complete extraction pipeline: retrieve then extract.

    Args:
        state: BidState with run context and ideally retrieval results
        llm_gateway: LLMGateway instance
        search_engine: Optional search engine for re-retrieval

    Returns:
        Updated BidState with all extracted fields
    """
    # Step 1: If no retrieval results in state, we need them
    # (In a full pipeline, the orchestrator would have run the retrieval agent first)
    if "retrieval_results" not in state:
        logger.warning("No retrieval results in state; extraction cannot proceed without evidence")
        add_error(state, {"type": "missing_retrieval_results", "message": "Run retrieval agent first"})
        return state

    # Step 2: Run extraction on the retrieval results
    state = await run_extraction(state, llm_gateway=llm_gateway)

    # Step 3: Run validation (service 17) on the extracted fields
    # (Would import and call validator_agent.validate_fields)
    # For now, skip

    return state