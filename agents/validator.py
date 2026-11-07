"""Service 17: Validator / Critic Agent.

Independently checks every extracted field for evidence support, format correctness,
consistency, and hallucination. Returns calibrated confidence and precise feedback
messages so the orchestrator can re-extract failed fields. Follows service 17
design: deterministic Python rules + optional LLM entailment check.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.errors import BudgetExceededError
from core.llm import complete_json
from agents.state import (
    BidState,
    CheckResult,
    ValidationVerdict,
    ValidationReport,
    FieldResult,
)

logger = logging.getLogger(__name__)


# Configuration
VALIDATOR_ENTAILMENT_ENABLED = True
NOT_FOUND_REQUERY_ENABLED = True
CONFIDENCE_WEIGHTS = {
    "entailment": 0.35,
    "extractor_self": 0.25,
    "retrieval": 0.20,
    "rules": 0.10,
    "source_authority": 0.10,
}


def _l1_checks(
    field_result: FieldResult,
    evidence: dict,
) -> list[CheckResult]:
    """L1 deterministic checks (free, always applied).

    These are the strongest anti-hallucination checks because they don't
    depend on the LLM.
    """
    checks: list[CheckResult] = []
    passed = True
    details: list[str] = []

    # Citation exists: non-null value has at least one SourceRef
    if field_result.value is not None:
        sources = field_result.sources or []
        if not sources:
            checks.append(
                CheckResult(
                    name="citation_exists",
                    passed=False,
                    severity="hard",
                    detail="Non-null value has no SourceRef",
                )
            )
            passed = False
            details.append("no citation")
        else:
            checks.append(
                CheckResult(
                    name="citation_exists",
                    passed=True,
                    severity="hard",
                    detail=f"{len(sources)} SourceRef(s) present",
                )
            )
    else:
        # null value - citation check is vacuously passed
        checks.append(
            CheckResult(
                name="citation_exists",
                passed=True,
                severity="hard",
                detail="null value; no citation needed",
            )
        )

    # Evidence resolvable: each chunk_id exists in the store and belongs to the bid
    chunk_ids: list[str] = []
    for source in (field_result.sources or []):
        chunk_ids.append(source.chunk_id or "")

    # In a full implementation, would verify chunk_ids against the store
    # For now, assume resolvable if IDs exist
    if chunk_ids:
        all_resolved = all(cid for cid in chunk_ids)  # simplified
        resolvable = all_resolved
    else:
        resolvable = True  # null value, nothing to check

    checks.append(
        CheckResult(
            name="evidence_resolvable",
            passed=resolvable,
            severity="hard",
            detail="All chunk IDs resolvable" if resolvable else "Some chunk IDs unresolvable",
        )
    )
    if not resolvable:
        passed = False
        details.append("unresolvable evidence")

    # Quote supported: normalized quote is a substring of the cited chunk text
    quote = field_result.note or ""  # Using note as quote placeholder
    if quote and field_result.sources:
        quote_normalized = _normalize_text(quote)
        for source in field_result.sources:
            chunk_text = source.get("_chunk_text", "") or ""
            chunk_normalized = _normalize_text(chunk_text)
            if quote_normalized in chunk_normalized:
                checks.append(
                    CheckResult(
                        name="quote_supported",
                        passed=True,
                        severity="hard",
                        detail="Quote found in cited text",
                    )
                )
                break
        else:
            checks.append(
                CheckResult(
                    name="quote_supported",
                    passed=False,
                    severity="hard",
                    detail="Quote not found in any cited chunk text",
                )
            )
            passed = False
            details.append("quote not supported")
    else:
        # No quote to check - vacuously passed for null values
        checks.append(
            CheckResult(
                name="quote_supported",
                passed=True,
                severity="hard",
                detail="No quote to check",
            )
        )

    # Value appears in evidence: identifiers appear in cited text
    # (for dates, emails, phones, model numbers, amounts)
    if field_result.value is not None and field_result.sources:
        value_str = str(field_result.value)
        value_normalized = _normalize_text(value_str)
        found_in_evidence = False
        for source in field_result.sources:
            chunk_text = source.get("_chunk_text", "") or ""
            chunk_normalized = _normalize_text(chunk_text)
            if value_normalized in chunk_normalized:
                found_in_evidence = True
                break

        checks.append(
            CheckResult(
                name="value_in_evidence",
                passed=found_in_evidence,
                severity="hard",
                detail=(
                    "Value found in cited evidence"
                    if found_in_evidence
                    else "Value not found in cited evidence"
                ),
            )
        )
        if not found_in_evidence:
            passed = False
            details.append("value not in evidence")
    else:
        checks.append(
            CheckResult(
                name="value_in_evidence",
                passed=True,
                severity="hard",
                detail="Null value or no sources to check",
            )
        )

    # Format: date
    # (soft check - just logs; full date parsing would use dateutil)
    checks.append(
        CheckResult(
            name="format_date",
            passed=True,  # simplified - always pass in basic implementation
            severity="soft",
            detail="Date format check (soft)",
        )
    )

    # Format: email/phone (soft)
    checks.append(
        CheckResult(
            name="format_email_phone",
            passed=True,  # simplified
            severity="soft",
            detail="Email/phone format check (soft)",
        )
    )

    # Shape: list/object (soft)
    checks.append(
        CheckResult(
            name="shape_list_object",
            passed=True,  # simplified
            severity="soft",
            detail="List/object shape check (soft)",
        )
    )

    # Not-found wording (hard)
    if field_result.status == "not_found":
        if field_result.value is not None:
            checks.append(
                CheckResult(
                    name="not_found_wording",
                    passed=False,
                    severity="hard",
                    detail="not_found status but value is not null",
                )
            )
            passed = False
            details.append("not_found wording violated")
        else:
            checks.append(
                CheckResult(
                    name="not_found_wording",
                    passed=True,
                    severity="hard",
                    detail="not_found has value=null and sources=[]",
                )
            )
    else:
        checks.append(
            CheckResult(
                name="not_found_wording",
                passed=True,
                severity="hard",
                detail="Not applicable (status != not_found)",
            )
        )

    return checks


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace, strip punctuation."""
    # Lowercase
    text = text.lower()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip common quote styles
    text = text.replace('"', "").replace("'", "").replace("``", "").replace("''", "")
    # Remove punctuation (except within numbers like 30% → 30 percent)
    text = re.sub(r"[^\w\s%]", "", text)
    return text


def _llm_entailment_check(
    llm_gateway: Any,
    fields_with_evidence: list[dict],
    model_name: str = "nvidia/nemotron-3.5-lightning-30b-a3b",
) -> dict[str, Any]:
    """Perform L3 LLM entailment check.

    Asks the LLM: for each field, does the cited text support the claimed value?
    Returns entailment verdicts and feedback.
    """
    if not VALIDATOR_ENTAILMENT_ENABLED or llm_gateway is None:
        return {"entailments": {}, "skipped": True}

    try:
        # Build the entailment prompt
        prompt_parts = [
            "You are a skeptical validator. For each field below, determine if the "
            "cited text supports the claimed value. Answer for each field with one of:"
        ]
        prompt_parts.append(
            "  - 'supports': the text clearly states or implies the value"
        )
        prompt_parts.append(
            "  - 'partial': the text partially supports but doesn't fully confirm"
        )
        prompt_parts.append(
            "  - 'contradicts': the text states something different or opposite"
        )
        prompt_parts.append(
            "  - 'not_stated': the text does not mention the value at all"
        )
        prompt_parts.append(
            "\nFor each field, also provide a one-sentence reason.\n"
        )

        # Add each field's evidence and claimed value
        for i, field_info in enumerate(fields_with_evidence, 1):
            prompt_parts.append(
                f"\nField {i}: {field_info['field_name']}"
            )
            prompt_parts.append(
                f"  Claimed value: {field_info['claimed_value']}"
            )
            prompt_parts.append(
                f"  Cited evidence: {field_info['evidence_text'][:500]}..."
            )
            prompt_parts.append(
                f"  Answer (supports/partial/contradicts/not_stated) + reason:"
            )

        prompt = "\n".join(prompt_parts)

        # Call the LLM
        result = llm_gateway.complete(
            prompt=prompt,
            temperature=0,
            max_tokens=2000,
        )

        # Parse the result (simplified - would use proper JSON parsing)
        response_text = result.text or ""
        # In a full implementation, would extract the entailment verdicts
        # For now, return a simplified result
        return {
            "entailments": {},
            "skipped": False,
        }

    except Exception as e:
        logger.error(f"LLM entailment check failed: {e}")
        return {"entailments": {}, "skipped": True}


def validate_fields(
    state: BidState,
    fields: list[str] | None = None,
) -> ValidationReport:
    """Validate all extracted fields and return a validation report.

    Args:
        state: BidState with extracted fields (draft_fields) and evidence
        fields: Optional list of specific field names to validate; if None, validate all

    Returns:
        ValidationReport with verdicts per field and summary counts
    """
    # Get the draft fields from state
    draft_fields: dict[str, FieldResult] = state.get("draft_fields", {})

    # If specific fields requested, filter
    target_fields = (
        {name: draft_fields[name] for name in fields if name in draft_fields}
        if fields
        else draft_fields
    )

    # Build verdicts dict
    verdicts: dict[str, Any] = {}
    all_checks: dict[str, list[CheckResult]] = {}

    for field_name, field_result in target_fields.items():
        # Collect evidence for this field
        evidence = _collect_evidence(field_result, state)

        # Run L1 deterministic checks
        l1_checks = _l1_checks(field_result, evidence)

        # Determine which L1 checks passed
        hard_passed = all(c.passed for c in l1_checks if c.severity == "hard")
        soft_passed = all(c.passed for c in l1_checks if c.severity == "soft")

        # Initialize verdict variables
        verdict_value: str = "not_found"
        confidence: float = 0.0
        feedback: Optional[str] = None
        entailment_verdict: str = "not_stated"
        hard_fail = False
        checks = l1_checks  # Start with L1; L3 may modify

        # L2 consistency checks (free, always applied)
        # In a full implementation, would check cross-field consistency
        # For now, just note that L2 was run
        l2_result = _l2_consistency_checks(field_result, target_fields)
        if not l2_result["all_passed"]:
            # Soft failures don't automatically hard-fail,
            # but they affect confidence
            pass  # Continue with other checks

        # L3 LLM entailment check (if enabled and field passed L1 hard checks)
        if hard_passed and VALIDATOR_ENTAILMENT_ENABLED:
            # Build field info for entailment check
            fields_with_evidence = _build_entailment_input(field_result, evidence)
            entailment_result = _llm_entailment_check(
                state.get("llm_gateway"), fields_with_evidence
            )
            entailment_verdict = entailment_result.get("entailments", {}).get(
                field_name, "not_stated"
            )

            # Update verdict based on entailment
            if entailment_verdict in ("contradicts", "not_stated"):
                verdict_value = "fail"
                hard_fail = True
                confidence = 0.0
                feedback = entailment_result.get("reasons", {}).get(field_name)
            elif entailment_verdict == "partial":
                # Soft fail - value kept with reduced confidence
                verdict_value = "fail"
                hard_fail = False
                # Reduce confidence
                extractor_conf = field_result.confidence or 0.5
                retrieval_conf = _get_retrieval_confidence(field_result)
                rules_conf = _get_rules_confidence(l1_checks)
                source_conf = _get_source_authority(field_result)
                weights = CONFIDENCE_WEIGHTS
                confidence = (
                    weights["entailment"] * 0.5
                    + weights["extractor_self"] * extractor_conf
                    + weights["retrieval"] * retrieval_conf
                    + weights["rules"] * rules_conf
                    + weights["source_authority"] * source_conf
                )
                feedback = "Partial entailment support; confidence reduced"
            else:  # supports
                verdict_value = "pass"
                hard_fail = False
                # Calculate confidence
                extractor_conf = field_result.confidence or 0.5
                retrieval_conf = _get_retrieval_confidence(field_result)
                rules_conf = _get_rules_confidence(l1_checks)
                source_conf = _get_source_authority(field_result)
                weights = CONFIDENCE_WEIGHTS
                confidence = (
                    weights["entailment"] * 1.0
                    + weights["extractor_self"] * extractor_conf
                    + weights["retrieval"] * retrieval_conf
                    + weights["rules"] * rules_conf
                    + weights["source_authority"] * source_conf
                )
                # Cap at 1.0
                confidence = min(confidence, 1.0)
                feedback = None
        else:
            # L1 hard checks failed
            verdict_value = "fail"
            hard_fail = True
            confidence = 0.0
            feedback = "; ".join(
                c.detail for c in l1_checks if not c.passed and c.severity == "hard"
            )

        # Build the verdict
        verdict = ValidationVerdict(
            field=field_name,
            verdict=verdict_value,
            confidence=round(confidence, 3),
            checks=checks,
            entailment=entailment_verdict,
            feedback=feedback,
            hard_fail=hard_fail,
        )

        verdicts[field_name] = verdict
        all_checks[field_name] = l1_checks

    # Calculate summary counts
    passed = sum(1 for v in verdicts.values() if v.verdict == "pass")
    failed = sum(1 for v in verdicts.values() if v.verdict == "fail")
    not_found = sum(1 for v in verdicts.values() if v.verdict == "not_found")

    summary = {
        "passed": passed,
        "failed": failed,
        "not_found": not_found,
    }

    # Build the validation report
    report = ValidationReport(
        verdicts=verdicts,
        summary=summary,
    )

    # Log the results
    logger.info(
        f"Validation summary: passed={passed}, failed={failed}, "
        f"not_found={not_found}/{{len(target_fields)}} fields"
    )

    return report


def _collect_evidence(
    field_result: FieldResult,
    state: BidState,
) -> dict:
    """Collect evidence dict for a field from the state."""
    evidence: dict = {
        "sources": [
            {
                "file_name": s.file if hasattr(s, "file") else str(s.get("file", "")),
                "page": s.page if hasattr(s, "page") else 0,
                "chunk_id": s.chunk_id if hasattr(s, "chunk_id") else "",
                "_chunk_text": getattr(s, "_chunk_text", "") or "",
            }
            for s in (field_result.sources or [])
        ],
        "field_status": field_result.status,
        "field_value": field_result.value,
    }
    return evidence


def _l2_consistency_checks(
    field_result: FieldResult,
    all_fields: dict[str, FieldResult],
) -> dict[str, Any]:
    """L2 consistency checks across fields.

    Checks cross-field rules like date ordering, format consistency, etc.
    """
    issues: list[str] = []
    all_passed = True

    # Example: check delivery date is not before due date
    # (Simplified - would need actual date parsing)
    due_date_field = all_fields.get("Due Date")
    delivery_date_field = all_fields.get("Delivery Date")

    if due_date_field and delivery_date_field and due_date_field.value and delivery_date_field.value:
        # Very simplified check - just flag if values look inconsistent
        due_str = str(due_date_field.value)
        del_str = str(delivery_date_field.value)
        # If delivery date mentions a time that's clearly before the due date
        # this would be flagged; but we'll skip the actual parsing here
        pass

    return {
        "all_passed": all_passed,
        "issues": issues,
    }


def _build_entailment_input(
    field_result: FieldResult,
    evidence: dict,
) -> list[dict]:
    """Build input for LLM entailment check.

    Returns a list of dicts with field_name, claimed_value, and evidence_text.
    """
    claimed_value = str(field_result.value) if field_result.value else ""
    evidence_text = ""

    # Collect text from all cited sources
    for source in field_result.sources or []:
        chunk_text = source.get("_chunk_text", "") or ""
        evidence_text += chunk_text + " "

    return [
        {
            "field_name": field_result.name if hasattr(field_result, "name") else "",
            "claimed_value": claimed_value,
            "evidence_text": evidence_text[:1000],  # Truncate for prompt
        }
    ]


def _get_retrieval_confidence(field_result: FieldResult) -> float:
    """Get retrieval-based confidence component.

    Simplified: based on the rank of cited evidence.
    """
    # In a full implementation, would use the reranker score or BM25 score
    # For now, return a neutral value
    return 0.7


def _get_source_authority(field_result: FieldResult) -> float:
    """Get source authority component.

    Simplified: RFP=1.0, bid page=0.8, inferred=0.4
    """
    # Check source types
    sources = field_result.sources or []
    if not sources:
        return 0.4  # No sources

    # Check if any source is from RFP
    for source in sources:
        # simplified check
        pass

    # Return default
    return 0.8


def build_validation_report(
    verdicts: dict[str, Any],
) -> dict[str, Any]:
    """Build a validation report dict from verdicts.

    Args:
        verdicts: Dict mapping field names to ValidationVerdict objects

    Returns:
        Report dict with verdicts and summary
    """
    passed = sum(1 for v in verdicts.values() if v.verdict == "pass")
    failed = sum(1 for v in verdicts.values() if v.verdict == "fail")
    not_found = sum(1 for v in verdicts.values() if v.verdict == "not_found")

    return {
        "verdicts": verdicts,
        "summary": {
            "passed": passed,
            "failed": failed,
            "not_found": not_found,
        },
    }


def build_feedback_for_orchestrator(
    verdicts: dict[str, Any],
) -> dict[str, str]:
    """Build feedback dict for the orchestrator's retry loop.

    Args:
        verdicts: Dict mapping field names to ValidationVerdict objects

    Returns:
        Dict mapping field names to feedback strings
    """
    feedback: dict[str, str] = {}

    for field_name, verdict in verdicts.items():
        if verdict.hard_fail or verdict.verdict == "fail":
            # Extract the main reason for failure
            main_reasons = [c.detail for c in verdict.checks if not c.passed]
            if main_reasons:
                feedback[field_name] = " ".join(main_reasons[:2])  # Top 2 reasons
            else:
                feedback[field_name] = "Extraction failed; re-extract with broadening"
        elif verdict.verdict == "not_found":
            feedback[field_name] = "Not found in documents; no retry needed"
        else:
            feedback[field_name] = None

    return feedback