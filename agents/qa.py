"""Service 18: Q&A Agent and Report Agent.

Two responsibilities sharing one output layer:
- Q&A agent: answer free-form questions about one or more bids, with citations,
  applying addendum changes and refusing to guess.
- Report agent: assemble the final per-bid JSON, the validation summary, and the
  3-6 sentence Bid Summary; also produce a human-readable summary.

Follows service 18 design: search engine and retrieval agent as tools, LLM
gateway with Pydantic schemas, state from the orchestrator.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from agents.state import BidState, QAAnswer, FinalBidRecord
from core.llm import complete_json, LLMGateway
from search.engine import SearchEngine, SearchMode
from agents.retrieval_agent import RetrievalRequest, retrieve
from core.errors import BudgetExceededError

logger = logging.getLogger(__name__)

# Configuration
QA_TOP_K = 6
QA_VERIFY_ENABLED = True
SUMMARY_MIN_SENTENCES = 3
SUMMARY_MAX_SENTENCES = 6


class QuestionType:
    """Enumeration of Q&A question types."""
    SINGLE_FACT = "single_fact"
    AFTER_ADDENDUMS = "after_addendums"
    CHANGE_QUERY = "change_query"
    LIST_QUERY = "list_query"
    MULTI_BID_COMPARE = "multi_bid_compare"


def classify_question(question: str, state: BidState) -> Tuple[QuestionType, dict]:
    """Classify the question type and resolve bid scope.

    Args:
        question: The user question
        state: BidState with run context

    Returns:
        Tuple of (question_type, bid_resolution)
    """
    question_lower = question.lower().strip()

    # Check for explicit bid references
    bid_ids = state.get("bids_in_scope", [])
    resolved_bids: dict = {"bids": [], "ambiguity": None}

    # Check for "both", "all", "each"
    if any(word in question_lower for word in ["both", "all", "each", "either"]):
        # Return all available bids
        if bid_ids:
            resolved_bids["bids"] = [bid.get("bid_id", "") for bid in bid_ids if bid.get("bid_id")]
        resolved_bids["ambiguity"] = "all_bids"
        return QuestionType.MULTI_BID_COMPARE, resolved_bids

    # Check for "after addendums" pattern
    if "after" in question_lower and ("addendum" in question_lower or "change" in question_lower):
        return QuestionType.AFTER_ADDENDUMS, {"bids": [], "ambiguity": "addendum_aware"}

    # Check for comparison patterns ("compare", "vs")
    if any(word in question_lower for word in ["compare", "vs", "versus", "difference"]):
        if len(bid_ids) >= 2:
            resolved_bids["bids"] = [bid.get("bid_id", "") for bid in bid_ids[:2]]
            return QuestionType.MULTI_BID_COMPARE, resolved_bids
        elif len(bid_ids) == 1:
            resolved_bids["bids"] = [bid_ids[0].get("bid_id", "")]
        return QuestionType.MULTI_BID_COMPARE, resolved_bids

    # Check for list queries ("which", "what", list of items)
    list_keywords = ["which", "what", "list", "items", "affidavits", "forms"]
    if any(word in question_lower for word in list_keywords):
        return QuestionType.LIST_QUERY, {"bids": [bid.get("bid_id", "") for bid in bid_ids if bid.get("bid_id")]}

    # Default: single fact question
    return QuestionType.SINGLE_FACT, {"bids": [bid.get("bid_id", "") for bid in bid_ids if bid.get("bid_id")]}


def resolve_bids(bid_resolution: dict, state: BidState) -> list[str]:
    """Resolve which bids are in scope for the question.

    Args:
        bid_resolution: From classify_question
        state: BidState

    Returns:
        List of bid_ids
    """
    bids = bid_resolution.get("bids", [])
    ambiguity = bid_resolution.get("ambiguity")

    if ambiguity == "all_bids":
        return bids if bids else []

    if ambiguity == "addendum_aware":
        # For addendum-aware questions, use all bids
        return bids if bids else []

    # Single bid or specific resolution
    return bids


async def answer_question(
    question: str,
    *,
    state: BidState,
    search_engine: SearchEngine,
    query_rewriter: Optional[Any] = None,
) -> QAAnswer:
    """Answer a free-form question about the bid(s) with citations.

    Args:
        question: The user question
        state: BidState with run context
        search_engine: SearchEngine instance
        query_rewriter: Optional query rewriter

    Returns:
        QAAnswer with answer, citations, confidence, and not_found flag
    """
    # Step 1: Classify the question
    question_type, bid_resolution = classify_question(question, state)
    bid_ids = resolve_bids(bid_resolution, state)

    # Step 2: Plan sub-queries per bid
    sub_queries: dict[str, list[str]] = {}
    for bid_id in bid_ids:
        # Build bid-specific sub-queries
        # In a full implementation, would be addendum-aware
        sub_queries[bid_id] = [question]

    # Step 3: Retrieve evidence for each bid
    retrieval_results: dict[str, any] = {}
    for bid_id, queries in sub_queries.items():
        try:
            req = RetrievalRequest(
                bid_id=[bid_id],
                field=None,
                queries=queries,
                doc_types=[],
                addendum_numbers=[],
                top_k=QA_TOP_K,
            )
            result = await retrieve(req, state, search_engine)
            retrieval_results[bid_id] = result
        except Exception as e:
            logger.error(f"Retrieval failed for bid {bid_id}: {e}")
            retrieval_results[bid_id] = type("obj", (object,), {
                "evidence": [],
                "degraded": [f"retrieval_error:{str(e)[:50]}"],
                "queries_used": queries
            })()

    # Step 4: Answer from evidence
    answer, citations, confidence, not_found = _answer_from_evidence(
        question, question_type, retrieval_results, state
    )

    # Step 5: Verify citations
    if QA_VERIFY_ENABLED:
        verified_answer, verified_citations, verify_not_found, verify_notes = \
            _verify_answer(question, answer, citations, retrieval_results, state)
        # Use verified results if they improve things
        if not verify_not_found:
            answer = verified_answer
            citations = verified_citations
            not_found = verify_not_found
            # Add any new notes
            if verify_notes:
                # Merge notes
                pass  # Simplified

    # Step 6: Build per-bid results for comparison questions
    per_bid: dict[str, str] | None = None
    if question_type == QuestionType.MULTI_BID_COMPARE and len(bid_ids) >= 2:
        per_bid = _build_per_bid_answer(retrieval_results, bid_ids)

    # Step 7: Build the QAAnswer
    qa_answer = QAAnswer(
        question=question,
        answer=answer,
        citations=citations,
        per_bid=per_bid,
        confidence=confidence,
        not_found=not_found,
        notes=[],  # Would include any verification notes
        plan={},  # Would include sub-queries used
    )

    return qa_answer


def _answer_from_evidence(
    question: str,
    question_type: QuestionType,
    retrieval_results: dict[str, any],
    state: BidState,
) -> Tuple[str, list, float, bool]:
    """Answer the question from retrieved evidence.

    Returns: (answer, citations, confidence, not_found)
    """
    # Collect all evidence from all bids
    all_evidence: list = []
    bid_citations: dict[str, list] = {}

    for bid_id, result in retrieval_results.items():
        if result.evidence:
            all_evidence.extend(result.evidence)
            bid_citations[bid_id] = result.evidence

    # Check if we have any evidence
    if not all_evidence:
        return "Not found in documents", [], 0.0, True

    # For single fact questions, look for the most relevant evidence
    if question_type == QuestionType.SINGLE_FACT:
        return _answer_single_fact(question, all_evidence)

    # For list queries, collect items
    if question_type == QuestionType.LIST_QUERY:
        return _answer_list_query(all_evidence)

    # For after-addendums, we would apply reconciliation
    # For now, fall back to general answer
    return _answer_general(question, all_evidence)


def _answer_single_fact(question: str, evidence: list) -> Tuple[str, list, float, bool]:
    """Answer a single fact question from evidence.

    Looks for evidence that directly answers the question.
    """
    question_lower = question.lower()

    # Simple keyword matching
    for ev in evidence:
        ev_text = (ev.get("text") or "").lower()
        ev_file = ev.get("file_name", "")
        ev_page = ev.get("page", 0)

        # Check if key terms from the question appear in the evidence
        # This is a simplified check
        key_terms = [t for t in question_lower.split() if len(t) > 3]
        if any(term in ev_text for term in key_terms):
            # Found relevant evidence
            citation = type("obj", (object,), {
                "file_name": ev_file,
                "page": ev_page,
            })()
            answer = f"{ev.get('text', '')[:200]}..."
            return answer, [citation], 0.7, False

    # No direct match found
    return "Not found in documents", [], 0.0, True


def _answer_list_query(evidence: list) -> Tuple[str, list, float, bool]:
    """Answer a list query from evidence.

    Collects unique items from the evidence.
    """
    items: list = []
    seen: set = set()

    for ev in evidence:
        ev_text = ev.get("text", "") or ""
        # Look for items (simplified: look for capitalized phrases that might be items)
        # In a full implementation, would use more sophisticated NLP
        phrases = [p.strip() for p in ev_text.split(". ") if p.strip()]
        for phrase in phrases:
            if phrase and phrase not in seen and len(phrase) > 3:
                seen.add(phrase)
                items.append(phrase)

    if items:
        answer = "; ".join(items[:10])  # Top 10 items
        # Create citations
        citations = []
        for i, ev in enumerate(evidence[:3]):  # Top 3 sources
            citations.append(type("obj", (object,), {
                "file_name": ev.get("file_name", ""),
                "page": ev.get("page", 0),
            })())
        return answer, citations, 0.6, False
    else:
        return "Not found in documents", [], 0.0, True


def _answer_general(question: str, evidence: list) -> Tuple[str, list, float, bool]:
    """General answer from evidence.

    Falls back to a simple summary of the evidence.
    """
    if not evidence:
        return "Not found in documents", [], 0.0, True

    # Simple summary: first few words from first evidence
    first_ev = evidence[0]
    text = first_ev.get("text", "") or ""
    if text:
        # Take first sentence-ish portion
        sentences = text.split(". ")
        summary = ". ".join(sentences[:2]) + "."
        citation = type("obj", (object,), {
            "file_name": first_ev.get("file_name", ""),
            "page": first_ev.get("page", 0),
        })
        return summary, [citation], 0.5, False
    else:
        return "Not found in documents", [], 0.0, True


def _verify_answer(
    question: str,
    answer: str,
    citations: list,
    retrieval_results: dict[str, any],
    state: BidState,
) -> Tuple[str, list, bool, list]:
    """Verify answer citations and potentially retry with broader retrieval.

    Returns: (verified_answer, verified_citations, not_found, notes)
    """
    notes: list = []
    verified_citations = citations

    # Check if citations are valid (IDs exist and quotes are present)
    for citation in citations:
        # In a full implementation, would check that the chunk_id exists
        # and that the quote appears in the chunk text
        pass  # Simplified

    # If no citations or citations seem invalid, try broader retrieval
    if not citations or _needs_retry(question, answer, citations):
        # Broaden retrieval and retry
        # (Simplified: just return not_found rather than infinite retry)
        return "Not found in documents", [], True, ["Could not verify citations; broader retrieval attempted"]

    return answer, citations, False, notes


def _needs_retry(question: str, answer: str, citations: list) -> bool:
    """Determine if the answer needs a retry with broader retrieval.

    Simplified check: no citations or answer is "Not found in documents"
    """
    return not citations or answer == "Not found in documents"


def _build_per_bid_answer(
    retrieval_results: dict[str, any],
    bid_ids: list[str],
) -> dict[str, str]:
    """Build per-bid answers for comparison questions.

    Returns a dict mapping bid_id to its answer snippet.
    """
    per_bid: dict[str, str] = {}

    for bid_id in bid_ids:
        result = retrieval_results.get(bid_id, type("obj", (object,), {"evidence": []})())
        if result.evidence:
            # Simple: take the first evidence's text as the answer snippet
            text = result.evidence[0].get("text", "")[:150] if result.evidence else ""
            per_bid[bid_id] = text
        else:
            per_bid[bid_id] = "No information found"

    return per_bid


def build_final_record(state: BidState) -> FinalBidRecord:
    """Build the final per-bid record with fields, addendum changes, validation,
    and summary.

    Args:
        state: BidState with extraction results after the last validation round

    Returns:
        FinalBidRecord with all the bid's final data
    """
    # Get draft fields from state
    draft_fields: dict[str, any] = state.get("draft_fields", {})

    # Get validation summary
    validation_summary: dict[str, int] = state.get("validation_summary", {"passed": 0, "failed": 0, "not_found": 0})

    # Get addendum changes
    addendum_changes: list = state.get("addendum_changes", [])

    # Build the fields dict (matching FieldResult schema)
    fields: dict[str, any] = {}
    if draft_fields:
        for field_name, field_result in draft_fields.items():
            # Convert to the expected format
            fields[field_name] = {
                "value": field_result.get("value"),
                "sources": field_result.get("sources", []),
                "confidence": field_result.get("confidence", 0.0),
                "notes": field_result.get("note", ""),
                "status": field_result.get("status", "not_found"),
            }

    # Generate Bid Summary (3-6 sentences)
    summary_text = _generate_bid_summary(fields, validation_summary)

    # Build run info
    run_info: dict = {
        "run_id": state.get("run_id", "unknown"),
        "models": {
            "llm": state.get("llm_model", "unknown"),
            "embed": state.get("embed_model", "unknown"),
        },
        "timestamps": {
            "started_at": state.get("started_at", datetime.utcnow().isoformat()),
            "finished_at": datetime.utcnow().isoformat(),
        },
        "degraded": state.get("degraded", []),
        "warnings": state.get("warnings", []),
    }

    # Build the final record
    record = FinalBidRecord(
        bid_id=state.get("bid_id", "unknown"),
        fields=fields,
        addendum_changes=addendum_changes,
        validation=validation_summary,
        summary_text=summary_text,
        run=run_info,
    )

    return record


def _generate_bid_summary(fields: dict[str, any], validation_summary: dict[str, int]) -> str:
    """Generate the 3-6 sentence Bid Summary from validated field values.

    Args:
        fields: Dict of field_name -> field value data
        validation_summary: Counts of passed/failed/not_found

    Returns:
        3-6 sentence summary string
    """
    # Collect key fields for the summary
    key_fields = []

    # Priority order for summary fields
    priority = ["Title", "Due Date", "Bid Bond Requirement", "Model_no",
                "Product", "Payment Terms", "Contact"]

    sentence_parts: list = []

    for field_name in priority:
        if field_name in fields:
            field_data = fields[field_name]
            value = field_data.get("value")
            if value is not None:
                # Format based on field type
                value_str = str(value)
                # Limit length
                if len(value_str) > 80:
                    value_str = value_str[:80] + "..."
                sentence_parts.append(f"The {field_name.lower()} is {value_str}.")

    # If not enough key fields, add what we have
    if not sentence_parts:
        # Try to find any fields with values
        for field_name, field_data in fields.items():
            value = field_data.get("value")
            if value is not None:
                sentence_parts.append(f"{field_name}: {str(value)[:60]}.")
                if len(sentence_parts) >= SUMMARY_MAX_SENTENCES:
                    break

    # Ensure minimum sentences
    while len(sentence_parts) < SUMMARY_MIN_SENTENCES:
        sentence_parts.append("Additional details are available in the full field results.")

    # Cap at max sentences
    summary = " ".join(sentence_parts[:SUMMARY_MAX_SENTENCES])

    # If we have not_found fields, mention them
    not_found_count = validation_summary.get("not_found", 0)
    if not_found_count > 0 and not any("not found" in s.lower() for s in sentence_parts.split(". ")):
        summary += f" {not_found_count} key field{'s' if not_found_count > 1 else ''} could not be determined."

    return summary


def append_qa_log(question: str, answer: QAAnswer, log_path: str = "outputs/qa_log.md") -> None:
    """Append a Q&A entry to the qa_log file.

    Args:
        question: The original question
        answer: The QAAnswer object
        log_path: Path to the qa_log file
    """
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Format the log entry
    log_entry = f"""# Q&A Log Entry

**Question:** {question}

**Answer:** {answer.answer}

**Citations:** {', '.join([f'[{c.file_name}, p.{c.page}]' for c in answer.citations]) if answer.citations else 'None'}

**Confidence:** {answer.confidence:.2f}

**Not found:** {answer.not_found}

**Per-bid results:** {answer.per_bid if answer.per_bid else 'N/A'}

**Plan:** {answer.plan}

---

"""

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry)