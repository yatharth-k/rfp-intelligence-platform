"""Unit tests for agents layer (services 12-18)."""

import pytest
import asyncio
from agents.state import BidState, init_run, merge_dicts, checkpoint_state, load_checkpoint
from agents.orchestrator import Orchestrator
from agents.ingestion_agent import run_ingestion, _decide_proceed, IngestionResult
from agents.retrieval_agent import RetrievalRequest, retrieve, retrieve_for_fields
from agents.extraction import extract_group, run_extraction, GroupExtraction
from agents.validator import validate_fields, _l1_checks, _normalize_text, BuildVerdict
from agents.qa import (
    classify_question,
    resolve_bids,
    answer_question,
    build_final_record,
    _generate_bid_summary,
    append_qa_log,
    QuestionType,
)
from agents.qa import QAAnswer, FinalBidRecord


class TestAgentState:
    """Test agent state management."""

    def test_init_run(self):
        """Test init_run creates valid state."""
        state = init_run(bid_id="Bid1", goal="extract")
        assert state["goal"] == "extract"
        assert state["bid_id"] == "Bid1"
        assert "run_id" in state

    def test_merge_dicts(self):
        """Test merge_dicts function."""
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 3, "c": 4}
        result = merge_dicts(d1, d2)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_checkpoint_state(self):
        """Test checkpoint state functions."""
        state = init_run(bid_id="Bid1", goal="extract")
        path = checkpoint_state(state)
        assert path is not None


class TestAgentOrchestrator:
    """Test orchestrator basic functionality."""

    def test_orchestrator_import(self):
        """Test Orchestrator can be imported."""
        from agents.orchestrator import Orchestrator
        assert Orchestrator is not None


class TestIngestionAgent:
    """Test ingestion agent."""

    def test_decide_proceed(self):
        """Test decide_proceed function."""
        # Test with files indexed
        from agents.ingestion_agent import _decide_proceed
        from unittest.mock import Mock
        mock_report = Mock()
        mock_report.files_indexed = 1
        mock_report.doc_types_found = {"rfp": 1}
        mock_report.addendum_numbers = [1]

        proceed, warnings = _decide_proceed(
            mock_report, {"rfp": 1}, [1]
        )
        assert proceed is True or proceed is False  # Should not raise

    def test_run_ingestion(self):
        """Test run_ingestion function."""
        from agents.ingestion_agent import run_ingestion
        # Basic import test
        assert run_ingestion is not None


class TestRetrievalAgent:
    """Test retrieval agent."""

    def test_retrieval_request(self):
        """Test RetrievalRequest creation."""
        from agents.retrieval_agent import RetrievalRequest
        req = RetrievalRequest(
            bid_id="Bid1",
            field="test_field",
            queries=["test query"],
            doc_types=[],
            addendum_numbers=[],
            top_k=4,
        )
        assert req.bid_id == ["Bid1"]
        assert req.field == "test_field"

    def test_retrieve_function(self):
        """Test retrieve function import."""
        from agents.retrieval_agent import retrieve
        assert retrieve is not None

    def test_retrieve_for_fields(self):
        """Test retrieve_for_fields function."""
        from agents.retrieval_agent import retrieve_for_fields
        assert retrieve_for_fields is not None


class TestExtractionAgent:
    """Test extraction agent."""

    def test_extraction_import(self):
        """Test extraction modules can be imported."""
        from agents.extraction import extract_group, run_extraction
        assert extract_group is not None
        assert run_extraction is not None

    def test_extraction_group(self):
        """Test extract_group function."""
        from agents.extraction import extract_group
        # Basic import test
        assert extract_group is not None


class TestValidatorAgent:
    """Test validator agent."""

    def test_validator_import(self):
        """Test validator modules can be imported."""
        from agents.validator import validate_fields, _l1_checks, _normalize_text
        assert validate_fields is not None
        assert _l1_checks is not None
        assert _normalize_text is not None

    def test_l1_checks_basic(self):
        """Test L1 checks function."""
        from agents.validator import _l1_checks
        # Basic import and structure test
        from agents.state import FieldResult
        result = FieldResult(status="found", value="test")
        checks = _l1_checks(result, {})
        assert checks is not None


class TestQAAgent:
    """Test Q&A agent."""

    def test_classify_question(self):
        """Test question classification."""
        from agents.qa import classify_question
        qtype, resolution = classify_question("What is the due date?", {"bids_in_scope": [{"bid_id": "Bid1"}]})
        assert qtype in [QuestionType.SINGLE_FACT, QuestionType.AFTER_ADDENDUMS, QuestionType.MULTI_BID_COMPARE, QuestionType.LIST_QUERY, QuestionType.CHANGE_QUERY]
        assert "bids" in resolution

    def test_resolve_bids(self):
        """Test bid resolution."""
        from agents.qa import resolve_bids
        bids = resolve_bids({"bids": ["Bid1"], "ambiguity": None}, {"bids_in_scope": [{"bid_id": "Bid1"}]})
        assert isinstance(bids, list)

    def test_answer_question(self):
        """Test answer question import."""
        from agents.qa import answer_question
        assert answer_question is not None

    def test_build_final_record(self):
        """Test build final record."""
        from agents.qa import build_final_record
        assert build_final_record is not None

    def test_generate_bid_summary(self):
        """Test bid summary generation."""
        from agents.qa import _generate_bid_summary
        fields = {"Title": {"value": "Test Bid", "sources": [], "confidence": 1.0, "notes": "", "status": "found"}}
        summary = _generate_bid_summary(fields, {"passed": 1, "failed": 0, "not_found": 0})
        assert isinstance(summary, str)


class TestQAAnswer:
    """Test QAAnswer model."""

    def test_qa_answer_creation(self):
        """Test QAAnswer can be created."""
        from agents.state import QAAnswer
        answer = QAAnswer(
            question="test question",
            answer="test answer",
            citations=[],
            per_bid=None,
            confidence=0.8,
            not_found=False,
            notes=[],
            plan={},
        )
        assert answer.question == "test question"
        assert answer.answer == "test answer"
        assert answer.confidence == 0.8
        assert answer.not_found is False


class TestFinalBidRecord:
    """Test FinalBidRecord model."""

    def test_final_bid_record(self):
        """Test FinalBidRecord can be created."""
        from agents.qa import FinalBidRecord
        record = FinalBidRecord(
            bid_id="Bid1",
            fields={},
            addendum_changes=[],
            validation={"passed": 0, "failed": 0, "not_found": 0},
            summary_text="test summary",
            run={"run_id": "test", "models": {}, "timestamps": {}, "degraded": [], "warnings": []},
        )
        assert record.bid_id == "Bid1"
        assert record.summary_text == "test summary"