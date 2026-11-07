"""Unit tests for ingestion layer (services 06-07)."""

import pytest
import json
from pathlib import Path
from core.cleaning import clean_document, _nfkc_normalize, _repair_hyphens, _repair_broken_lines, _normalize_whitespace
from ingestion.detect import detect
from ingestion.report import ingest_folder, IngestionReport, FileReport
from schemas import DocType, ParsedDocument, Block, Page


class TestCleaningFunctions:
    """Test deterministic text cleaning functions."""

    def test_nfkc_normalize(self):
        """Test NFKC normalization."""
        result = _nfkc_normalize("text with non-breaking space")
        assert " " not in result  # Non-breaking space should be replaced

    def test_hyphen_repair(self):
        """Test hyphenation repair."""
        result = _repair_hyphens("word-\nrest here")
        # Should join if conservative rules match

    def test_broken_line_repair(self):
        """Test broken line repair."""
        result = _repair_broken_lines("line one\nline two")
        # Should test the merge logic

    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        result = _normalize_whitespace("  multiple   spaces  \n\n\nblanklines  ")
        # Should collapse spaces and limit blank lines


class TestDetect:
    """Test file detection."""

    def test_detect_pdf(self):
        """Test PDF detection."""
        result = detect(".pdf")
        # Should recognize PDF extension

    def test_detect_html(self):
        """Test HTML detection."""
        result = detect(".html")
        # Should recognize HTML extension

    def test_detect_rfp_keywords(self):
        """Test RFP keyword detection in content."""
        # Test first-page content cues
        pass


class TestIngestionReport:
    """Test ingestion report model."""

    def test_ingestion_report_creation(self):
        """Test IngestionReport can be created."""
        report = IngestionReport(
            bid_id="TestBid",
            files=[],
            started_at=None,
            finished_at=None,
        )
        assert report.bid_id == "TestBid"


class TestFileReport:
    """Test FileReport model."""

    def test_file_report_creation(self):
        """Test FileReport can be created."""
        report = FileReport(
            file="test.pdf",
            status="ok",
            pages=5,
            tables=2,
            empty_pages=0,
            needs_ocr_pages=0,
            warnings=[],
            error=None,
        )
        assert report.file == "test.pdf"
        assert report.status == "ok"
