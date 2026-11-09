"""
Service 01: Data Schemas

Pydantic models for the RFP Intelligence Platform data structures.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
import re


# ============================================================
# Core document and chunk schemas
# ============================================================


class Block(BaseModel):
    """A text block on a document page."""
    kind: str = Field(..., description="Type of block (e.g., 'paragraph', 'heading', 'table')")
    text: str = Field(..., description="The block text content")
    page: int = Field(..., ge=1, description="Page number (1-based)")
    font_size: Optional[float] = Field(
        None, description="Font size if available"
    )
    bbox: Optional[List[float]] = Field(
        None, description="Bounding box [x1, y1, x2, y2] if available"
    )
    level: Optional[int] = Field(
        None, description="Heading level (1-based, None for non-heading blocks)"
    )


class Table(BaseModel):
    """A table found on a document page."""
    page_start: int = Field(..., ge=1, description="Starting page number (1-based)")
    page_end: int = Field(..., ge=1, description="Ending page number (1-based)")
    header: List[str] = Field(
        default_factory=list, description="Table header row"
    )
    rows: List[List[str]] = Field(
        default_factory=list, description="Table data rows"
    )
    caption: Optional[str] = Field(
        None, description="Table caption"
    )
    source_strategy: Optional[str] = Field(
        None, description="How the table was extracted"
    )
    bbox: Optional[List[float]] = Field(
        None, description="Bounding box [x1, y1, x2, y2] if available"
    )


class Page(BaseModel):
    """A page of a parsed document."""
    number: int = Field(..., ge=1, description="1-based page number")
    blocks: List["Block"] = Field(
        default_factory=list,
        description="Text blocks on the page with their content and metadata"
    )
    tables: List["Table"] = Field(
        default_factory=list,
        description="Tables found on the page"
    )
    char_count: int = Field(
        0, description="Total character count of blocks on this page"
    )
    empty: bool = Field(
        False,
        description="Whether the page appears empty/scanned"
    )
    needs_ocr: bool = Field(
        False,
        description="Whether the page likely requires OCR"
    )
    flags: List[str] = Field(
        default_factory=list,
        description="Page flags (empty, needs_ocr, etc.)"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Actionable warnings about the page"
    )


class ParsedDocument(BaseModel):
    """A fully parsed document resulting from ingestion."""
    file_name: str = Field(..., description="Original file name")
    file_kind: Literal["pdf", "html", "text", "other"] = Field(
        ..., description="Type of file detected"
    )
    doc_type: Literal["rfp", "addendum", "spec", "affidavit", "other", "bid_page"] = Field(
        ..., description="Document type from detection"
    )
    addendum_number: Optional[int] = Field(
        None, description="Addendum number if applicable"
    )
    doc_date: Optional[datetime] = Field(
        None, description="Document date if detectable"
    )
    title: Optional[str] = Field(
        None, description="Title from first heading/metadata"
    )
    pages: List[Page] = Field(
        ..., description="List of pages with blocks and tables"
    )
    file_hash: str = Field(
        ..., description="SHA-256 hash of the file bytes"
    )
    bid_id: Optional[str] = Field(
        None, description="Bid ID this document belongs to"
    )
    n_chunks: int = Field(
        0, description="Number of chunks this document was split into"
    )
    chunks: Optional[List["Chunk"]] = Field(
        None, description="Chunk objects for indexing"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings from parsing/cleaning"
    )

    @field_validator("file_kind", "doc_type")
    @classmethod
    def normalize_lower(cls, v: str) -> str:
        return v.lower().strip()


class Chunk(BaseModel):
    """A text chunk resulting from document chunking."""
    chunk_id: str = Field(..., description="Deterministic sha256 hash identifier")
    text: str = Field(..., description="The chunk text content")
    token_count: int = Field(
        ..., ge=1, description="Estimated token count"
    )
    section: Optional[str] = Field(
        None, description="Document section heading if applicable"
    )
    doc_type: Literal["rfp", "addendum", "spec", "affidavit", "other", "bid_page"] = Field(
        ..., description="Document type"
    )
    addendum_number: Optional[int] = Field(
        None, description="Addendum number if applicable"
    )
    page_start: int = Field(
        ..., ge=1, description="Starting page number (1-based)"
    )
    page_end: int = Field(
        ..., ge=1, description="Ending page number (1-based)"
    )
    bid_id: Optional[str] = Field(
        None, description="Bid ID if associated with a bid folder"
    )
    file_name: str = Field(
        ..., description="Source file name"
    )
    file_hash: str = Field(
        ..., description="SHA-256 hash of source file"
    )
    embedded: bool = Field(
        False, description="Whether the chunk has been embedded"
    )
    embed_text: Optional[str] = Field(
        None, description="Text version used for embedding (context prefix stripped)"
    )
    kind: Literal["prose", "table", "addendum", "heading"] = Field(
        "prose", description="Type of chunk content"
    )
    ordinal: int = Field(
        ..., ge=1, description="Sequential ordinal within the document"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings from chunking process"
    )

    @field_validator("chunk_id")
    @classmethod
    def chunk_id_must_be_sha256(cls, v: str) -> str:
        """Validate chunk_id is a sha256 hex string."""
        if not re.match(r"^[a-f0-9]{32}$", v):
            raise ValueError("chunk_id must be a 32-character sha256 hex string")
        return v


class Evidence(BaseModel):
    """A piece of evidence supporting an extracted field."""
    citation_ids: List[str] = Field(
        ..., description="Chunk IDs that support this evidence"
    )
    quote: str = Field(..., description="Direct quote from the source text")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in this evidence (0-1)"
    )
    supports: Literal["supports", "partial", "contradicts", "not_stated"] = Field(
        "supports", description="How the evidence relates to the claim"
    )
    notes: Optional[str] = Field(
        None, description="Additional notes about the evidence"
    )
    field_name: str = Field(
        ..., description="The field this evidence supports"
    )


# ============================================================
# Field specification schemas
# ============================================================


class FieldSpec(BaseModel):
    """A field specification from fields.yaml defining what to extract."""
    name: str = Field(..., description="Field name (e.g., 'due_date', 'contract_price')")
    description: str = Field(
        ..., description="Human-readable description of the field"
    )
    doc_types: List[Literal["rfp", "addendum", "spec", "affidavit", "bid_page"]] = Field(
        ..., description="Document types this field applies to"
    )
    required: bool = Field(
        False, description="Whether this field must be found"
    )
    value_type: Literal["date", "currency", "percentage", "text", "yes_no", "multi"] = Field(
        "text", description="Expected value type"
    )
    format_pattern: Optional[str] = Field(
        None, description="Regex pattern the value must match"
    )
    soft: bool = Field(
        False, description="Soft check (fail gracefully vs hard error)"
    )
    not_found_wording: Optional[List[str]] = Field(
        None, description="Acceptable 'not found' wording variations"
    )
    evidence_required: int = Field(
        1, ge=0, description="Minimum number of evidence chunks required"
    )


class SynonymGroup(BaseModel):
    """A group of synonyms for query expansion."""
    term: str = Field(..., description="Primary term")
    synonyms: List[str] = Field(
        default_factory=list, description="List of synonyms/alternatives"
    )
    doc_types: List[Literal["rfp", "addendum", "spec", "affidavit", "bid_page"]] = Field(
        ..., description="Document types this applies to"
    )


# ============================================================
# Query rewrite schema
# ============================================================


class QueryRewrite(BaseModel):
    """Result of query rewriting/expansion."""
    original: str = Field(..., description="Original user query")
    expanded: str = Field(..., description="Expanded query with synonyms/model terms")
    model_terms: List[str] = Field(
        default_factory=list, description="Additional model-specific terms added"
    )
    soft_suffixes: List[str] = Field(
        default_factory=list, description="Soft suffixes like 'rfp', 'addendum'"
    )
    hard_suffixes: List[str] = Field(
        default_factory=list, description="Hard suffixes like bid ids"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the rewrite quality"
    )


# Type aliases and additional schemas
DocType = Literal["rfp", "addendum", "spec", "affidavit", "other", "bid_page"]


# Additional data schemas used across the codebase


class DetectionResult(BaseModel):
    """Result of detecting document type and metadata."""
    file_kind: Literal["pdf", "html", "text", "other"] = Field(
        ..., description="Type of file detected"
    )
    doc_type: DocType = Field(..., description="Detected document type")
    addendum_number: Optional[int] = Field(
        None, description="Addendum number if applicable"
    )
    doc_date: Optional[datetime] = Field(
        None, description="Document date if detectable"
    )
    title: Optional[str] = Field(
        None, description="Title from first heading/metadata"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Warnings from detection"
    )


class FileReport(BaseModel):
    """Report for a single file during ingestion."""
    file: str = Field(..., description="File name")
    status: str = Field(..., description="Processing status: ok, partial, failed, skipped")
    pages: int = Field(
        ..., ge=0, description="Number of pages"
    )
    tables: int = Field(
        0, ge=0, description="Number of tables"
    )
    empty_pages: int = Field(
        0, ge=0, description="Number of empty pages"
    )
    needs_ocr_pages: int = Field(
        0, ge=0, description="Number of pages needing OCR"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Warnings from processing"
    )
    error: Optional[str] = Field(
        None, description="Error message if failed"
    )


class IngestionReport(BaseModel):
    """Report for the entire ingestion run."""
    bid_id: str = Field(..., description="Bid ID")
    files: List[FileReport] = Field(
        default_factory=list, description="Per-file reports"
    )
    started_at: datetime = Field(
        ..., description="Start time of ingestion"
    )
    finished_at: datetime = Field(
        ..., description="End time of ingestion"
    )
    ok_count: int = Field(
        ..., ge=0, description="Number of successfully processed files"
    )
    partial_count: int = Field(
        ..., ge=0, description="Number of partially processed files"
    )
    failed_count: int = Field(
        ..., ge=0, description="Number of failed files"
    )
    skipped_count: int = Field(
        ..., ge=0, description="Number of skipped files"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Warnings from the ingestion process"
    )