"""Service 07: Cleaning, Metadata and Ingestion Pipeline - Ingestion report.

Provides the IngestionReport model and ingest_folder() function that processes
a bid folder through detection, parsing, cleaning, metadata tagging, and report
generation. Follows service 07 design sections 6.3-6.5.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

from schemas import (
    Block,
    IngestionReport,
    FileReport,
    ParsedDocument,
    DocType,
)


def ingest_folder(
    folder: Path,
    *,
    bid_id: str | None = None,
    only_files: List[str] | None = None,
    recursive: bool = False,
) -> Tuple[List[ParsedDocument], IngestionReport]:
    """Process a bid folder: detect→parse→clean→tag→report.

    Design from service 07 section 6.4:
    1. List files (non-recursive by default; --recursive option), ignore hidden
       and temp files.
    2. Sort deterministically (so chunk ordinals and IDs are stable).
    3. For each file: compute hash; if the file registry says it is already
       indexed with the same hash and only_files is in use, mark skipped.
    4. detect → parse → clean → tag.
    5. Save storage/parsed/<bid_id>/<file_name>.json (debugging, offline
       re-chunking without re-parsing).
    6. Return documents and report.

    Returns:
        Tuple of (list[ParsedDocument], IngestionReport)
    """
    # Start time
    started_at = datetime.utcnow()

    # Resolve bid_id
    if bid_id is None:
        bid_id = folder.name

    # Ensure output directory exists
    parsed_dir = Path(f"storage/parsed/{bid_id}")
    parsed_dir.mkdir(parents=True, exist_ok=True)

    # 1. List files
    supported_extensions = {".pdf", ".html", ".htm"}
    files: List[Path] = []

    if recursive:
        for root, _dirs, filenames in os.walk(folder):
            for fname in filenames:
                fpath = Path(root) / fname
                if fpath.suffix.lower() in supported_extensions:
                    # Skip hidden and temp files
                    if fpath.name.startswith(".") or fpath.name.startswith("~"):
                        continue
                    files.append(fpath)
    else:
        for item in folder.iterdir():
            if item.is_file() and item.suffix.lower() in supported_extensions:
                # Skip hidden and temp files
                if item.name.startswith(".") or item.name.startswith("~"):
                    continue
                files.append(item)

    # 2. Sort deterministically
    files.sort(key=lambda p: (p.name.lower(), p.suffix.lower()))

    # 3. Check registry for unchanged files (simplified: no actual registry check
    # in this standalone version; would check file_hash against stored registry)
    # For now, we process all files

    # 4. Process each file
    parsed_docs: List[ParsedDocument] = []
    file_reports: List[FileReport] = []
    files_indexed = 0
    files_skipped_unchanged = 0
    files_failed: List[FileReport] = []
    empty_pages = 0
    needs_ocr_pages = 0

    for filepath in files:
        # Compute file hash
        try:
            file_bytes = filepath.read_bytes()
            file_hash = _file_hash(file_bytes)
        except Exception:
            file_hash = "unknown"
            file_bytes = b""

        # Check if only_files filter matches
        if only_files and filepath.name not in only_files:
            continue

        # 4. Process: detect → parse → clean → tag
        file_status = "ok"
        file_warnings: List[str] = []
        file_empty_pages = 0
        file_needs_ocr = 0

        try:
            # Step 4a: Detect file type
            # In a full implementation, would use the detection module
            # For now, classify by extension
            if filepath.suffix.lower() == ".pdf":
                doc_type = DocType.RFP  # Simplified
                addendum_number = None
            elif filepath.suffix.lower() in (".html", ".htm"):
                doc_type = DocType.BID_PAGE
                addendum_number = None
            else:
                doc_type = DocType.OTHER
                file_status = "skipped"
                file_warnings.append(f"Unsupported extension classified as 'other'")

            if file_status == "ok":
                # Step 4b: Parse
                # Simplified parsing - in full use pdf.py or html.py
                try:
                    parsed_doc = _simple_parse(filepath, doc_type)
                except Exception as e:
                    parsed_doc = ParsedDocument(
                        file_name=filepath.name,
                        file_hash=file_hash,
                        doc_type=doc_type,
                        addendum_number=None,
                        doc_date=None,
                        title=None,
                        pages=[],
                        warnings=[f"Parse error: {str(e)}"],
                    )
                    file_status = "failed"

            if file_status == "ok" and parsed_doc:
                # Step 4c: Clean
                try:
                    from core.cleaning import clean_document
                    cleaned_doc = clean_document(parsed_doc)
                except Exception as e:
                    cleaned_doc = parsed_doc
                    file_warnings.append(f"Cleaning error: {str(e)}")

                # Step 4d: Tag metadata
                # bid_id already set; other metadata from detection
                # For now, just ensure title is set
                if not cleaned_doc.title:
                    cleaned_doc.title = _extract_title_from_name(filepath.name)

                # Step 4e: Save parsed JSON
                json_path = parsed_dir / f"{filepath.name}.json"
                try:
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(_doc_to_dict(cleaned_doc), f, indent=2, default=str)
                except Exception as e:
                    file_warnings.append(f"Save error: {str(e)}")

                # Track results
                files_indexed += 1
                parsed_docs.append(cleaned_doc)

                # File report
                fr = FileReport(
                    file=filepath.name,
                    status=file_status,
                    pages=len(cleaned_doc.pages) if cleaned_doc.pages else 0,
                    tables=0
                    if not (cleaned_doc.pages and cleaned_doc.pages[0].tables)
                    else sum([len(t.rows) if t.rows else 0 for t in cleaned_doc.pages[0].tables]),
                    empty_pages=file_empty_pages,
                    needs_ocr_pages=file_needs_ocr,
                    warnings=file_warnings,
                    error=None if file_status != "failed" else "Parsing or cleaning failed",
                )
                file_reports.append(fr)

                # Count pages
                if cleaned_doc.pages:
                    file_page_count = len(cleaned_doc.pages)
                    empty_pages += sum(1 for p in cleaned_doc.pages if p.flags and "empty" in p.flags)
                    needs_ocr_pages += sum(1 for p in cleaned_doc.pages if p.flags and "needs_ocr" in p.flags)

        except Exception as e:
            # File-level failure
            file_status = "failed"
            file_warnings.append(f"File processing error: {str(e)}")
            files_failed.append(
                FileReport(
                    file=filepath.name,
                    status=file_status,
                    pages=0,
                    tables=0,
                    empty_pages=0,
                    needs_ocr_pages=0,
                    warnings=[file_warnings] if file_warnings else [],
                    error=str(e),
                )
            )

    # 6. Build and return ingestion report
    finished_at = datetime.utcnow()

    ok_count = sum(1 for f in file_reports if f.status == "ok")
    partial_count = sum(1 for f in file_reports if f.status == "partial")
    failed_count = sum(1 for f in file_reports if f.status == "failed")
    skipped_count = sum(1 for f in file_reports if f.status == "skipped")

    report = IngestionReport(
        bid_id=bid_id,
        files=file_reports,
        started_at=started_at,
        finished_at=finished_at,
        ok_count=ok_count,
        partial_count=partial_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
    )

    # Add summary to report notes (informal)
    report.warnings = report.warnings or []
    report.warnings.append(
        f"Summary: {ok_count} ok, {partial_count} partial, {failed_count} failed, "
        f"{skipped_count} skipped. "
        f"Total pages: {sum(f.pages for f in file_reports)}, "
        f"Empty: {empty_pages}, Needs OCR: {needs_ocr_pages}"
    )

    return parsed_docs, report


def _simple_parse(filepath: Path, doc_type: DocType) -> ParsedDocument:
    """Simplified parsing for the ingestion pipeline.

    In a full implementation, would use pdf.py or html.py from service 06.
    """
    file_name = filepath.name
    file_hash = _file_hash(filepath.read_bytes())

    # Create a minimal parsed document based on doc type
    if doc_type == DocType.RFP:
        # PDF RFP: minimal blocks
        blocks = [Block(kind="paragraph", text="RFP document placeholder", page=1, level=None)]
    elif doc_type == DocType.BID_PAGE:
        # HTML bid page: minimal blocks
        blocks = [Block(kind="key_value", text="Bid Information: placeholder", page=1, level=None)]
    else:
        blocks = [Block(kind="paragraph", text="Document placeholder", page=1, level=None)]

    page = Page(
        number=1,
        blocks=blocks,
        tables=[],
        char_count=len("placeholder"),
        flags=[],
    )

    return ParsedDocument(
        file_name=file_name,
        file_hash=file_hash,
        doc_type=doc_type,
        addendum_number=None,
        doc_date=None,
        title=_extract_title_from_name(filepath.name),
        pages=[page],
        warnings=[],
    )


def _doc_to_dict(doc: ParsedDocument) -> dict:
    """Convert a ParsedDocument to a JSON-serializable dict."""
    return {
        "file_name": doc.file_name,
        "file_hash": doc.file_hash,
        "doc_type": doc.doc_type.value if hasattr(doc.doc_type, "value") else doc.doc_type,
        "addendum_number": doc.addendum_number,
        "doc_date": str(doc.doc_date) if doc.doc_date else None,
        "title": doc.title,
        "pages": [
            {
                "number": p.number,
                "blocks": [
                    {
                        "kind": b.kind,
                        "text": b.text,
                        "level": b.level,
                        "page": b.page,
                        "font_size": b.font_size,
                        "bbox": list(b.bbox) if b.bbox else None,
                    }
                    for b in p.blocks
                ],
                "tables": [
                    {
                        "page_start": t.page_start,
                        "page_end": t.page_end,
                        "header": t.header,
                        "rows": t.rows,
                        "caption": t.caption,
                        "source_strategy": t.source_strategy,
                        "bbox": list(t.bbox) if t.bbox else None,
                    }
                    for t in p.tables
                ],
                "char_count": p.char_count,
                "flags": p.flags,
            }
            for p in doc.pages
        ],
        "warnings": doc.warnings,
    }


def _extract_title_from_name(filename: str) -> Optional[str]:
    """Extract a title-like string from a filename."""
    # Remove extension
    name = filename.rsplit(".", 1)[0] if "." in filename else filename

    # Replace underscores and hyphens with spaces
    name = name.replace("_", " ").replace("-", " ")

    # Title-case
    title = " ".join(word.capitalize() for word in name.split())

    # Cap length
    if len(title) > 80:
        title = title[:80] + "..."

    return title if title else None


def _file_hash(bytes_data: bytes) -> str:
    """Compute SHA-256 hash of file bytes."""
    import hashlib
    return hashlib.sha256(bytes_data).hexdigest()[:16]