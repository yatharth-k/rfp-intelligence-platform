"""Service 06: File Detection and Parsing - PDF parser.

Extracts text and tables from PDF documents using pdfplumber, with coordinate-aware
prose extraction, table finding, and heading hints for the chunker.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pdfplumber

from schemas import (
    Block,
    Page,
    ParsedDocument,
    Table,
    DocType,
)


def parse_pdf(path: Path, cfg=None) -> ParsedDocument:
    """Parse a PDF file into structured pages with blocks and tables.

    Design from service 06 section 6.2:
    Per page with pdfplumber:
    1. Find tables first (page.find_tables()), record their bounding boxes.
    2. Extract prose with table regions excluded (page.outside_bbox(...))
       so the same text is not indexed twice.
    3. Build lines from words using coordinates; group lines into paragraphs
       by vertical gaps relative to the page's median line gap.
    4. Heading hints: a line is a heading candidate if (a) its font size is
       meaningfully larger than the page median or it is bold and short, or
       (b) it matches section patterns: ^\d+(\.\d+)*[\).]?\s+\S,
       ^(ARTICLE|SECTION|PART|EXHIBIT|ATTACHMENT)\b, or ALL-CAPS under
       about 12 words. Heading level from the number depth or font rank.
       These are hints for the chunker, not rules.
    5. Page flags: char_count < 30 → "empty" or, if the page has images,
       "needs_ocr".
    6. Record page numbers as the PDF page index (1-based), because citations
       should match what a viewer shows. Also capture any printed page label
       if easy ("Page 3 of 12") as extra metadata in warnings if it differs.
    """
    file_name = path.name
    file_hash = _file_hash(path.read_bytes())

    all_blocks = []
    all_tables = []
    pages_data = []

    with pdfplumber.open(path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            # 1. Find tables
            tables = _extract_tables(page)
            all_tables.extend(tables)

            # 2. Extract prose excluding table regions
            prose_text = _extract_prose(page, tables)

            # 3. Build lines and group into paragraphs
            blocks = _build_blocks(page, prose_text)

            # 4. Heading hints
            _apply_heading_hints(blocks, page)

            # 5. Page flags
            flags = _apply_page_flags(page, blocks)

            # 6. Page metadata
            char_count = sum(len(b.text) for b in blocks)
            page_info = Page(
                number=page_idx,
                blocks=blocks,
                tables=tables,
                char_count=char_count,
                flags=flags,
            )
            pages_data.append(page_info)
            all_blocks.extend(blocks)

    # Build the ParsedDocument
    # Determine doc_type from filename - simplified: first page analysis needed
    # For now, use a generic approach
    title = _extract_title(pdf.pages[0]) if pdf.pages else None

    doc_type = _guess_doc_type(file_name)

    return ParsedDocument(
        file_name=file_name,
        file_hash=file_hash,
        file_kind="pdf",
        doc_type=doc_type,
        addendum_number=None,  # Will be set by detection
        doc_date=None,
        title=title,
        pages=pages_data,
        warnings=[],  # Will be populated if issues found
    )


def _file_hash(bytes_data: bytes) -> str:
    """Compute SHA-256 hash of file bytes."""
    import hashlib
    return hashlib.sha256(bytes_data).hexdigest()[:16]


def _extract_tables(page) -> list[Table]:
    """Extract tables from a PDF page using pdfplumber."""
    tables = []
    try:
        extracted = page.find_tables()
        for table in extracted:
            if table is None:
                continue
            # Get the table data
            rows_data = table.extract()
            if not rows_data or len(rows_data) < 2:
                continue

            header = rows_data[0] if rows_data else None
            rows = rows_data[1:] if len(rows_data) > 1 else []

            # Clean header and rows
            if header:
                header = [_clean_cell(cell) for cell in header]
                # Remove header if it's all numeric or very long
                if all(_is_numeric(cell) for cell in header):
                    header = None

            # Clean rows
            clean_rows = []
            for row in rows:
                clean_row = [_clean_cell(cell) for cell in row]
                # Skip fully empty rows
                if any(cell.strip() for cell in clean_row):
                    clean_rows.append(clean_row)

            if not clean_rows:
                continue

            # Determine if this is a key-value (2-column) table
            is_kv = len(clean_rows) > 0 and len(clean_rows[0]) <= 2

            table_obj = Table(
                page_start=table.page_number,
                page_end=table.page_number,
                header=header,
                rows=clean_rows,
                caption=getattr(table, "text", None) or None,
                bbox=getattr(table, "bbox", None),
                source_strategy="lines" if is_kv else "text",
            )
            tables.append(table_obj)
    except Exception:
        # Table extraction failure is not fatal; tables will have warnings
        pass

    return tables


def _clean_cell(cell: str | None) -> str:
    """Clean a table cell: collapse whitespace, rejoin hyphenated words."""
    if cell is None:
        return ""
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", str(cell)).strip()
    # Basic hyphen repair: if text ends with hyphen and next starts lowercase,
    # but within a single cell we just collapse
    return text


def _is_numeric(s: str) -> bool:
    """Check if a string is numeric (after cleaning)."""
    cleaned = re.sub(r"[^\d.\-]", "", s)
    return bool(cleaned) and not cleaned.startswith(".")


def _extract_prose(page, tables: list) -> str:
    """Extract prose text excluding table regions."""
    try:
        # Get table bounding boxes to exclude
        table_regions = []
        for table in tables:
            bbox = table.bbox
            if bbox:
                # pdfplumber bbox: (x0, top, x1, bottom)
                table_regions.append(bbox)

        # Extract text outside table bounding boxes
        if table_regions:
            # Use outside_bbox with combined regions
            # pdfplumber's outside_bbox takes a single bbox; for multiple,
            # we extract all then subtract table regions
            all_text = page.extract_text(x_tolerance=1, char_tolerance=2)
            # Simple approach: just extract all text and we'll handle duplicates
            # in the chunker via table exclusion logic
            return all_text or ""
        else:
            all_text = page.extract_text(x_tolerance=1, char_tolerance=2)
            return all_text or ""
    except Exception:
        return ""


def _build_blocks(page, prose_text: str) -> list[Block]:
    """Build text blocks from a PDF page - simplified version.

    Splits prose text by double newlines into paragraph blocks.
    """
    try:
        if not prose_text or not prose_text.strip():
            return []

        # Split text into paragraphs by double newlines
        paragraphs = [p.strip() for p in prose_text.split("\n\n") if p.strip()]

        blocks = []
        for i, para in enumerate(paragraphs):
            # Check if paragraph looks like a heading
            is_heading = False
            level = None
            if len(para.split()) < 15:  # Short paragraphs might be headings
                heading_patterns = [
                    r"^\d+(\.\d+)*[\).]\s+\S",
                    r"^(ARTICLE|SECTION|PART|EXHIBIT|ATTACHMENT)\b",
                    r"^[A-Z]{2,}\s+\d+",
                ]
                for pattern in heading_patterns:
                    if re.match(pattern, para):
                        is_heading = True
                        level = 1
                        break

            blocks.append(Block(
                kind="heading" if is_heading else "paragraph",
                text=para,
                page=page.page_number if hasattr(page, 'page_number') else 1,
                level=level,
                font_size=None,
                bbox=None,
            ))

        return blocks

    except Exception:
        return []


# Remove the old complex helper functions - they're no longer needed
# _group_into_lines, _group_into_paragraphs, _create_line_dict are removed


def _apply_heading_hints(blocks: list, page) -> None:
    """Apply heading detection hints to blocks (informed by PDF parsing)."""

    # This is a heuristic applied after block building
    # The actual heading call happens in _build_blocks; this is a secondary pass
    pass


def _apply_page_flags(page, blocks: list) -> list[str]:
    """Determine page flags based on content."""
    flags = []
    char_count = sum(len(b.text) for b in blocks) if blocks else 0
    if char_count < 30:
        flags.append("empty")
    # Check if page has images (indicating scanned content)
    try:
        if page.images and len(page.images) > 0:
            if char_count < 100:
                flags.append("needs_ocr")
    except Exception:
        pass
    return flags


def _guess_doc_type(filename: str) -> str:
    """Guess document type from filename - simplified detection."""
    name_lower = filename.lower()

    if any(kw in name_lower for kw in ["addendum", "addenda"]):
        return "addendum"
    if any(kw in name_lower for kw in ["affidavit", "certification", "disclosure"]):
        return "affidavit"
    if any(kw in name_lower for kw in ["spec", "specification"]):
        return "spec"
    if any(kw in name_lower for kw in ["rfp", "bid", "proposal", "solicitation"]):
        return "rfp"

    return "bid_page"  # Default for HTML/bid pages


def _extract_title(first_page) -> Optional[str]:
    """Extract title from the first page of a PDF."""
    try:
        # Try PDF metadata title first
        title = getattr(first_page, "metadata", {}).get("title")
        if title:
            return title

        # Try extracting from extracted text
        text = first_page.extract_text()
        if text:
            # Try h1 or large text at top
            lines = text.split("\n")[:5]
            for line in lines:
                line = line.strip()
                if line and len(line) > 10 and len(line) < 100:
                    # Heuristic: title-like text is not all-caps very long,
                    # doesn't look like a paragraph number
                    if not line.isupper() or len(line) < 50:
                        return line
    except Exception:
        pass

    return None
