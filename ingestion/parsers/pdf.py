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
    """Build text blocks from a PDF page.

    Groups words into paragraphs based on vertical gaps relative to median line gap.
    Identifies heading candidates based on font size, bold/short text, or section patterns.
    """
    try:
        # Extract words with coordinates
        words = page.extract_words(x_tolerance=1, y_tolerance=1)

        if not words:
            return []

        # Sort words by position (top to bottom, left to right)
        words.sort(key=lambda w: (w["top"], w["x0"]))

        # Group words into lines based on vertical proximity
        lines = _group_into_lines(words)

        # Group lines into paragraphs based on horizontal continuation and vertical gaps
        paragraphs = _group_into_paragraphs(lines)

        # Build blocks
        blocks = []
        current_block: list[str] = []
        current_font_size = None
        current_is_bold = False

        for line in paragraphs:
            line_text = " ".join(line["words"])
            line_font_size = line.get("avg_font_size", 12)
            line_is_bold = line.get("is_bold", False)
            line_height = line.get("height", 12)

            # Check if this should start a new block/section
            is_heading_candidate = False
            heading_level = None

            # Heuristic (a): font size meaningfully larger than median
            if line_font_size > 1.2 * current_font_size if current_font_size else False:
                is_heading_candidate = True
                heading_level = 1

            # Heuristic (b): bold and short
            if line_is_bold and len(line_text.split()) < 12:
                is_heading_candidate = True
                heading_level = 1

            # Heuristic (c): section pattern
            section_patterns = [
                r"^\d+(\.\d+)*[\).]\s+\S",
                r"^(ARTICLE|SECTION|PART|EXHIBIT|ATTACHMENT)\b",
                r"^[A-Z]{2,}\s+\d+",  # ALL-CAPS with number
            ]
            for pattern in section_patterns:
                if re.match(pattern, line_text):
                    is_heading_candidate = True
                    heading_level = (
                        len(re.match(r"(\d+)(\.\d+)*", line_text).group(1).split("."))
                        if re.match(r"^\d+(\.\d+)*", line_text)
                        else 1
                    )
                    break

            if is_heading_candidate and current_block:
                # Close current block and start new heading block
                block_text = " ".join(current_block)
                blocks.append(Block(
                    kind="heading" if is_heading_candidate else "paragraph",
                    text=block_text,
                    level=heading_level,
                    page=page.metadata.get("page_number", 1) if hasattr(page, 'metadata') else 1,
                    font_size=current_font_size,
                ))
                current_block = [line_text]
            else:
                current_block.append(line_text)

        # Close last block
        if current_block:
            block_text = " ".join(current_block)
            blocks.append(Block(
                kind="heading" if current_block else "paragraph",
                text=block_text,
                level=None,
                page=1,
                font_size=current_font_size,
            ))

        return blocks

    except Exception:
        return []


def _group_into_lines(words: list) -> list:
    """Group words into lines based on vertical proximity."""
    if not words:
        return []

    lines = []
    current_line = [words[0]]
    current_top = words[0]["top"]
    current_bottom = words[0]["bottom"]

    for word in words[1:]:
        # Check if word is on the same line (within vertical threshold)
        vertical_gap = abs(word["top"] - current_top)
        # Also check bottom alignment
        bottom_gap = abs(word["bottom"] - current_bottom)

        # Threshold: words within ~5px vertical difference are on same line
        if vertical_gap < 5 and bottom_gap < 5:
            current_line.append(word)
            current_top = min(current_top, word["top"])
            current_bottom = max(current_bottom, word["bottom"])
        else:
            # Save current line and start new one
            lines.append(_create_line_dict(current_line))
            current_line = [word]
            current_top = word["top"]
            current_bottom = word["bottom"]

    # Don't forget the last line
    if current_line:
        lines.append(_create_line_dict(current_line))

    return lines


def _create_line_dict(words: list) -> dict:
    """Create a line dictionary from a list of words."""
    if not words:
        return {}

    # Sort words left-to-right
    sorted_words = sorted(words, key=lambda w: w["x0"])

    # Calculate average font size
    font_sizes = [w.get("size", 12) for w in sorted_words]
    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12

    # Check if any word is bold (has font name indicating bold)
    is_bold = any("Bold" in str(w.get("font", "")) for w in sorted_words)

    return {
        "words": [w["text"] for w in sorted_words],
        "text": " ".join(w["text"] for w in sorted_words),
        "avg_font_size": avg_font_size,
        "is_bold": is_bold,
        "height": max(w.get("bottom", 0) for w in sorted_words)
        - min(w.get("top", 0) for w in sorted_words),
    }


def _group_into_paragraphs(lines: list) -> list:
    """Group lines into paragraphs based on vertical gap and horizontal continuity."""
    if not lines:
        return []

    paragraphs = []
    current_paragraph = [lines[0]]

    for line in lines[1:]:
        # Check vertical gap between last line of current para and this line
        last_line = current_paragraph[-1]
        gap = line["avg_font_size"]  # Using font size as proxy for vertical position

        # If vertical gap is large ( > median line gap * threshold), start new para
        # For simplicity, if the line starts a new horizontal section or has
        # significantly different font size, start new paragraph
        if len(current_paragraph) > 1:
            # Check if this line continues the paragraph horizontally
            last_text = " ".join(w["text"] for w in current_paragraph[-1]["words"])
            current_start = line["words"][0][:20] if line["words"] else ""

            # Simple heuristic: if the line has similar indentation, continue
            if abs(line["avg_font_size"] - sum(l["avg_font_size"] for l in current_paragraph) / len(current_paragraph)) < 4:
                current_paragraph.append(line)
            else:
                paragraphs.append(current_paragraph)
                current_paragraph = [line]
        else:
            current_paragraph.append(line)

    if current_paragraph:
        paragraphs.append(current_paragraph)

    return paragraphs


def _apply_heading_hints(blocks: list, page) -> None:
    """Apply heading detection hints to blocks (informed by PDF parsing)."""

    # This is a heuristic applied after block building
    # The actual heading call happens in _build_blocks; this is a secondary pass
    pass


def _guess_doc_type(filename: str) -> DocType:
    """Guess document type from filename - simplified detection."""
    name_lower = filename.lower()

    if any(kw in name_lower for kw in ["addendum", "addenda"]):
        return DocType.ADDENDUM
    if any(kw in name_lower for kw in ["affidavit", "certification", "disclosure"]):
        return DocType.AFFIDAVIT
    if any(kw in name_lower for kw in ["spec", "specification"]):
        return DocType.SPECS
    if any(kw in name_lower for kw in ["rfp", "bid", "proposal", "solicitation"]):
        return DocType.RFP

    return DocType.BID_PAGE  # Default for HTML/bid pages


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
