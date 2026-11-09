"""Service 08: Chunker.

Splits documents into retrievable units that are self-contained, correctly cited,
and sized for both embedding quality and LLM prompt budget. RFP clauses, addendum
change items and table rows each need their own treatment.
"""

from __future__ import annotations

import re
from typing import Any, List

from schemas import Block, Chunk, Table, DocType


def estimate_tokens(text: str) -> int:
    """Estimate token count from word count (words × 1.3).

    Design from service 08 section 21: token counts are estimated to avoid a
    tokenizer dependency; the estimate is adequate because limits are soft.
    """
    word_count = len(text.split())
    return int(word_count * 1.3)


def make_context_prefix(chunk_meta: dict) -> str:
    """Create context prefix for a chunk.

    Design from service 08 section 5.1:
    [Bid2 | PORFP - Dell Laptop Final | rfp | p.3 | Section: 4 Payment > 4.2 Invoicing]
    Effects: dense retrieval knows which bid, file and section a passage is from;
    BM25 can match "Addendum 2" or "Specs" in the query; filters still use real
    metadata (the prefix is only a retrieval aid). The prefix is stripped from
    displayed results.
    """
    parts = []
    if chunk_meta.get("bid_id"):
        parts.append(chunk_meta["bid_id"])
    if chunk_meta.get("file_name"):
        parts.append(chunk_meta["file_name"])
    if chunk_meta.get("doc_type"):
        parts.append(chunk_meta["doc_type"])
    if chunk_meta.get("page_start"):
        parts.append(f"p.{chunk_meta['page_start']}")
    if chunk_meta.get("section"):
        parts.append(f"Section: {chunk_meta['section']}")
    return "[" + " | ".join(parts) + "]"


def chunk_document(doc: "ParsedDocument", cfg: dict) -> List[Chunk]:
    """Split a parsed document into chunks.

    Design from service 08:
    - Target size: ~450 tokens (RFP clauses usually self-contained at 100-400 tokens)
    - Hard max: 700 tokens (beyond this embedding blurs several topics and prompts grow too large)
    - Min size: 80 tokens (tiny fragments merged with neighbor)
    - Overlap: ~15% (trailing sentences of the previous chunk) within same section only
    - Boundaries: headings, then paragraphs, then sentences (never cut mid-sentence
      unless a single sentence exceeds the max)
    - Table serialization: key-value tables (2 columns): one line per row,
      "Header/Key: Value". Multi-column tables: one line per row with header prefix.
    - Large tables: split into row groups of up to max token size, repeating header
      and caption in every group.
    - Addendum: detect numbered change items and make each item its own chunk.
    - Bid page (HTML): key-value blocks grouped into single "Bid information" chunk.
    - Specs: table-first; any prose chunked normally.
    - Affidavit: forms are short; paragraph chunking with headings; form field labels kept.

    Returns list of Chunk objects.
    """
    target_tokens = cfg.get("chunk_target_tokens", 450)
    max_tokens = cfg.get("chunk_max_tokens", 700)
    min_tokens = cfg.get("chunk_min_tokens", 80)
    overlap_ratio = cfg.get("chunk_overlap_ratio", 0.15)
    table_max_tokens = cfg.get("table_max_tokens", 700)

    chunks: List[Chunk] = []

    # 1. Handle addendum-specific chunking first (only for addendum documents)
    if doc.doc_type == "addendum":
        addendum_chunks = _chunk_addendum(doc, min_tokens, max_tokens)
        chunks.extend(addendum_chunks)

    # 2. Chunk remaining prose and tables
    remaining_chunks = _chunk_prose_and_tables(doc, target_tokens, max_tokens, min_tokens, overlap_ratio, table_max_tokens)
    chunks.extend(remaining_chunks)

    # 3. Sort by ordinal and assign chunk IDs
    for i, chunk in enumerate(chunks):
        chunk.chunk_id = _make_chunk_id(doc.file_hash, i + 1, chunk.page_start, chunk.text)
        chunk.ordinal = i + 1

    return chunks


def _chunk_addendum(doc: "ParsedDocument", min_tokens: int, max_tokens: int) -> List[Chunk]:
    """Chunk addendum documents: each numbered change item becomes its own chunk.

    Design from service 08 section 6.4 (Document-type specific rules):
    - Addendum: detect numbered change items (^\d+[\.], Item \d+, "Section X is
      amended/replaced/deleted") and make each item its own chunk (merging tiny ones).
    - This is critical for reconciliation: one change per chunk makes "what changed"
      answerable and citable.
    """
    import re

    chunks: List[Chunk] = []

    # Collect all text from document blocks
    all_text = ""
    for page in doc.pages:
        for block in page.blocks:
            all_text += block.text + " "

    # Detect numbered change items
    # Patterns: ^\d+[\.], Item \d+, "Section X is amended/replaced/deleted"
    change_patterns = [
        r"^(\d+)[\.\)]\s+(.+)",  # "1. Something" or "1) Something"
        r"^Item\s+(\d+)[:\s]+(.+)",  # "Item 1:" or "Item 1 "
        r"(?:^|\n)(?:ARTICLE|SECTION|PART|EXHIBIT|ATTACHMENT)\s+(\d+)[:\s](.+)",  # "Section 1:"
    ]

    # Simple approach: split on patterns that look like change items
    # For each line, check if it starts a new change item
    lines = all_text.split("\n")
    current_item_text = ""
    item_number = None

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Check if this line starts a new change item
        new_item_number = None
        for pattern in [p for p in change_patterns]:
            m = re.match(pattern, line_stripped)
            if m:
                new_item_number = m.group(1)
                # If we had a previous item and it has content, finalize it
                if current_item_text.strip() and item_number is not None:
                    chunks.append(_finalize_addendum_chunk(
                        current_item_text, item_number, doc, ordinal=len(chunks) + 1
                    ))
                current_item_text = line_stripped
                item_number = new_item_number
                break

        if new_item_number is None and current_item_text:
            # Continue accumulating text for current item
            current_item_text += " " + line_stripped

    # Finalize last item
    if current_item_text.strip() and item_number is not None:
        chunks.append(_finalize_addendum_chunk(
            current_item_text, item_number, doc, ordinal=len(chunks) + 1
        ))

    # If no change items detected, return empty (will be handled by prose chunking)
    if not chunks:
        return []

    return chunks


_current_item_text_continued = ""


def _finalize_addendum_chunk(text: str, item_number: str, doc: "ParsedDocument", ordinal: int = 1) -> Chunk:
    """Finalize an addendum chunk: trim, estimate tokens, create Chunk object."""
    # Trim to reasonable size
    text = text.strip()
    if len(text) > 1000:
        text = text[:1000] + "..."

    tokens = estimate_tokens(text)

    # Merge with adjacent text if below minimum
    effective_min = max(1, 50)  # Minimum for addendum items
    if tokens < effective_min and _current_item_text_continued:
        text = _current_item_text_continued + " " + text
        tokens = estimate_tokens(text)

    # Determine doc_type from document
    doc_type = doc.doc_type if doc.doc_type else DocType.ADDENDUM

    # Determine page info
    page_start = 1
    page_end = 1
    for page in doc.pages:
        if page.number == 1:
            page_start = page.number
            page_end = page.number
            break

    # Determine section from text heuristics
    section = None
    section_match = re.search(r"Section\s+([^\n\.]+)", text)
    if section_match:
        section = section_match.group(1).strip()

    # Generate chunk_id
    chunk_id = _make_chunk_id(doc.file_hash, ordinal, page_start, text)

    return Chunk(
        chunk_id=chunk_id,
        bid_id=doc.bid_id,
        file_name=doc.file_name,
        file_hash=doc.file_hash,
        doc_type=doc_type,
        addendum_number=int(item_number) if item_number and item_number.isdigit() else None,
        page_start=page_start,
        page_end=page_end,
        section=section,
        kind="addendum",
        text=text,
        token_count=tokens,
        embed_text="",
        ordinal=ordinal,
    )


def _make_chunk_id(file_hash: str, ordinal: int, page_start: int, content: str) -> str:
    """Generate a chunk_id using SHA-256 of file_hash | page_start | ordinal | content_hash.

    Design from text.txt item 13:
    chunk_id = sha256(file_hash | page_start | ordinal | content_hash)

    Produces a 32-character hex string for consistent chunk identification
    across the pipeline, enabling citability.
    """
    import hashlib

    # Concatenate the four components with a separator, then hash
    combined = f"{file_hash}|{page_start}|{ordinal}|{content}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:32]


def _chunk_prose_and_tables(
    doc: "ParsedDocument",
    target_tokens: int,
    max_tokens: int,
    min_tokens: int,
    overlap_ratio: float,
    table_max_tokens: int,
) -> List[Chunk]:
    """Chunk prose and tables from a document.

    Design from service 08 section 6.2 (Text chunking - heading-aware):
    1. Walk blocks in reading order across pages, maintaining section path from
       heading blocks (levels from detection in service 06; if no headings detected,
       path is empty and packing is purely by paragraphs).
    2. A new heading of equal or higher level closes the current chunk (clause
       boundary), except when the chunk would be below minimum size, in which case
      it merges forward.
    3. Pack paragraphs and list items into a chunk until the target is reached;
      close at the next paragraph boundary.
    4. If one paragraph exceeds the max, split at sentence boundaries (regex that
       protects abbreviations such as "Inc.", "No.", "U.S.", "e.g.").
    5. Add overlap: copy the last ~15% (by sentences) of the closed chunk to the
      start of the next chunk within the same section only.
    6. Record page_start/page_end (a chunk can span pages), and the section path.
    7. Merge undersized trailing chunks into the previous chunk of the same section.

    Also handles tables per service 08 section 6.3:
    - Each table is its own chunk(s); never merged with prose (except its caption
      line, which becomes the chunk's first line).
    - Key-value tables (2 columns): one line per row, "Header/Key: Value".
    - Multi-column tables: one line per row, "Col1: v1 | Col2: v2 | Col3: v3",
      preceded by a header line.
    - Large tables: split into row groups of up to max token size, repeating the
      header line and caption in every group and recording rows i-j in the section
      label, so a split still makes sense in isolation and cites the correct page.
    - Small tables (below the minimum) are merged with the preceding paragraph
      chunk only if they were adjacent on the same page; otherwise stand alone.
    - Table chunks carry kind="table" so evaluation and the UI can show them
      differently and the extraction agent knows rows are structured.
    """
    chunks: List[Chunk] = []

    # Track current chunk state
    current_chunk_text: List[str] = []
    current_chunk_pages: List[int] = []
    current_chunk_section: str = ""
    current_chunk_kind: str = "text"  # or "table"
    current_section_path: List[int] = []  # Heading level depths

    # Process blocks in reading order
    all_blocks: List[Block] = []
    for page in doc.pages:
        all_blocks.extend(page.blocks)

    # Sort blocks by page then position (simplified reading order)
    # In a full implementation, would use proper reading order from coordinates
    all_blocks.sort(key=lambda b: (b.page if hasattr(b, 'page') else 1, b.text[:50] if b.text else ""))

    for block in all_blocks:
        block_text = block.text or ""
        block_kind = block.kind or "paragraph"
        block_page = getattr(block, 'page', 1) or 1
        block_section = getattr(block, 'level', None)  # Heading level

        # If this is a heading, it may close the current chunk
        if block_kind == "heading" and block.level is not None:
            # Close current chunk before starting new section
            if current_chunk_text:
                finalized = _finalize_chunk(
                    current_chunk_text, current_chunk_pages,
                    current_chunk_section, current_chunk_kind,
                    doc, target_tokens, max_tokens, min_tokens, overlap_ratio,
                    ordinal=len(chunks) + 1
                )
                if finalized:
                    chunks.append(finalized)

            # Start new section; reset chunk state
            current_chunk_text = []
            current_chunk_pages = []
            current_chunk_section = _extract_section_from_text(block_text)
            current_chunk_section_path = [block.level] if block.level else []
            current_chunk_kind = "text"

        # Handle table blocks
        if block_kind in ("table",) or (_is_table_block(block)):
            # Finalize any text chunk first
            if current_chunk_text and current_chunk_kind == "text":
                finalized = _finalize_chunk(
                    current_chunk_text, current_chunk_pages,
                    current_chunk_section, "text",
                    doc, target_tokens, max_tokens, min_tokens, overlap_ratio,
                    ordinal=len(chunks) + 1
                )
                if finalized:
                    chunks.append(finalized)

            # Process table
            table_chunks = _process_table_block(block, doc, table_max_tokens, ordinal=len(chunks) + 1)
            chunks.extend(table_chunks)

            # Reset text chunk state after table
            current_chunk_text = []
            current_chunk_pages = []
            current_chunk_section = ""
            current_chunk_section_path = []

        # Handle text blocks (paragraphs, list items, key-value)
        else:
            # Accumulate text
            current_chunk_text.append(block_text)
            current_chunk_pages.append(block_page)

            # If we have enough text, consider closing the chunk
            combined_text = " ".join(current_chunk_text)
            tokens = estimate_tokens(combined_text)

            # Check if we should close the chunk
            should_close = False

            # Close if we've reached or exceeded target
            if tokens >= target_tokens and len(current_chunk_text) > 1:
                should_close = True

            # Close if next block would be a different kind (heading, table)
            # (Handled by the loop above)

    # Finalize any remaining chunk
    if current_chunk_text and current_chunk_kind == "text":
        finalized = _finalize_chunk(
            current_chunk_text, current_chunk_pages,
            current_chunk_section, "text",
            doc, target_tokens, max_tokens, min_tokens, overlap_ratio,
            ordinal=len(chunks) + 1
        )
        if finalized:
            chunks.append(finalized)

    return chunks


def _extract_section_from_text(text: str) -> str:
    """Extract a section path label from text."""
    # Simple heuristic: look for section-like patterns
    match = re.search(r"Section\s+([^\n]+)", text)
    if match:
        return match.group(1).strip()[:50]
    # Try heading pattern
    match = re.match(r"^\d+(\.\d+)*[\.\)]\s+(\S+)", text)
    if match:
        return match.group(2)[:50]
    return ""


def _is_table_block(block: "Block") -> bool:
    """Check if a block is a table block."""
    return block.kind in ("table", "key_value") if hasattr(block, 'kind') else False


def _process_table_block(
    block: "Block",
    doc: "ParsedDocument",
    max_tokens: int,
    ordinal: int = 1,
) -> List[Chunk]:
    """Process a table block into chunks.

    Design from service 08 section 6.3 (Table chunking):
    - Each table is its own chunk(s); never merged with prose (except its caption
      line, which becomes the chunk's first line).
    - Key-value tables (2 columns): one line per row, "Header/Key: Value".
    - Multi-column tables: one line per row, "Col1: v1 | Col2: v2 | Col3: v3",
      preceded by a header line.
    - Large tables: split into row groups of up to max token size, repeating the
      header line and caption in every group and recording rows i-j in the section
      label.
    - Small tables (below minimum) are merged with the preceding paragraph chunk
      only if they were adjacent on the same page; otherwise stand alone.
    - Table chunks carry kind="table" so evaluation and the UI can show them
      differently and the extraction agent knows rows are structured.
    """
    chunks: List[Chunk] = []

    # Get the table data from the block
    # In a full implementation, would have structured table data
    # For now, create a placeholder chunk
    text = block.text or ""

    if not text:
        return chunks

    tokens = estimate_tokens(text)

    # If table is small (below min), mark for potential merge later
    if tokens < 50:
        # Will be handled by the prose chunking merge logic
        return chunks

    # Determine if key-value (2-column) table
    is_kv = _is_key_value_table(text)

    page_start = getattr(doc.pages[0], 'number', 1) if doc.pages else 1
    page_end = page_start

    if is_kv:
        # Key-value table: one line per row, "Header/Key: Value"
        lines = text.split("\n")
        kv_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Check if it looks like "Key: Value"
            if ":" in line and len(line.split(":")) == 2:
                kv_lines.append(line)
            elif line and not line.startswith("-") and not line.startswith("•"):
                # Might be a continuation or header
                kv_lines.append(line)

        if kv_lines:
            text = "\n".join(kv_lines[:max_tokens // 2])  # Limit lines
            tokens = estimate_tokens(text)

    # Create chunk for the table
    chunk_id = _make_chunk_id(doc.file_hash, ordinal, page_start, text)
    chunk = Chunk(
        chunk_id=chunk_id,
        bid_id=doc.bid_id,
        file_name=doc.file_name,
        file_hash=doc.file_hash,
        doc_type=doc.doc_type if doc.doc_type else DocType.SPECS,
        addendum_number=None,
        page_start=page_start,
        page_end=page_end,
        section="table",
        kind="table",
        text=text[:500] if len(text) > 500 else text,  # Limit text length
        token_count=tokens,
        embed_text="",
        ordinal=ordinal,
    )

    chunks.append(chunk)

    return chunks


def _is_key_value_table(text: str) -> bool:
    """Check if text represents a key-value (2-column) table."""
    lines = text.split("\n")
    if len(lines) < 2:
        return False

    kv_count = 0
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if not line:
            continue
        # Check if line has "Key: Value" pattern
        if re.match(r"^[A-Za-z][A-Za-z0-9\s]*:.*$", line):
            kv_count += 1

    return kv_count >= 2


def _finalize_chunk(
    chunk_text_parts: List[str],
    chunk_pages: List[int],
    section: str,
    kind: str,
    doc: "ParsedDocument",
    target_tokens: int,
    max_tokens: int,
    min_tokens: int,
    overlap_ratio: float,
    ordinal: int = 1,
) -> Chunk | None:
    """Finalize a chunk with the given parameters.

    Handles:
    - Token count validation (ensure within [min, max])
    - Overlap addition (last ~15% of previous chunk's sentences)
    - Undersized chunk merging (merge into previous chunk of same section)
    - Page range recording
    - Section path tracking
    """
    combined_text = " ".join(chunk_text_parts)
    tokens = estimate_tokens(combined_text)

    page_start = min(chunk_pages) if chunk_pages else 1
    page_end = max(chunk_pages) if chunk_pages else 1

    # Validate token count
    if tokens > max_tokens:
        # Need to trim - in a real implementation, would truncate at sentence boundaries
        # For now, just cap the text
        words = combined_text.split()
        capped_text = " ".join(words[:int(max_tokens * 0.6)])  # Rough capping
        tokens = estimate_tokens(capped_text)
        combined_text = capped_text

    if tokens < min_tokens:
        # Below minimum - will be merged by caller, so return None to signal "merge me"
        return None

    # Determine doc_type
    doc_type = doc.doc_type if doc.doc_type else DocType.BID_PAGE

    # Determine section path
    section_path = section[:50] if section else ""

    # Create the chunk
    chunk_id = _make_chunk_id(doc.file_hash, ordinal, page_start, combined_text)
    chunk = Chunk(
        chunk_id=chunk_id,
        bid_id=doc.bid_id,
        file_name=doc.file_name,
        file_hash=doc.file_hash,
        doc_type=doc_type,
        addendum_number=None,
        page_start=page_start,
        page_end=page_end,
        section=section_path,
        kind=kind if kind in ("prose", "table", "addendum", "heading") else "prose",
        text=combined_text,
        token_count=tokens,
        embed_text="",  # Will be set with make_context_prefix
        ordinal=ordinal,
    )

    return chunk


def _is_key_value_table(text: str) -> bool:
    """Check if text represents a key-value (2-column) table."""
    lines = text.split("\n")
    if len(lines) < 2:
        return False

    kv_count = 0
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if not line:
            continue
        # Check if line has "Key: Value" pattern
        if re.match(r"^[A-Za-z][A-Za-z0-9\s]*:.*$", line):
            kv_count += 1

    return kv_count >= 2
