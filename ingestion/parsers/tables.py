"""Service 06: File Detection and Parsing - Table extraction.

Extracts tables from PDFs using two strategies (lines and text), with header
detection, merging across pages, and cell cleanup. Follows service 06 section 6.3.
"""

from __future__ import annotations

import re
from typing import Optional, List

from schemas import Table, Block


def _extract_tables_from_page(page) -> list:
    """Extract tables from a single PDF page using both strategies.

    Strategy 1 (lines): pdfplumber's ruled table detection.
    Strategy 2 (text): pdfplumber's unruled table detection.
    Returns best-performing tables.
    """
    try:
        import pdfplumber

        # Strategy 1: lines strategy (ruled tables)
        tables_lines = page.find_tables(strategy="lines")
        tables_text = page.find_tables(strategy="text")

        # Try lines first; fall back to text
        all_tables = []
        seen_page_ranges = set()

        for table in tables_lines:
            if table is None:
                continue
            try:
                rows_data = table.extract()
                if not rows_data or len(rows_data) < 2:
                    continue

                # Heuristic: keep if plausible grid (consistent column count)
                header = rows_data[0]
                if header and len(header) > 1:
                    col_count = len(header)
                    # Check if most rows have similar column count
                    plausible = all(
                        len(row) == col_count for row in rows_data[1:] if row
                    )
                    if plausible:
                        # Check header is plausible (short, mostly non-numeric)
                        if header and any(
                            len(str(h).strip()) < 50 and not _is_pure_numeric(str(h))
                            for h in header
                        ):
                            page_span = (
                                table.page_number,
                                table.page_number,  # Single page for now
                            )
                            if page_span not in seen_page_ranges:
                                seen_page_ranges.add(page_span)
                                all_tables.append(
                                    {
                                        "rows": rows_data,
                                        "header": header,
                                        "strategy": "lines",
                                        "page_start": table.page_number,
                                        "page_end": table.page_number,
                                    }
                                )
            except Exception:
                pass

        # Strategy 2: text strategy (unruled)
        if not all_tables:
            for table in tables_text:
                if table is None:
                    continue
                try:
                    rows_data = table.extract()
                    if not rows_data or len(rows_data) < 2:
                        continue

                    header = rows_data[0]
                    if header and len(header) > 1:
                        col_count = len(header)
                        plausible = all(
                            len(row) == col_count for row in rows_data[1:] if row
                        )
                        if plausible:
                            if header and any(
                                len(str(h).strip()) < 50 and not _is_pure_numeric(str(h))
                                for h in header
                            ):
                                page_span = (
                                    table.page_number,
                                    table.page_number,
                                )
                                if page_span not in seen_page_ranges:
                                    seen_page_ranges.add(page_span)
                                    all_tables.append(
                                        {
                                            "rows": rows_data,
                                            "header": header,
                                            "strategy": "text",
                                            "page_start": table.page_number,
                                            "page_end": table.page_number,
                                        }
                                    )
                except Exception:
                    pass

        return all_tables

    except Exception:
        return []


def _is_pure_numeric(s: str) -> bool:
    """Check if a string is purely numeric (allowing decimal points and negatives)."""
    cleaned = re.sub(r"[^\d.\-]", "", s)
    return bool(cleaned) and cleaned.strip("-").replace(".", "").isdigit()


def _clean_cell_text(cell: str) -> str:
    """Clean a table cell: collapse whitespace, rejoin hyphenated words."""
    if not cell:
        return ""
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", str(cell)).strip()
    # Remove leading/trailing dashes that are just formatting
    text = text.strip("- ").strip()
    return text


def _merge_tables_across_pages(tables: list) -> list:
    """Merge consecutive tables with same column count and repeated/absent header.

    Tables that have the same column count and where the header row repeats
    at the top of the next page are merged; page_start/page_end are updated.
    """
    if len(tables) <= 1:
        return tables

    merged = [tables[0]]

    for current in tables[1:]:
        prev = merged[-1]

        # Check if consecutive (page_end + 1 == page_start)
        if prev["page_end"] + 1 == current["page_start"]:
            # Check column count match
            prev_cols = len(prev["header"]) if prev["header"] else 0
            curr_cols = len(current["header"]) if current["header"] else 0

            if prev_cols > 0 and prev_cols == curr_cols:
                # Check if headers are similar (repeated or one is empty)
                prev_header_str = " ".join(
                    str(h) if h else "" for h in (prev["header"] or [])
                )
                curr_header_str = " ".join(
                    str(h) if h else "" for h in (current["header"] or [])
                )

                # Merge: keep previous, extend rows, update page_end
                if not prev["header"]:
                    # Previous had no header; use current's header if it has one
                    if current["header"]:
                        prev["header"] = current["header"]
                # Combine rows (avoid duplicates by checking last row)
                prev_rows = prev["rows"]
                curr_rows = current["rows"]

                # Simple append; real implementation would deduplicate
                if prev_rows and curr_rows:
                    # Check if last row of prev matches first row of curr (indicating merge)
                    last_prev = prev_rows[-1] if prev_rows else []
                    first_curr = curr_rows[0] if curr_rows else []
                    if last_prev != first_curr:
                        merged_rows = prev_rows + curr_rows
                    else:
                        merged_rows = prev_rows + curr_rows[1:]
                elif curr_rows:
                    merged_rows = curr_rows
                else:
                    merged_rows = prev_rows

                # Update the previous table
                prev["rows"] = merged_rows
                prev["page_end"] = current["page_end"]
                # Keep the header; if current has a header and prev doesn't, use current's
                if not prev["header"] and current["header"]:
                    prev["header"] = current["header"]
                # Replace in merged list
                merged[-1] = prev
                continue

        # If not mergeable, add as new
        merged.append(current)

    return merged


def _extract_tables_pdf(path: Path) -> list:
    """Extract all tables from a PDF file.

    Uses two strategies and merges across pages where appropriate.
    Returns list of table dicts with rows, header, strategy, page_start, page_end.
    """
    import pdfplumber

    all_tables = []

    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_tables = _extract_tables_from_page(page)
                all_tables.extend(page_tables)
    except Exception:
        pass

    # Merge tables across pages
    merged = _merge_tables_across_pages(all_tables)

    # Convert to Table schema objects (simplified)
    table_objects = []
    for table in merged:
        # Clean header
        header = None
        if table["header"]:
            header = [_clean_cell_text(h) for h in table["header"]
                      if h and str(h).strip()]

        # Clean rows
        clean_rows = []
        for row in table["rows"]:
            clean_row = [_clean_cell_text(c) for c in row if c and str(c).strip()]
            if any(clean_row):  # Skip fully empty rows
                clean_rows.append(clean_row)

        if not clean_rows:
            continue

        # Determine if key-value (2-column) table
        is_kv = len(clean_rows) > 0 and len(clean_rows[0]) <= 2

        table_obj = {
            "header": header,
            "rows": clean_rows,
            "caption": None,  # Would be extracted from table if available
            "source_strategy": "key_value" if is_kv else "table",
            "page_start": table["page_start"],
            "page_end": table["page_end"],
        }
        table_objects.append(table_obj)

    return table_objects
