"""Service 06: File Detection and Parsing - Detection module.

Detects file type, document type, and addendum number from file paths and content.
Follows service 06 design: extension rules, filename patterns, first-page content cues,
LLM fallback for ambiguous cases.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from schemas import DocType, DetectionResult


# Filename pattern mappings
ADDENDUM_MAP = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
}


def detect(path: Path, first_page_text: str | None = None) -> DetectionResult:
    """Detect file kind, doc type, and addendum number.

    Order of evidence (stop at first confident answer):
    1. Extension: .html/.htm → bid_page; .pdf → continue; other → other
    2. Filename rules (case-insensitive):
       addendum\s*(no\.?|#)?\s*(\d+) → addendum, number captured
       affidavit|certification|disclosure → affidavit
       spec|specification → specs
       rfp|bid|solicitation|invitation|porfp|proposal → rfp
    3. First-page content cues
    4. LLM fallback (only when 1-3 are ambiguous)
    """
    path_str = path.name.lower()

    # 1. Extension check
    if path.suffix.lower() in (".html", ".htm"):
        doc_type = DocType.BID_PAGE
        addendum_number = _extract_addendum_from_name(path_str)
        return DetectionResult(
            file_kind="html",
            doc_type=doc_type,
            addendum_number=addendum_number,
            doc_date=None,
            title=None,
            warnings=[],
        )

    if path.suffix.lower() == ".pdf":
        # Continue to other checks; doc_type determined below
        pass
    else:
        # Unknown extension
        return DetectionResult(
            file_kind="other",
            doc_type=DocType.OTHER,
            addendum_number=None,
            doc_date=None,
            title=None,
            warnings=["Unsupported file extension; classified as 'other'"],
        )

    # 2. Filename rules (case-insensitive)
    doc_type, addendum_number, date_hint = _detect_from_filename(path_str)
    if doc_type != DocType.OTHER:
        return DetectionResult(
            file_kind="pdf",
            doc_type=doc_type,
            addendum_number=addendum_number,
            doc_date=date_hint,
            title=None,
            warnings=[],
        )

    # 3. First-page content cues
    doc_type, addendum_number, date_hint = _detect_from_content(first_page_text)
    if doc_type != DocType.OTHER:
        return DetectionResult(
            file_kind="pdf",
            doc_type=doc_type,
            addendum_number=addendum_number,
            doc_date=date_hint,
            title=None,
            warnings=[],
        )

    # 4. LLM fallback (placeholder - would send first 1500 chars)
    # In a full implementation, this would call the LLM with:
    # - file name and first 1,500 characters
    # - schema: {doc_type, addendum_number|null, confidence}
    # - result cached and logged as fallback event
    return DetectionResult(
        file_kind="pdf",
        doc_type=DocType.OTHER,
        addendum_number=None,
        doc_date=None,
        title=None,
        warnings=["Ambiguous document; LLM fallback needed for classification"],
    )


def _extract_addendum_from_name(name: str) -> Optional[int]:
    """Extract addendum number from filename using known mappings."""
    # Check for "Addendum X" or "Addendum X" patterns
    for word, num in ADDENDUM_MAP.items():
        # Match "Addendum One", "addendum 1", etc.
        patterns = [
            rf"\baddendum\s*{word}\b",
            rf"\baddendum\s*#?\s*{word}\b",
        ]
        for pattern in patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return num
    return None


def _detect_from_filename(name: str) -> tuple:
    """Detect doc type and addendum from filename patterns."""
    # Addendum pattern
    addendum_match = re.search(
        r"addendum\s*(no\.?|#)?\s*(\d+)", name, re.IGNORECASE
    )
    if addendum_match:
        try:
            num = int(addendum_match.group(2))
            # Check for affidavit/spec/rfp keywords alongside
            if re.search(r"\b(affidavit|certification|disclosure)\b", name):
                return DocType.AFFIDAVIT, num, None
            if re.search(r"\b(spec|specification)\b", name):
                return DocType.SPECS, num, None
            if re.search(r"\b(rfp|bid|solicitation|invitation)\b", name):
                return DocType.RFP, num, None
            # Default: addendum number captured but doc type from context
            return DocType.ADDENDUM, num, None
        except ValueError:
            pass

    # Affidavit pattern
    if re.search(r"\b(affidavit|certification|disclosure)\b", name):
        return DocType.AFFIDAVIT, None, None

    # Spec pattern
    if re.search(r"\b(spec|specification)\b", name):
        return DocType.SPECS, None, None

    # RFC pattern
    if re.search(r"\b(rfp|bid|solicitation|invitation|porfp|proposal)\b", name):
        return DocType.RFP, None, None

    return DocType.OTHER, None, None


def _detect_from_content(text: str | None) -> tuple:
    """Detect doc type and addendum from first-page content cues."""
    if not text:
        return DocType.OTHER, None, None

    text_lower = text.lower()

    # Addendum cues
    if re.search(r"addendum\s*no\.?\s*\d+", text_lower) or re.search(
        r"addendum\s*#\s*\d+", text_lower
    ):
        # Extract number
        num_match = re.search(r"(\d+)", text_lower)
        num = int(num_match.group(1)) if num_match else None
        # Check type keywords alongside
        if "rfp" in text_lower or "bid" in text_lower:
            return DocType.RFP, num, None
        if "affidavit" in text_lower or "certification" in text_lower:
            return DocType.AFFIDAVIT, num, None
        if "spec" in text_lower or "specification" in text_lower:
            return DocType.SPECS, num, None
        return DocType.ADDENDUM, num, None

    # Affidavit cues
    if re.search(r"\b(affidavit|certification|disclosure)\b", text_lower):
        return DocType.AFFIDAVIT, None, None

    # Spec cues
    if re.search(r"\b(spec|specification)\b", text_lower):
        return DocType.SPECS, None, None

    # RFC cues
    if re.search(r"\b(rfp|bid|solicitation|invitation)\b", text_lower):
        return DocType.RFP, None, None

    # Date cues
    date_patterns = [
        r"dated\s+",
        r"date:\s*",
        r"issued\s+",
    ]
    for pattern in date_patterns:
        if re.search(pattern, text_lower):
            # Extract potential date
            date_match = re.search(r"(\w+\s+\d+|\d+\s+\w+\s+\d+)", text_lower)
            date_hint = date_match.group(1) if date_match else None
            return DocType.RFP, None, date_hint

    return DocType.OTHER, None, None
