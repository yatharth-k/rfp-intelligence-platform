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

# Doc type strings (lowercase, matching DocType Literal)
DOC_TYPE_RFP = "rfp"
DOC_TYPE_ADDENDUM = "addendum"
DOC_TYPE_SPEC = "spec"
DOC_TYPE_AFFIDAVIT = "affidavit"
DOC_TYPE_OTHER = "other"
DOC_TYPE_BID_PAGE = "bid_page"


def detect(path: Path, first_page_text: str | None = None) -> DetectionResult:
    """Detect file kind, doc type, and addendum number.

    Order of evidence (stop at first confident answer):
    1. Extension: .html/.htm → bid_page; .pdf → continue; other → other
    2. Filename rules (case-insensitive):
       addendum\\s*(no\\.?|#)?\\s*(\\d+) → addendum, number captured
       affidavit|certification|disclosure → affidavit
       spec|specification → specs
       rfp|bid|solicitation|invitation|porfp|proposal → rfp
    3. First-page content cues
    4. LLM fallback (only when 1-3 are ambiguous)
    """
    path_str = path.name.lower()

    # 1. Extension check
    if path.suffix.lower() in (".html", ".htm"):
        doc_type = "bid_page"
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
            doc_type="other",
            addendum_number=None,
            doc_date=None,
            title=None,
            warnings=["Unsupported file extension; classified as 'other'"],
        )

    # 2. Filename rules (case-insensitive)
    doc_type, addendum_number, date_hint = _detect_from_filename(path_str)
    if doc_type != "other":
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
    if doc_type != "other":
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
        doc_type="other",
        addendum_number=None,
        doc_date=None,
        title=None,
        warnings=["Ambiguous document; LLM fallback needed for classification"],
    )


def _extract_addendum_from_name(name: str) -> Optional[int]:
    """Extract addendum number from filename using known mappings."""
    # Check for "Addendum X" or "addendum 1" patterns
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
    # Strip extension for cleaner matching
    name = name.rsplit(".", 1)[0] if "." in name else name
    # Addendum pattern - addendum rule wins over rfp rule
    addendum_match = re.search(
        r"addendum\s*(no\.?|#)?\s*(\d+)", name, re.IGNORECASE
    )
    if addendum_match:
        try:
            num = int(addendum_match.group(2))
            # Check for affidavit/spec keywords alongside (these override addendum)
            if re.search(r"(?:^|[-_\s])(affidavit|certification|disclosure)(?:$|[-_\s])", name):
                return "affidavit", num, None
            if re.search(r"(?:^|[-_\s])(specs?|specification)(?:$|[-_\s])", name):
                return "spec", num, None
            # Addendum rule wins: if addendum number found, classify as addendum
            return "addendum", num, None
        except ValueError:
            pass

    # Affidavit pattern (allow underscore/hyphen as separator)
    if re.search(r"(?:^|[-_\s])(affidavit|certification|disclosure)(?:$|[-_\s])", name):
        return "affidavit", None, None

    # Spec pattern (allow underscore/hyphen as separator)
    if re.search(r"(?:^|[-_\s])(specs?|specification)(?:$|[-_\s])", name):
        return "spec", None, None

    # RFC pattern (allow underscore/hyphen as separator)
    if re.search(r"(?:^|[-_\s])(rfp|bid|solicitation|invitation|porfp)(?:$|[-_\s])", name):
        return "rfp", None, None
    # Bid number pattern (e.g., JA-207652) - case insensitive
    if re.search(r"(?:^|[-_\s])([a-z]{1,4}-\d{5,})(?:$|[-_\s])", name):
        return "rfp", None, None

    return "other", None, None


def _detect_from_content(text: str | None) -> tuple:
    """Detect doc type and addendum from first-page content cues."""
    if not text:
        return "other", None, None

    text_lower = text.lower()

    # Addendum cues - addendum rule wins over rfp rule
    if re.search(r"addendum\s*no\.?\s*\d+", text_lower) or re.search(
        r"addendum\s*#\s*\d+", text_lower
    ):
        # Extract number
        num_match = re.search(r"(\d+)", text_lower)
        num = int(num_match.group(1)) if num_match else None
        # Check type keywords alongside - affidavit/spec override addendum
        if re.search(r"\b(affidavit|certification|disclosure)\b", text_lower):
            return "affidavit", num, None
        if re.search(r"\b(specs?|specification)\b", text_lower):
            return "spec", num, None
        # Addendum rule wins
        return "addendum", num, None

    # Affidavit cues
    if re.search(r"\b(affidavit|certification|disclosure)\b", text_lower):
        return "affidavit", None, None

    # Spec cues
    if re.search(r"\b(specs?|specification)\b", text_lower):
        return "spec", None, None

    # RFC cues
    if re.search(r"\b(rfp|bid|solicitation|invitation|porfp)\b", text_lower):
        return "rfp", None, None

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
            return "rfp", None, date_hint

    return "other", None, None
