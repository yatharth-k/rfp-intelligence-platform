"""Service 10: Query Rewriter.

Implements query expansion and reformulation for improved retrieval.
Supplies reformulated queries, query expansion with entity extraction,
and query rewrite log tracking. Uses the LLM gateway for reformulation
when enabled, with deterministic fallback patterns.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from schemas import Chunk, DocType


class QueryRewriteError(Exception):
    """Raised when query rewriting fails."""


class QueryRewriter:
    """Rewrites and expands search queries for better retrieval.

    Modes:
    - deterministic: Apply rule-based expansion (suffixes, synonyms)
    - llm: Use the LLM gateway for reformulation
    - hybrid: Combine deterministic and LLM approaches
    """

    def __init__(self, use_llm: bool = False, llm_gateway=None):
        self.use_llm = use_llm
        self.llm_gateway = llm_gateway

    def rewrite(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Rewrite a search query for improved retrieval.

        Args:
            query: The original user query
            context: Optional context (bid_id, doc_type, etc.)

        Returns:
            Rewritten query string
        """
        if not query or not query.strip():
            return query

        # Step 1: Deterministic expansion
        expanded = self._deterministic_rewrite(query)

        # Step 2: Optionally apply LLM reformulation
        if self.use_llm and self.llm_gateway:
            try:
                rewritten = self._llm_rewrite(expanded, context)
                if rewritten and rewritten.strip():
                    return rewritten.strip()
            except Exception:
                # Fall back to deterministic if LLM fails
                pass

        return expanded

    def _deterministic_rewrite(self, query: str) -> str:
        """Apply rule-based query expansion.

        Adds common RFP search patterns:
        - Expands model/part numbers (e.g., "7440" -> "Latitude 7440")
        - Adds common RFP suffixes
        - Extracts and reformats key terms
        """
        expanded_terms = []
        terms = query.split()

        for term in terms:
            term = term.strip()
            if not term:
                continue

            # Model/part number pattern: numeric identifiers
            if re.match(r"^\d+$", term):
                # Keep as-is; may be expanded by context
                expanded_terms.append(term)
                continue

            # Check for common RFP prefixes/suffixes to expand
            expanded_term = term

            # Add "RFP" context if looking for documents
            if term.lower() in {"spec", "proposal", "bid", "rfp"}:
                expanded_term = f"{term} request for proposal"

            # Handle hyphenated terms
            if "-" in term:
                expanded_term = f"{term} {term.replace('-', ' ')}"

            # Capitalize for consistency
            expanded_term = expanded_term.capitalize()

            expanded_terms.append(expanded_term)

        result = " ".join(expanded_terms)
        return result if result else query

    def _llm_rewrite(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Use the LLM gateway to rewrite the query.

        Passes the query and context to the LLM for reformulation.
        The LLM is asked to expand the query for better retrieval
        while preserving the original intent.
        """
        if self.llm_gateway is None:
            raise QueryRewriteError("No LLM gateway available for query rewriting.")

        # Build the prompt for query rewriting
        context_str = ""
        if context:
            context_parts = []
            if context.get("bid_id"):
                context_parts.append(f"bid_id: {context['bid_id']}")
            if context.get("doc_type"):
                context_parts.append(f"doc_type: {context['doc_type']}")
            if context.get("addendum_number"):
                context_parts.append(f"addendum: {context['addendum_number']}")
            context_str = " | ".join(context_parts)

        prompt = f"""Rewrite this search query for an RFP intelligence platform to improve retrieval accuracy.

Original query: "{query}"
{context_str if context_str else ""}

The rewritten query should:
- Expand abbreviations and partial model/part numbers
- Add relevant RFP/document context terms
- Preserve the core search intent
- Be concise (max 200 characters)

Return only the rewritten query, no explanation.
"""

        try:
            result = self.llm_gateway.complete(
                prompt=prompt,
                temperature=0,
                max_tokens=300,
            )

            # Extract the rewritten query from the response
            rewritten = result.text or ""
            # Remove any leftover prompt artifacts
            rewritten = re.sub(r'^Original query:.*?\n', '', rewritten)
            rewritten = re.sub(r'The rewritten query should:.*', '', rewritten)
            rewritten = rewritten.strip()

            return rewritten if rewritten else query
        except Exception:
            raise QueryRewriteError("LLM query rewriting failed")

    def expand_query(self, query: str, add_entities: bool = True) -> str:
        """Expand a query with additional entities and synonyms.

        Args:
            query: The original query
            add_entities: Whether to add detected entities (model numbers, etc.)

        Returns:
            Expanded query string
        """
        expanded = [query.strip()]

        if add_entities:
            # Extract potential entity patterns
            # Model numbers like JA-207652, Latitude 7440, etc.
            entity_patterns = re.findall(r"[A-Z]{2,}-\d+[A-Z]?", query)
            for entity in entity_patterns:
                # Add the entity as an additional search term
                expanded.append(entity)

            # Add common RFP synonyms
            synonyms = {
                "price": "cost price amount",
                "delivery": "shipping lead time",
                "quality": "performance specification",
            }
            for word, synonym in synonyms.items():
                if word in query.lower():
                    expanded.append(synonym)

        return " ".join(set(expanded))  # Remove duplicates


class QueryRewriteLog:
    """Simple log for tracking query rewrites.

    Useful for evaluating the effectiveness of different rewrite strategies
    over time.
    """

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        self.entries: List[Dict[str, Any]] = []
        if log_path:
            self._load()

    def _load(self) -> None:
        """Load existing log entries from disk."""
        try:
            import json
            if self.log_path and __import__("os").path.exists(self.log_path):
                with open(self.log_path, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
        except Exception:
            self.entries = []

    def record(self, original: str, rewritten: str, success: bool,
               context: Optional[Dict[str, Any]] = None) -> None:
        """Record a query rewrite attempt."""
        entry = {
            "original": original,
            "rewritten": rewritten,
            "success": success,
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        }
        if context:
            entry["context"] = context
        self.entries.append(entry)
        self._save()

    def _save(self) -> None:
        """Save log entries to disk."""
        if self.log_path:
            try:
                import json
                # Read existing, merge, write back
                existing = []
                if __import__("os").path.exists(self.log_path):
                    with open(self.log_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                existing.extend(self.entries)
                with open(self.log_path, "w", encoding="utf-8") as f:
                    json.dump(existing, f, indent=2)
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from the rewrite log."""
        if not self.entries:
            return {"total": 0, "success_rate": 0}

        total = len(self.entries)
        successful = sum(1 for e in self.entries if e.get("success"))
        return {
            "total": total,
            "success_rate": successful / total if total > 0 else 0,
        }
