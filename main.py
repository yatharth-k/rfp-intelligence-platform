#!/usr/bin/env python3
"""Service 19: CLI and API entry point for the RFP Intelligence Platform.

Provides commands for:
- index: Index bid folders
- extract: Extract fields from documents
- ask: Ask questions about bids
- qa_log: View Q&A log
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import typer

from schemas import DocType
from ingestion.detect import detect
from ingestion.report import ingest_folder
from search.chunker import chunk_document, estimate_tokens
from search.indexer import Indexer
from agents.orchestrator import Orchestrator
from agents.ingestion_agent import run_ingestion, _decide_proceed
from agents.retrieval_agent import RetrievalRequest, retrieve, retrieve_for_fields
from agents.extraction import run_extraction, GroupExtraction
from agents.validator import validate_fields
from agents.qa import (
    classify_question,
    resolve_bids,
    answer_question,
    build_final_record,
    _generate_bid_summary,
    append_qa_log,
    QuestionType,
)
from agents.qa import QAAnswer, FinalBidRecord


app = typer.Typer(help="RFP Intelligence Platform CLI")


@app.command("index")
def cmd_index(
    path: str = typer.Argument(..., help="Path to bid folder"),
    bid_id: Optional[str] = typer.Option(None, help="Bid ID (defaults to folder name)"),
    force: bool = typer.Option(False, help="Force reindex even if unchanged"),
) -> None:
    """Index a bid folder."""
    import os
    from pathlib import Path

    folder_path = Path(path)
    if not folder_path.exists():
        typer.echo(f"Error: Folder '{path}' does not exist")
        raise SystemExit(1)

    bid_id = bid_id or folder_path.name

    # Run ingestion
    report = ingest_folder(str(folder_path), bid_id=bid_id)

    # Print summary
    typer.echo(f"Bid '{bid_id}': {report.files_indexed} files indexed")
    typer.echo(f"  - OK: {report.ok_count}")
    typer.echo(f"  - Partial: {report.partial_count}")
    typer.echo(f"  - Failed: {report.failed_count}")
    typer.echo(f"  - Skipped: {report.skipped_count}")

    if report.warnings:
        typer.echo("Warnings:")
        for w in report.warnings[:5]:
            typer.echo(f"  - {w}")
        if len(report.warnings) > 5:
            typer.echo(f"  ... and {len(report.warnings) - 5} more")


@app.command("extract")
def cmd_extract(
    bid_id: str = typer.Argument(..., help="Bid ID"),
    field: Optional[str] = typer.Option(None, help="Specific field to extract"),
) -> None:
    """Extract fields from a bid."""
    from agents.extraction import extract_group, run_extraction, GroupExtraction

    # Run extraction
    result = run_extraction(bid_id=bid_id)

    typer.echo(f"Extraction results for bid '{bid_id}':")
    typer.echo(f"  Status: {result.status}")
    typer.echo(f"  Fields found: {len(result.fields)}")
    typer.echo(f"  Validation: passed={result.validation.passed}, failed={result.validation.failed}, not_found={result.validation.not_found}")

    if field:
        if field in result.fields:
            fy = result.fields[field]
            typer.echo(f"  {field}: {fy.value} (confidence={fy.confidence:.2f}, sources={fy.sources})")
        else:
            typer.echo(f"  Field '{field}' not found")


@app.command("ask")
def cmd_ask(
    bid_id: str = typer.Argument(..., help="Bid ID"),
    question: str = typer.Argument(..., help="Question to ask"),
    mode: str = typer.Option("fusion", help="Search mode: bm25, vector, fusion, rerank"),
    top_k: int = typer.Option(5, help="Number of top results"),
) -> None:
    """Ask a question about a bid."""
    from agents.qa import build_final_record

    # Build final record
    record = build_final_record(bid_id=bid_id)

    typer.echo(f"Bid '{bid_id}' summary:")
    typer.echo(f"  Summary: {record.summary_text}")
    typer.echo(f"  Validation: passed={record.validation.passed}, failed={record.validation.failed}, not_found={record.validation.not_found}")

    # Classify and answer question
    qtype, resolution = classify_question(question, {"bids_in_scope": [{"bid_id": bid_id}]})
    typer.echo(f"  Question type: {qtype}")
    typer.echo(f"  Resolution: {resolution}")

    # Try to resolve bids
    bids = resolve_bids({"bids": ["Bid1"], "ambiguity": None}, {"bids_in_scope": [{"bid_id": bid_id}]})
    typer.echo(f"  Resolved bids: {bids}")


@app.command("qa_log")
def cmd_qa_log(bid_id: Optional[str] = typer.Option(None, help="Bid ID (show all if omitted)")) -> None:
    """View Q&A log."""
    import json
    from pathlib import Path

    # Look for Q&A log file
    log_path = Path("storage/qa_log.json")
    if not log_path.exists():
        typer.echo("No Q&A log found")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        log = json.load(f)

    # Filter by bid_id if specified
    entries = log
    if bid_id:
        entries = [e for e in log if e.get("bid_id") == bid_id]

    typer.echo(f"Q&A log ({len(entries)} entries):")
    for entry in entries[:20]:
        typer.echo(f"  [{entry.get('bid_id', '?')}] {entry.get('question', '')[:80]}")
        typer.echo(f"    Answer: {entry.get('answer', '')[:80]}...")
        typer.echo(f"    Confidence: {entry.get('confidence', 0):.2f}")
        typer.echo()


if __name__ == "__main__":
    app()