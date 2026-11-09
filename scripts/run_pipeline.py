import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sys, time, inspect, traceback
from pathlib import Path

bid = sys.argv[1] if len(sys.argv) > 1 else "Bid1"
folder = Path(__file__).resolve().parents[1] / "data" / bid
if not folder.exists():
    sys.exit(f"Folder not found: {folder.resolve()}")
debug_dir = Path("storage/debug") / bid
debug_dir.mkdir(parents=True, exist_ok=True)

def fail(stage, e):
    print(f"\n[FAILED at {stage}] {type(e).__name__}: {e}")
    print("".join(traceback.format_exc().splitlines(True)[-10:]))
    sys.exit(1)

# ---------- 1. INGEST ----------
print(f"\n===== 1. INGEST {bid} =====")
t0 = time.time()
try:
    try:
        from ingestion.pipeline import ingest_folder
    except ImportError:
        from ingestion.report import ingest_folder
    docs, report = ingest_folder(folder)
    for d in docs:
        n_tab = sum(len(p.tables) for p in d.pages)
        flags = sorted({f for p in d.pages for f in p.flags})
        print(f"- {d.file_name}\n    type={d.doc_type} addendum={d.addendum_number} date={d.doc_date} "
              f"pages={len(d.pages)} tables={n_tab} flags={flags} warnings={len(d.warnings)}")
        lines = []
        for p in d.pages:
            lines.append(f"\n===== PAGE {p.number} =====")
            lines += [b.text for b in p.blocks]
            for t in p.tables:
                lines.append(f"--- TABLE p{t.page_start} header={t.header} ---")
                lines += [" | ".join(str(c) for c in r) for r in t.rows]
        (debug_dir / (Path(d.file_name).stem[:60] + ".txt")).write_text("\n".join(lines), encoding="utf-8")
    print(f"files parsed: {len(docs)}  ({time.time()-t0:.1f}s)")
    print("report:", report.model_dump() if hasattr(report, "model_dump") else report)
except Exception as e:
    fail("ingest", e)

# ---------- 2. CHUNK ----------
print("\n===== 2. CHUNK =====")
try:
    from search.chunker import chunk_document
    from config.settings import get_settings
    cfg = get_settings()
    total, chunk_lines = 0, []
    for d in docs:
        ch = chunk_document(d, cfg)
        total += len(ch)
        tok = [len(c.text.split()) * 1.3 for c in ch] or [0]
        print(f"- {d.file_name}: {len(ch)} chunks, tokens min/max = {int(min(tok))}/{int(max(tok))}, "
              f"table chunks = {sum(c.kind == 'table' for c in ch)}")
        for c in ch:
            chunk_lines += [f"\n##### {c.chunk_id} | {c.doc_type} | p.{c.page_start} | {c.kind} | {c.section}", c.embed_text]
    (debug_dir / "_chunks.txt").write_text("\n".join(chunk_lines), encoding="utf-8")
    print("total chunks:", total)
except Exception as e:
    try: print("chunk_document signature:", inspect.signature(chunk_document))
    except Exception: pass
    fail("chunk", e)

# ---------- 3. INDEX (calls the embedding API) ----------
print("\n===== 3. INDEX =====")
try:
    from search.indexer import Indexer
    from search.keyword_store import KeywordStore, FileRegistry
    from search.vector_store import QdrantVectorStore
    from core.embeddings import EmbeddingClient
    from core.tracing import Tracer
    from core.limiter import RateLimiter
    from config.settings import get_settings

    cfg = get_settings()

    # Initialize dependencies
    storage_dir = Path("storage")
    storage_dir.mkdir(exist_ok=True)

    keyword_store = KeywordStore(storage_dir / "keyword.sqlite")
    file_registry = FileRegistry(storage_dir / "file_registry.sqlite")
    vector_store = QdrantVectorStore(path=str(storage_dir / "qdrant"), dim=2048)
    embedding_client = EmbeddingClient(
        model=cfg.EMBED_MODEL,
        embed_url=cfg.NVIDIA_BASE_URL,
        api_key=cfg.NVIDIA_EMBED_KEY or cfg.NVIDIA_API_KEY,
        batch_size=cfg.EMBED_BATCH_SIZE,
    )
    tracer = Tracer(run_id=f"index_{bid}")
    rate_limiter = RateLimiter(max_per_minute=cfg.MAX_RPM)

    ix = Indexer(
        keyword_store=keyword_store,
        vector_store=vector_store,
        file_registry=file_registry,
        embedding_client=embedding_client,
        tracer=tracer,
        rate_limiter=rate_limiter,
    )

    # Attach chunks to documents for indexing
    for d in docs:
        d.chunks = chunk_document(d, cfg)

    t = time.time()
    rep = ix.index_documents(docs, bid_id=bid, embed_model=cfg.EMBED_MODEL)
    print(f"done in {time.time()-t:.1f}s:", rep)
except Exception as e:
    try: print("Indexer signature:", inspect.signature(Indexer))
    except Exception: pass
    fail("index", e)

# ---------- 4. SEARCH ----------
print("\n===== 4. SEARCH =====")
try:
    from search.engine import SearchEngine, SearchMode
    from search.keyword_store import KeywordStore
    from search.vector_store import QdrantVectorStore

    # Re-initialize stores for search
    storage_dir = Path("storage")
    keyword_store = KeywordStore(storage_dir / "keyword.sqlite")
    vector_store = QdrantVectorStore(path=str(storage_dir / "qdrant"), dim=2048)

    eng = SearchEngine(keyword_store=keyword_store, vector_store=vector_store)
    for q in ["What is the bid submission deadline?", "bid bond", "What changed in the addendum?"]:
        r = eng.search(q, filters={"bid_ids": [bid]}, top_k=3, mode=SearchMode.FUSION)
        print(f"\nQ: {q}   degraded={r.get('degraded', False)}")
        for x in r.get('results', []):
            print(f"  {x.get('rank', '?')}. {x.get('file_name', 'unknown')} p.{x.get('page_start', '?')} | {x.get('text', '')[:140].replace(chr(10), ' ')}")
except Exception as e:
    try: print("SearchEngine signature:", inspect.signature(SearchEngine))
    except Exception: pass
    fail("search", e)

print("\nALL STAGES PASSED")