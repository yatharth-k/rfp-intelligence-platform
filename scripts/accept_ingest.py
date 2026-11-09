import sys, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pdfplumber
try:
    from ingestion.pipeline import ingest_folder
except ImportError:
    from ingestion.report import ingest_folder

# (file-name fragment, doc_type, addendum_number, strings that must appear, min characters)
EXPECT = {
    "Bid1": [
        ("Addendum 1", "addendum", 1, ["addendum no. 1"], 8000),
        ("Addendum 2", "addendum", 2, ["july 9, 2024"], 400),
        ("FINAL.pdf", "rfp", None, ["ja-207652", "dallas isd"], 100000),
        (".html", "bid_page", None, [], 1000),
    ],
    "Bid2": [
        ("Contract_Affidavit", "affidavit", None, ["authority"], 5000),
        ("Mercury_Affidavit", "affidavit", None, ["mercury"], 1000),
        ("Dell_Laptop_Specs", "spec", None, ["210-blyz", "latitude 5550"], 5000),
        ("PORFP", "rfp", None, ["e20p4600040"], 4000),
        (".html", "bid_page", None, [], 1000),
    ],
}

def norm(s):
    return re.sub(r"\s+", " ", s.lower())

fails = []
for bid, rules in EXPECT.items():
    folder = ROOT / "data" / bid
    on_disk = sorted(p.name for p in folder.iterdir() if p.is_file())
    docs, _ = ingest_folder(folder)
    by_name = {d.file_name: d for d in docs}
    print(f"\n##### {bid}: {len(docs)} documents for {len(on_disk)} files")
    for name in on_disk:
        d = by_name.get(name)
        rule = next((r for r in rules if r[0] in name), None)
        if d is None:
            fails.append(f"{name}: no document produced"); print(f"  MISSING  {name}"); continue
        text = " ".join(b.text for p in d.pages for b in p.blocks)
        text += " " + " ".join(str(c) for p in d.pages for t in p.tables for r in t.rows for c in r)
        n = norm(text)
        real_pages = None
        if name.lower().endswith(".pdf"):
            with pdfplumber.open(folder / name) as pdf:
                real_pages = len(pdf.pages)
        problems = []
        if rule:
            _, want_type, want_num, must, min_chars = rule
            if d.doc_type != want_type: problems.append(f"type {d.doc_type}, want {want_type}")
            if want_num is not None and d.addendum_number != want_num:
                problems.append(f"addendum {d.addendum_number}, want {want_num}")
            if len(text) < min_chars: problems.append(f"{len(text)} chars, want >= {min_chars}")
            for m in must:
                if m not in n: problems.append(f"missing text '{m}'")
        if real_pages and len(d.pages) != real_pages:
            problems.append(f"{len(d.pages)} pages parsed, PDF has {real_pages}")
        tables = sum(len(p.tables) for p in d.pages)
        print(f"  {'FAIL' if problems else 'ok  '}  {name[:50]:50s} type={d.doc_type} add={d.addendum_number} "
              f"pages={len(d.pages)}/{real_pages} chars={len(text)} tables={tables}")
        for pr in problems: print(f"         - {pr}"); fails.append(f"{name}: {pr}")

print("\nINGESTION ACCEPTED" if not fails else f"\n{len(fails)} FAILURES")
sys.exit(1 if fails else 0)
