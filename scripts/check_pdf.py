import sys
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    sys.exit("pdfplumber is NOT installed in this Python. Run: python -m pip install pdfplumber")

root = Path(__file__).resolve().parents[1]
print("python:", sys.executable)
print("pdfplumber version:", pdfplumber.__version__)
for bid in ("Bid1", "Bid2"):
    pdfs = sorted((root / "data" / bid).glob("*.pdf"))
    print(f"\n##### {bid}: {len(pdfs)} PDFs")
    for p in pdfs:
        try:
            with pdfplumber.open(p) as pdf:
                chars = [len(pg.extract_text() or "") for pg in pdf.pages]
                tables = sum(len(pg.find_tables()) for pg in pdf.pages)
                first = (pdf.pages[0].extract_text() or "")[:200].replace("\n", " ")
                print(f"\n{p.name}")
                print(f"  pages={len(pdf.pages)}  pages_with_text={sum(c > 30 for c in chars)}  "
                      f"total_chars={sum(chars)}  tables_found={tables}")
                print(f"  first 200 chars: {first!r}")
        except Exception as e:
            print(f"\n{p.name}\n  FAILED: {type(e).__name__}: {e}")
