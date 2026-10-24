"""Build once, then record illustrative corpus-wide searches without generation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evidence_lab.extension_retrieval import build, search

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build", action="store_true")
args = parser.parse_args()
if args.build:
    print(build(ROOT))

questions = [
    "M270TF-XX Product Weight kg",
    "display mode QUAD selected radio button",
    "transformer decoder prediction heads FFN",
]
rows = [{"question": question, "method": method, "hits": search(ROOT, question, method)}
        for question in questions for method in ("bm25", "hybrid")]
folder = ROOT / "reports/extension-index-v1"
folder.mkdir(exist_ok=False)
(folder / "examples.json").write_text(json.dumps({
    "scope": "Illustrative corpus-wide searches, not a frozen benchmark or answer accuracy measure",
    "generation_calls": 0, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved {len(rows)} searches to {folder}")
