"""Read-only local demo for the structured-table extension.

Run: python -m evidence_lab.extension_demo
No model calls, credentials, document writes or paid operations are available here.
"""

import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from .tables import lookup, verify_reviewed_answer


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data/tables-v1/extraction/tables.json"


def render_page(tables, params, reviews=()):
    """Render escaped evidence and a live deterministic lookup, not a model answer."""
    facts = [fact for table in tables for fact in table.get("facts", [])]
    row_ids = sorted({fact["row_id"] for fact in facts})
    columns = sorted({fact["column"] for fact in facts})
    row_id = params.get("id", ["M270TF-XX"])[0]
    column = params.get("field", ["Product Weight"])[0]
    raw_conditions = params.get("conditions", ["{}"])[0]
    unit = params.get("unit", [""])[0].strip() or None

    def options(values, selected):
        return "".join(
            f'<option value="{escape(value, quote=True)}"'
            f'{" selected" if value == selected else ""}>{escape(value)}</option>'
            for value in values
        )

    try:
        conditions = json.loads(raw_conditions)
        if not isinstance(conditions, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in conditions.items()
        ):
            raise ValueError("Conditions must be a JSON object with text keys and values.")
        result = lookup(tables, row_id, column, conditions=conditions, unit=unit)
        result = verify_reviewed_answer(result, reviews)
    except (ValueError, TypeError) as error:
        result = {"status": "invalid request", "reason": str(error)}

    evidence = ""
    provenance = result.get("provenance") or {}
    if provenance and type(provenance.get("page")) is int and provenance["page"] > 0:
        doc_id, page = provenance["doc_id"], provenance["page"]
        source_url = f"/source?doc={quote(doc_id, safe='')}&page={page}"
        evidence = (
            f'<h2>Original source: physical PDF page {page}</h2>'
            f'<p>{escape(doc_id)}. Inspect the model header, row, units and conditions.</p>'
            f'<a href="{source_url}" target="_blank">Open source page at full size</a>'
            f'<img src="{source_url}" alt="Original PDF page for checking the selected value">'
        )

    sample_conditions = sorted({json.dumps(fact.get("conditions", {}), sort_keys=True) for fact in facts})
    return f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Document Evidence Lab — exact lookup</title>
<style>
body{{font:17px/1.6 system-ui,sans-serif;color:#17233b;background:#f5f7fc;margin:0}}
main{{max-width:1050px;margin:30px auto;padding:30px;background:white;border-radius:16px}}
h1{{line-height:1.2}}form{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
label{{display:block;font-weight:600}}input,select,button{{box-sizing:border-box;width:100%;padding:12px;font:inherit}}
button{{background:#354ac7;color:white;border:0;cursor:pointer}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef2ff;padding:20px}}
img{{display:block;max-width:100%;margin-top:20px}}a{{color:#354ac7}}
@media(max-width:650px){{main{{margin:8px;padding:18px}}form{{grid-template-columns:1fr}}}}
</style><main><h1>Exact values with traceable evidence</h1>
<p>Choose a component or row, a field, and any required conditions. This calls the structured lookup directly.
It is not natural-language query parsing and makes no paid requests.</p>
<form method="get">
<label>Component / row ID<select name="id">{options(row_ids, row_id)}</select></label>
<label>Field<select name="field">{options(columns, column)}</select></label>
<label>Conditions (JSON)<input name="conditions" value="{escape(raw_conditions, quote=True)}"></label>
<label>Required unit (optional)<input name="unit" value="{escape(unit or '', quote=True)}"></label>
<button type="submit">Look up the source value</button></form>
<p>Example conditions: {escape('; '.join(sample_conditions))}</p>
<h2>Result: {escape(result.get('status', 'unknown'))}</h2>
<p style="font-size:30px;font-weight:650">{escape(str(result.get('value') if result.get('value') is not None else result.get('reason', '')))} {escape(str(result.get('unit') or ''))}</p>
<p>Verification: {escape(result.get('verification', {}).get('status', 'not reviewed'))}</p>
<details><summary>Inspect the returned record and cell provenance</summary>
<pre>{escape(json.dumps(result, ensure_ascii=False, indent=2))}</pre></details>
<p>A unique cell reference does not prove correct extraction. <code>source_verified</code> only reflects
an explicit review recorded for that fact. Ambiguous matches are refused.</p>{evidence}
<h2>Continue in the notebooks</h2><p>09: extraction · 10: exact lookup · 11: table evaluation · 12: diagrams.
The original text-only experiments remain available separately.</p></main></html>'''


def handler_for(root):
    root = Path(root)
    source_dir = (root / "data/documents").resolve()
    documents = {}
    for entry in json.loads((root / "data/manifest.json").read_text("utf-8"))["documents"]:
        candidate = (root / "data" / entry["path"]).resolve()
        if candidate.is_relative_to(source_dir) and candidate.suffix.lower() == ".pdf" and candidate.is_file():
            documents[entry["doc_id"]] = candidate

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if len(parsed.query) > 4096:
                self.send_error(414)
                return
            params = parse_qs(parsed.query, keep_blank_values=True)
            try:
                if parsed.path == "/":
                    tables = json.loads((root / "data/tables-v1/extraction/tables.json").read_text("utf-8"))
                    review_path = root / "data/tables-v1/source-review.json"
                    reviews = json.loads(review_path.read_text("utf-8"))["reviews"] if review_path.exists() else []
                    body = render_page(tables, params, reviews).encode("utf-8")
                    content_type = "text/html; charset=utf-8"
                elif parsed.path == "/source":
                    import fitz
                    doc = params.get("doc", [""])[0]
                    page = int(params.get("page", ["0"])[0])
                    if doc not in documents:
                        self.send_error(404)
                        return
                    with fitz.open(documents[doc]) as pdf:
                        if not 1 <= page <= len(pdf):
                            self.send_error(404)
                            return
                        body = pdf[page - 1].get_pixmap(matrix=fitz.Matrix(1.25, 1.25)).tobytes("png")
                    content_type = "image/png"
                else:
                    self.send_error(404)
                    return
            except (ValueError, OSError, KeyError):
                self.send_error(400, "Cannot load the requested local artifact")
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    if not TABLES.exists():
        raise SystemExit("Run the table pilot first: data/tables-v1/extraction/tables.json is missing.")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(ROOT))
    print(f"Exact-value demo: http://127.0.0.1:{server.server_port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
