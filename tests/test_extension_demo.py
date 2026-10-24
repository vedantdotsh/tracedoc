"""The local exact-lookup viewer treats extracted data as untrusted text."""

import http.client
import json
import threading

import pymupdf

from evidence_lab import extension_demo


def table_fact(*, page=1, value="42"):
    return {"row_id": "AB-12", "column": "Capacity", "value": value,
            "unit": "MW", "conditions": {}, "source_verified": False,
            "provenance": {"doc_id": "a.pdf", "page": page,
                           "cell_bbox": [1, 2, 3, 4]}}


def test_render_escapes_extracted_values_and_refuses_ambiguity():
    fact = table_fact(value="<script>alert(1)</script>")
    html = extension_demo.render_page(
        [{"facts": [fact]}], {"id": ["AB-12"], "field": ["Capacity"]})
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>" not in html
    assert "Result: answer" in html

    duplicate = extension_demo.render_page(
        [{"facts": [fact, table_fact(value="99")]}],
        {"id": ["AB-12"], "field": ["Capacity"]})
    assert "Result: abstain" in duplicate
    assert "Original source" not in duplicate


def test_render_does_not_interpolate_untrusted_page_html():
    injected = "1<script>alert(2)</script>"
    html = extension_demo.render_page(
        [{"facts": [table_fact(page=injected)]}],
        {"id": ["AB-12"], "field": ["Capacity"]})
    assert "<script>" not in html


def request(port, path):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_source_viewer_allows_only_manifest_document_directory(tmp_path):
    docs = tmp_path / "data/documents"
    docs.mkdir(parents=True)
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), "Allowed source")
        document.save(docs / "a.pdf")
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), "Escaped source")
        document.save(tmp_path / "secret.pdf")
    manifest = tmp_path / "data/manifest.json"
    manifest.write_text(json.dumps({"documents": [
        {"doc_id": "a.pdf", "path": "documents/a.pdf"},
        {"doc_id": "escape.pdf", "path": "../secret.pdf"},
    ]}), encoding="utf-8")

    handler = extension_demo.handler_for(tmp_path)
    server = extension_demo.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert request(server.server_port, "/source?doc=a.pdf&page=1")[0] == 200
        assert request(server.server_port, "/source?doc=escape.pdf&page=1")[0] != 200
        assert request(server.server_port, "/source?doc=..%2Fsecret.pdf&page=1")[0] != 200
        assert request(server.server_port, "/source?doc=a.pdf&page=0")[0] != 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
