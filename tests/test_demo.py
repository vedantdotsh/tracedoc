"""Loopback demo requests use a fake retriever and one synthetic PDF."""

import http.client
import json
import threading
from urllib.parse import urlencode

import pymupdf
import dotenv

from evidence_lab import demo


def request(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


def post(port, body, *, origin=None, host=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    payload = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if origin is not None:
        headers["Origin"] = origin
    if host is not None:
        headers["Host"] = host
    try:
        conn.request("POST", "/api/ask", payload, headers)
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


def test_demo_search_page_and_path_guards(tmp_path, monkeypatch):
    pdf_dir = tmp_path / "data/documents"
    pdf_dir.mkdir(parents=True)
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), "Visible source page")
        doc.save(pdf_dir / "a.pdf")
    searches = []

    class FakeRetriever:
        rows = [{"doc_id": "a.pdf"}]

        def __init__(self, db, cache):
            pass

        def search(self, query, doc_id, top_k, method):
            searches.append((query, doc_id, top_k, method))
            if method not in ("bm25", "dense", "hybrid", "reranked"):
                raise ValueError("Unknown retrieval method")
            return [{"doc_id": "a.pdf", "page": 1, "chunk_id": "a#1",
                     "text": "Visible source page", "score": 0.5}]

    monkeypatch.setattr(demo, "Retriever", FakeRetriever)
    server = demo.create_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_port
        status, headers, body = request(port, "/")
        assert status == 200 and b"Document Evidence Lab" in body
        assert b"innerHTML" not in body  # result text is inserted through textContent
        assert headers["X-Content-Type-Options"] == "nosniff"
        status, _, body = request(port, "/api/docs")
        assert status == 200 and json.loads(body) == ["a.pdf"]
        status, _, body = request(port, "/api/answers")
        assert status == 200 and "No saved answer evaluation" in json.loads(body)["status"]
        path = "/api/search?" + urlencode({"q": "source", "doc": "a.pdf", "method": "hybrid"})
        status, _, body = request(port, path)
        assert status == 200 and json.loads(body)[0]["chunk_id"] == "a#1"
        assert searches == [("source", "a.pdf", 5, "hybrid")]
        for path in ("/api/search?q=", "/api/search?q=" + "x" * 2001,
                     "/api/search?q=x&doc=..%2Fsecret.pdf",
                     "/api/search?q=x&method=unknown", "/api/page?doc=a.pdf&page=0",
                     "/api/page?doc=a.pdf&page=2", "/api/page?doc=..%2Fsecret.pdf&page=1"):
            status, _, body = request(port, path)
            assert status == 400, (path, status, body)
        status, headers, body = request(port, "/api/page?doc=a.pdf&page=1")
        assert status == 200 and headers["Content-Type"] == "image/png"
        assert body.startswith(b"\x89PNG\r\n\x1a\n")
        assert request(port, "/does-not-exist")[0] == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_demo_paid_post_requires_same_origin_opt_in_and_session_budget(tmp_path, monkeypatch):
    calls = []

    class FakeRetriever:
        rows = [{"doc_id": "a.pdf"}]

        def __init__(self, db, cache):
            pass

        def search(self, query, doc_id, top_k, method):
            calls.append(("search", query, doc_id, top_k, method))
            return [{"doc_id": "a.pdf", "page": 1, "chunk_id": "a#1",
                     "text": "Visible evidence", "score": 0.5}]

    def fake_price():
        calls.append(("price",))
        return {"model": "fixture"}

    def fake_answer(question, hits, **kwargs):
        calls.append(("answer", question, kwargs))
        assert kwargs["allow_paid"] is True
        assert kwargs["provider_max_price"] == demo.evaluate_answers.PRICE_CEILINGS
        return {"execution_status": "ok", "status": "answered", "answer": "Visible",
                "abstained": False, "citations": [{"doc_id": "a.pdf", "page": 1}],
                "messages": [{"content": "private prompt"}], "raw_response": "private raw"}

    monkeypatch.setattr(demo, "Retriever", FakeRetriever)
    monkeypatch.setattr(demo.evaluate_answers, "current_prices", fake_price)
    monkeypatch.setattr(demo.evaluate_answers, "request_bound", lambda *a: {
        "cost_upper_bound_usd": 0.009, "output_token_limit": 384})
    monkeypatch.setattr(demo.answering, "answer_from_hits", fake_answer)
    server = demo.create_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_port
        origin = f"http://127.0.0.1:{port}"
        body = {"q": "Question", "doc": "a.pdf", "method": "hybrid", "allow_paid": True}
        assert post(port, body)[0] == 403
        assert post(port, body, origin="http://evil.example")[0] == 403
        assert post(port, body, origin=origin, host=f"evil.example:{port}")[0] == 403
        assert post(port, {**body, "allow_paid": False}, origin=origin)[0] == 400
        assert post(port, {**body, "doc": "unknown.pdf"}, origin=origin)[0] == 400
        assert calls == []
        for _ in range(5):
            status, result = post(port, body, origin=origin)
            assert status == 200
            assert result["answer"] == "Visible"
            assert "messages" not in result and "raw_response" not in result
        assert post(port, body, origin=origin)[0] == 400
        assert len([x for x in calls if x[0] == "answer"]) == 5
        assert len(list((tmp_path / "data/demo-answers").glob("*.json"))) == 5
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_demo_gate_declines_before_pricing_or_budget_reservation(tmp_path, monkeypatch):
    class FakeRetriever:
        rows = [{"doc_id": "a.pdf"}]

        def __init__(self, db, cache):
            pass

        def search(self, query, doc_id, top_k, method):
            return [{"doc_id": "a.pdf", "page": 1, "chunk_id": "a#1",
                     "text": "Section 8 requires an audit.", "score": 0.5}]

    monkeypatch.setattr(demo, "Retriever", FakeRetriever)
    monkeypatch.setattr(demo.evaluate_answers, "current_prices",
                        lambda: (_ for _ in ()).throw(RuntimeError(
                            "pricing requested for deterministic rejection")))
    monkeypatch.setattr(demo.answering, "_post",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError(
                            "provider called for deterministic rejection")))
    monkeypatch.setattr(dotenv, "load_dotenv",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError(
                            "environment loaded for deterministic rejection")))
    server = demo.create_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        origin = f"http://127.0.0.1:{server.server_port}"
        body = {"q": "What does section 7 require?", "doc": "a.pdf",
                "method": "bm25", "allow_paid": True, "evidence_gate": True}
        status, blocked = post(server.server_port, body, origin=origin)
        assert status == 200
        assert blocked["status"] == "gate_abstained"
        assert blocked["execution_status"] == "not_run"
        assert blocked["abstention_origin"] == "evidence_gate"
        assert blocked["session_reserved_usd"] == 0
        assert blocked["usage"] == {}
        assert blocked["evidence_gate"]["reasons"] == ["missing_typed_identifier"]

        calls = []
        monkeypatch.setattr(demo.evaluate_answers, "current_prices",
                            lambda: calls.append("price") or {"model": "fixture"})
        monkeypatch.setattr(demo.evaluate_answers, "request_bound",
                            lambda *a: {"cost_upper_bound_usd": 0.009,
                                        "output_token_limit": 384})

        def fake_post(*args, **kwargs):
            calls.append("provider")
            return json.dumps({"model": "fixture", "usage": {}, "choices": [{
                "finish_reason": "stop", "message": {"content": json.dumps({
                    "answer": "Section 8 requires an audit.", "citations": ["S1"],
                    "abstained": False})}}]}).encode()

        monkeypatch.setattr(demo.answering, "_post", fake_post)
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: None)
        status, generated = post(server.server_port, {**body, "evidence_gate": False},
                                 origin=origin)
        assert status == 200
        assert generated["status"] == "answered"
        assert generated["session_reserved_usd"] == 0.009
        assert calls == ["price", "provider"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
