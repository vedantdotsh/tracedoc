import json
import threading
import http.client
import pytest
import sqlite3
from evidence_lab import sources, answering
from evidence_lab.api_ocr import OpenRouterOCR
from evidence_lab.workspace import Workspace
from evidence_lab.workspace_chat import create_server


def test_locators_and_stale_source(evidence_db, inputs):
    ws = Workspace(evidence_db)
    hit = ws.search('material SS316')[0]
    assert hit['page'] is None
    assert hit['location']['sheet'] == 'Pumps'
    assert hit['location']['cells'] == 'A2, B2, C2'
    assert ws.source(hit['chunk_id'])['source_sha256']
    with pytest.raises(ValueError, match='Unknown'):
        ws.source('../.env')
    (inputs / 'pumps.xlsx').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Source changed'):
        ws.source(hit['chunk_id'])


def test_chat_local_search_and_origin_guards(evidence_db):
    server = create_server(Workspace(evidence_db), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port
    def call(path, body=None, origin=None):
        con = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
        try:
            con.request('POST' if body is not None else 'GET', path,
                        json.dumps(body) if body is not None else None,
                        {'Origin': origin} if origin else {})
            res = con.getresponse()
            return res.status, res.read()
        finally:
            con.close()
    try:
        assert b'innerHTML' not in call('/')[1]
        config = json.loads(call('/api/config')[1])
        assert config['default_method'] == 'bm25' and not config['embeddings_enabled']
        assert not config['generation_enabled']
        body = {'q': 'material SS316', 'method': 'bm25'}
        assert call('/api/search', body)[0] == 403
        status, raw = call('/api/search', body, f'http://127.0.0.1:{port}')
        assert status == 200 and json.loads(raw)['hits'][0]['location']['sheet'] == 'Pumps'
        assert call('/api/ask', {**body, 'allow_paid': True}, f'http://127.0.0.1:{port}')[0] == 403
        assert call('/api/source?chunk_id=../.env')[0] == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_api_ocr_and_hidden_sheet_provenance_reach_prompt(tmp_path, inputs, metadata):
    db = tmp_path / 'evidence.sqlite'
    sources.build_index([inputs / 'scan.png', inputs / 'pumps.xlsx'], db, metadata,
                        ocr=lambda data: OpenRouterOCR._records(['VALVE V-101']))
    with sqlite3.connect(db) as con:
        rows = con.execute("SELECT chunk_id,extra FROM evidence WHERE kind='xlsx_row'").fetchall()
        for key, raw in rows:
            extra = json.loads(raw)
            extra['sheet_state'] = 'hidden'
            con.execute('UPDATE evidence SET extra=? WHERE chunk_id=?', (json.dumps(extra), key))
    ws = Workspace(db)
    hit = ws.search('V-101')[0]
    assert hit['extraction']['source_verified'] is False
    assert hit['extraction']['engine'] == 'openrouter_vision'
    assert 'box_px' not in hit['location']
    prompt = answering._messages('valve?', answering._build_sources([hit]))[1]['content']
    assert 'openrouter_vision' in prompt and 'source_verified' in prompt
    hit = ws.search('SS316')[0]
    assert hit['location']['sheet_state'] == 'hidden'
    assert 'hidden' in answering._messages('material?', answering._build_sources([hit]))[1]['content']
