import hashlib
import json
import sqlite3

import pymupdf
import pytest

from evidence_lab import sources


def by_doc(db, doc_id):
    with sqlite3.connect(db) as con:
        rows = con.execute('SELECT c.page, c.text, e.kind, e.source_sha256, e.system, e.revision, '
                           'e.location, e.extra FROM chunks c JOIN evidence e ON e.chunk_id = c.chunk_id '
                           'WHERE c.doc_id = ? ORDER BY c.rowid', (doc_id,)).fetchall()
    keys = ('page', 'text', 'kind', 'sha', 'system', 'revision')
    return [{**dict(zip(keys, r[:6])), 'location': json.loads(r[6]), 'extra': json.loads(r[7])}
            for r in rows]


def test_each_format_keeps_hash_location_metadata_and_units(evidence_db, inputs):
    row = next(r for r in by_doc(evidence_db, 'pumps.xlsx') if 'P-101' in r['text'])
    assert row['sha'] == hashlib.sha256((inputs / 'pumps.xlsx').read_bytes()).hexdigest()
    assert (row['page'], row['location']) == (1, {'sheet': 'Pumps', 'row': 2, 'header_row': 1})
    assert (row['system'], row['revision']) == ('Pump skid', 'A')
    assert {'cell': 'B2', 'label': 'Pressure [bar]', 'value': '3.2', 'unit': 'bar',
            'formula': False} in row['extra']['fields']
    req = {r['location']['requirement_id']: r for r in by_doc(evidence_db, 'reqs.csv')}
    assert (req['REQ-1']['revision'], req['REQ-2']['revision']) == ('B', 'C')  # row overrides file
    assert req['REQ-1']['extra']['unit'] == 'bar' and req['REQ-2']['extra']['unit'] is None
    assert req['REQ-2']['extra']['quantities'] == [{'value': '40', 'unit': '°C'}]
    (valve,) = by_doc(evidence_db, 'valves.md')
    assert valve['location'] == {'requirement_id': 'SYS-12', 'line_start': 2, 'line_end': 3}
    assert 'after trip signal' in valve['text']
    assert valve['extra']['quantities'] == [{'value': '500', 'unit': 'ms'}]
    (block,) = by_doc(evidence_db, 'drawing.pdf')
    assert block['location']['page'] == 1 and len(block['location']['bbox_pt']) == 4
    assert block['extra']['quantities'] == [{'value': '120', 'unit': 'mm'}]
    (line,) = by_doc(evidence_db, 'scan.png')
    assert line['kind'] == 'ocr_line' and line['extra']['source_verified'] is False
    assert line['location']['image'] == 'scan.png' and line['location']['box_px']


def test_empty_sheets_and_textless_pages_are_reported(tmp_path, inputs, metadata, fake_ocr):
    blank = tmp_path / 'blank.pdf'
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), 'Title block')
        doc.new_page()
        doc.save(blank)
    files = [inputs / 'pumps.xlsx', blank]
    plain = sources.build_index(files, tmp_path / 'a.sqlite', metadata)
    assert {'doc_id': 'blank.pdf', 'pages_without_evidence': [2]} in plain['diagnostics']
    assert {'doc_id': 'pumps.xlsx', 'pages_without_evidence': [2]} in plain['diagnostics']
    sources.build_index(files, tmp_path / 'b.sqlite', metadata, fake_ocr)
    assert any(r['kind'] == 'ocr_line' and r['page'] == 2 for r in by_doc(tmp_path / 'b.sqlite', 'blank.pdf'))


def test_missing_metadata_fails_closed_and_keeps_existing_index(tmp_path, inputs):
    db = tmp_path / 'e.sqlite'
    db.write_bytes(b'previous index')
    with pytest.raises(ValueError, match='system and revision'):
        sources.build_index([inputs / 'valves.md'], db, {'*': {'system': 'Pump skid'}})
    assert db.read_bytes() == b'previous index'
    assert list(tmp_path.glob('*.tmp')) == []


@pytest.mark.parametrize('name, content, message', [
    ('spec.reqif', '<x/>', 'unsupported'),
    ('model.step', 'x', 'unsupported'),
    ('dup.md', '- A-1: one\n- A-1: two\n', 'duplicate requirement IDs'),
    ('prose.txt', 'no ids here\n', 'no requirement IDs'),
    ('bad.csv', 'name,text\nx,y\n', 'id and text columns'),
    ('shot.png', 'x', 'OCR engine'),
])
def test_unsupported_or_ambiguous_inputs_are_rejected(tmp_path, name, content, message):
    path = tmp_path / name
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        sources.build_index([path], tmp_path / 'out' / 'e.sqlite', {'*': {'system': 's', 'revision': 'r'}})


def test_oversized_workbook_is_rejected(tmp_path, inputs, metadata, monkeypatch):
    monkeypatch.setattr(sources, 'MAX_XLSX_BYTES', 10)
    with pytest.raises(ValueError, match='expands beyond'):
        sources.build_index([inputs / 'pumps.xlsx'], tmp_path / 'e.sqlite', metadata)
