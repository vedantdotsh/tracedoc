"""Located-evidence ingestion for PDF, XLSX, requirement and drawing sources.

Output is a separate SQLite file with the baseline's chunks/docs/meta tables, so
baseline.search and retrieval.Retriever work on it unchanged, plus an `evidence`
table holding source hash, exact location, system/revision and units. Drawings
contribute only their text and its coordinates; no geometry or dimension is inferred.
"""
import csv
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
import re
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from .corpus import require_distinct_paths
from .ocr import sha256 as sha256_file

SUFFIXES = {'.pdf', '.xlsx', '.csv', '.txt', '.md', '.png', '.jpg', '.jpeg'}
MAX_XLSX_BYTES = 200 * 2 ** 20
# Units are only recorded when written next to a number; extend as real data demands.
UNITS = ('mm', 'cm', 'km', 'ms', 'mbar', 'bar', 'psi', 'Pa', 'kPa', 'MPa', 'kN', 'Nm', 'kg',
         'V', 'mV', 'mA', 'kW', 'Hz', 'kHz', 'MHz', '°C', '°F', 'rpm')
QUANTITY = re.compile(r'(?<![\w.])(-?\d+(?:\.\d+)?)\s?('
                      + '|'.join(sorted(map(re.escape, UNITS), key=len, reverse=True)) + r')(?!\w)')
MAIN = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
REL_ID = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
HEADER_UNIT = re.compile(r'[\[(]([^\[\]()]+)[\])]\s*$')
REQ_LINE = re.compile(r'^\s*(?:[-*]\s+)?\[?((?=[^\s:\]]*\d)[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)+)\]?\s*:\s*(.+?)\s*$')


def quantities(text):
    return [{'value': v, 'unit': u} for v, u in QUANTITY.findall(text)]


def _record(text, page, kind, location, system=None, revision=None, **extra):
    return {'text': text, 'page': page, 'kind': kind, 'location': location,
            'system': system, 'revision': revision,
            'extra': {'quantities': quantities(text), **extra}}


def _ocr_records(lines, page, base):
    return [_record(' '.join(line['text'].split()), page, 'ocr_line',
                    {**base, 'line': n, **({'box_px': line['box']} if line.get('box') is not None else {})},
                    confidence=line.get('confidence'), source_verified=False,
                    **{k: line[k] for k in ('engine', 'model', 'extraction_sha256') if k in line})
            for n, line in enumerate(lines) if line['text'].strip()]


def rapidocr_lines(data):
    """OCR encoded image bytes with RapidOCR, as ocr.recover_page does for PDF pages."""
    global _ENGINE
    import cv2
    import numpy as np
    from rapidocr import RapidOCR
    _ENGINE = _ENGINE or RapidOCR(params={'Global.log_level': 'error'})
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('undecodable image')
    result = _ENGINE(image)
    if result.txts is None:
        return []
    return [{'text': t, 'confidence': float(s), 'box': b.tolist()}
            for t, s, b in zip(result.txts, result.scores, result.boxes)]


_ENGINE = None


def _pdf(path, ocr):
    import pymupdf
    records = []
    with pymupdf.open(path) as doc:
        for number, page in enumerate(doc, 1):
            blocks = [b for b in page.get_text('blocks', sort=True) if b[6] == 0 and b[4].strip()]
            for n, b in enumerate(blocks):
                records.append(_record(' '.join(b[4].split()), number, 'pdf_text_block',
                                       {'page': number, 'block': n,
                                        'bbox_pt': [round(x, 2) for x in b[:4]]}))
            if not blocks and ocr:  # image-only page: same candidate-evidence status as ocr.py
                pix = page.get_pixmap(dpi=150, alpha=False)
                records += _ocr_records(ocr(pix.tobytes('png')), number,
                                        {'page': number, 'render_dpi': 150,
                                         'size_px': [pix.width, pix.height]})
        return records, len(doc)


def _image(path, ocr):
    if ocr is None:
        raise ValueError(f'{path.name}: an OCR engine is required for images (use --ocr)')
    import pymupdf
    data = path.read_bytes()
    pix = pymupdf.Pixmap(data)  # rejects undecodable images
    return _ocr_records(ocr(data), 1, {'image': path.name, 'size_px': [pix.width, pix.height]}), 1


def _column(ref):
    return re.match(r'[A-Z]+', ref)[0]


def _sheet(root, name, number, state, shared):
    rows = []
    for row in root.iter(MAIN + 'row'):
        cells = {}
        for c in row.iter(MAIN + 'c'):
            v, kind = c.find(MAIN + 'v'), c.get('t')
            if kind == 'inlineStr':
                value = ''.join(t.text or '' for t in c.iter(MAIN + 't'))
            elif v is None or v.text is None:
                formula = c.find(MAIN + 'f')
                if formula is None:
                    continue
                value = '[formula not calculated: ' + (formula.text or '') + ']'
            elif kind == 's':
                value = shared[int(v.text)]
            else:  # numbers and formula results exactly as cached; dates stay serial numbers
                value = v.text
            if value.strip():
                cells[c.get('r')] = (value.strip(), c.find(MAIN + 'f') is not None)
        if cells:
            rows.append((int(row.get('r')), cells))
    if not rows:
        return []
    header_row, header = rows[0]  # assumption: first non-empty row holds the column labels
    labels = {_column(ref): value for ref, (value, _) in header.items()}
    records = []
    for r, cells in rows[1:]:
        fields = []
        for ref, (value, formula) in cells.items():
            label = labels.get(_column(ref), _column(ref))
            unit = HEADER_UNIT.search(label)
            fields.append({'cell': ref, 'label': label, 'value': value,
                           'unit': unit[1].strip() if unit else None, 'formula': formula})
        text = f'{name} row {r}: ' + '; '.join(f"{f['label']}: {f['value']}" for f in fields)
        records.append(_record(text, number, 'xlsx_row',
                               {'sheet': name, 'row': r, 'header_row': header_row},
                               sheet_state=state, fields=fields))
    return records


def _xlsx(path, ocr):
    with zipfile.ZipFile(path) as z:
        if sum(i.file_size for i in z.infolist()) > MAX_XLSX_BYTES:
            raise ValueError(f'{path.name}: workbook expands beyond {MAX_XLSX_BYTES} bytes')
        xml = lambda name: ET.fromstring(z.read(name))
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            shared = [''.join(t.text or '' for t in si.iter(MAIN + 't'))
                      for si in xml('xl/sharedStrings.xml').iter(MAIN + 'si')]
        rels = {r.get('Id'): r.get('Target') for r in xml('xl/_rels/workbook.xml.rels')}
        records, sheets = [], list(xml('xl/workbook.xml').iter(MAIN + 'sheet'))
        for number, sheet in enumerate(sheets, 1):
            target = rels[sheet.get(REL_ID)].lstrip('/')
            target = target if target.startswith('xl/') else 'xl/' + target
            records += _sheet(xml(target), sheet.get('name'), number,
                              sheet.get('state', 'visible'), shared)
    return records, len(sheets)


def _checked(path, records):
    ids = [r['location']['requirement_id'] for r in records]
    if not ids:
        raise ValueError(f'{path.name}: no requirement IDs found')
    duplicates = sorted(i for i, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f'{path.name}: duplicate requirement IDs {duplicates}')
    return records, 1


def _csv(path, ocr):
    records = []
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        cols = {name.strip().lower(): name for name in reader.fieldnames or []}
        if not {'id', 'text'} <= cols.keys():
            raise ValueError(f'{path.name}: CSV needs id and text columns')
        for n, row in enumerate(reader, 1):
            get = lambda key: ((row[cols[key]] if key in cols else '') or '').strip()
            if not get('id') or not get('text'):
                raise ValueError(f'{path.name}: CSV row {n} has a blank id or text')
            unit = get('unit') or None
            text = f"{get('id')}: {get('text')}" + (f' [unit: {unit}]' if unit else '')
            text += ''.join(f'; {name}: {value}' for name, value in row.items()
                            if name and name.strip().lower() not in ('id', 'text', 'unit') and value)
            records.append(_record(text, 1, 'requirement',
                                   {'requirement_id': get('id'), 'csv_row': n,
                                    'csv_line_end': reader.line_num},
                                   get('system') or None, get('revision') or None, unit=unit))
    return _checked(path, records)


def _requirements_text(path, ocr):
    found, current = [], None  # lines that are neither "ID: text" nor indented continuations are ignored
    for n, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        m = REQ_LINE.match(line)
        if m:
            current = [m[1], [m[2]], n, n]
            found.append(current)
        elif current and line[:1].isspace() and line.strip():
            current[1].append(line.strip())
            current[3] = n
        else:
            current = None
    return _checked(path, [
        _record(f'{i}: {" ".join(parts)}', 1, 'requirement',
                {'requirement_id': i, 'line_start': a, 'line_end': b})
        for i, parts, a, b in found])


EXTRACTORS = {'.pdf': _pdf, '.xlsx': _xlsx, '.csv': _csv, '.txt': _requirements_text,
              '.md': _requirements_text, '.png': _image, '.jpg': _image, '.jpeg': _image}


def expand(paths):
    """Directories contribute their supported files (not recursive); explicit files pass through."""
    out = []
    for p in map(Path, paths):
        out += ([f for f in sorted(p.iterdir()) if f.is_file() and f.suffix.lower() in SUFFIXES]
                if p.is_dir() else [p])
    return out


def build_index(inputs, db_path, metadata, ocr=None) -> dict:
    """metadata: {'*': {'system','revision'}, '<file name>': {...}}; requirement rows may override."""
    paths = sorted({Path(p) for p in inputs}, key=lambda p: p.name)
    if not paths:
        raise ValueError('no input files')
    names = [p.name for p in paths]
    if len(set(names)) != len(names):
        raise ValueError('input file names must be unique; they are the doc_ids')
    unsupported = [n for n in names if Path(n).suffix.lower() not in SUFFIXES]
    if unsupported:
        raise ValueError(f'unsupported file types (no ReqIF or STEP support): {unsupported}')
    db_path = Path(db_path)
    for path in paths:
        require_distinct_paths(db_path, path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=db_path.parent, prefix=db_path.name + '.',
                                     suffix='.tmp', delete=False) as stream:
        tmp = Path(stream.name)
    con = sqlite3.connect(tmp)
    stats, diagnostics, source_lines = {'docs': 0, 'records': 0, 'kinds': {}}, [], []
    try:
        con.execute('CREATE VIRTUAL TABLE chunks USING fts5('
                    'doc_id UNINDEXED, page UNINDEXED, chunk_id UNINDEXED, text)')
        con.execute('CREATE TABLE docs(doc_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, '
                    'pages INTEGER NOT NULL)')
        con.execute('CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        con.execute('CREATE TABLE evidence(chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, '
                    'kind TEXT NOT NULL, source_sha256 TEXT NOT NULL, system TEXT NOT NULL, '
                    'revision TEXT NOT NULL, location TEXT NOT NULL, extra TEXT NOT NULL)')
        for path in paths:
            sha = sha256_file(path)
            records, pages = EXTRACTORS[path.suffix.lower()](path, ocr)
            from .openrouter_embeddings import chunk_text
            records = [{**r, 'text': part} for r in records for part in chunk_text(r['text'])]
            if sha256_file(path) != sha:
                raise RuntimeError(f'{path.name} changed during ingestion')
            if not records:
                raise ValueError(f'{path.name}: no evidence extracted (scanned? try --ocr)')
            for n, r in enumerate(records):
                who = {**metadata.get('*', {}), **metadata.get(path.name, {}),
                       **{k: r[k] for k in ('system', 'revision') if r[k]}}
                if not all(isinstance(who.get(k), str) and who[k].strip()
                           for k in ('system', 'revision')):
                    raise ValueError(f'{path.name}: system and revision are required')
                chunk_id = f'{path.name}#e{n}'
                con.execute('INSERT INTO chunks VALUES (?, ?, ?, ?)',
                            (path.name, r['page'], chunk_id, r['text']))
                con.execute('INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                            (chunk_id, path.name, r['kind'], sha, who['system'], who['revision'],
                             json.dumps(r['location'], sort_keys=True),
                             json.dumps(r['extra'], sort_keys=True, ensure_ascii=False)))
                stats['kinds'][r['kind']] = stats['kinds'].get(r['kind'], 0) + 1
            con.execute('INSERT INTO docs VALUES (?, ?, ?)', (path.name, sha, pages))
            missing = sorted(set(range(1, pages + 1)) - {r['page'] for r in records})
            if missing:
                diagnostics.append({'doc_id': path.name, 'pages_without_evidence': missing})
            source_lines.append(f'{path.name} {sha}')
            stats['docs'] += 1
            stats['records'] += len(records)
        meta = {'page_base': 1,
                'page_meaning': 'pdf page | xlsx sheet index | 1 for requirement and image files',
                'ocr': getattr(ocr, '__name__', type(ocr).__name__) if ocr else 'none',
                'pymupdf': importlib.metadata.version('PyMuPDF'), 'tokenizer': 'fts5 unicode61',
                'source_paths': json.dumps({p.name: str(p.resolve()) for p in paths}),
                'chunk_tokenizer': 'cl100k_base', 'max_chunk_tokens': 800,
                'sources_sha256': hashlib.sha256(
                    '\n'.join(source_lines).encode()).hexdigest()}
        con.executemany('INSERT INTO meta VALUES (?, ?)', [(k, str(v)) for k, v in meta.items()])
        con.commit()
        con.close()
        os.replace(tmp, db_path)
    except BaseException:
        con.close()
        tmp.unlink(missing_ok=True)
        raise
    return {**stats, 'diagnostics': diagnostics, 'db': str(db_path), **meta}
