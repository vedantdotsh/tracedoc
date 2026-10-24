"""SQLite FTS5 (BM25) page-cited retrieval baseline over PDF text only."""
import ast
import hashlib
import json
import math
import os
import re
import sqlite3
import time
import tempfile
from contextlib import closing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from .corpus import require_distinct_paths

PAGE_BASE = 1  # physical PDF citations; audit quarantines incompatible source annotations
ALL_CHUNKS = 2 ** 31 - 1
TERM = re.compile(r'[^\W_]+')


def _check_int(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def chunk_page(text: str, words: int = 180, overlap: int = 30) -> list[str]:
    _check_int('words', words, 1)
    _check_int('overlap', overlap, 0)
    if overlap >= words:
        raise ValueError('overlap must be smaller than words')
    tokens = text.split()
    chunks = []
    for start in range(0, len(tokens), words - overlap):
        chunks.append(' '.join(tokens[start:start + words]))
        if start + words >= len(tokens):
            break
    return chunks


def _extract_pdf(pdf, strategy, words, overlap):
    import pymupdf
    from .chunking import page_chunks
    with pdf.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    diagnostics = []
    with pymupdf.open(pdf) as doc:
        parts = [page_chunks(page, strategy, words, overlap, diagnostics) for page in doc]
    return digest, parts, diagnostics


def build_index(pdf_dir: Path, db_path: Path, words: int = 180, overlap: int = 30,
                strategy: str = 'fixed', workers: int = 1) -> dict:
    import pymupdf
    from .chunking import page_chunks, STRATEGIES, MAX_DRAWING_ITEMS

    chunk_page('', words, overlap)  # validates settings
    _check_int('workers', workers, 1)
    if workers > 4:
        raise ValueError('workers must not exceed 4')
    if strategy not in STRATEGIES:
        raise ValueError(f'unknown chunk strategy: {strategy}')
    pdf_dir, db_path = Path(pdf_dir), Path(db_path)
    pdfs = sorted(p for p in pdf_dir.iterdir() if p.is_file() and p.suffix.lower() == '.pdf')
    if not pdfs:
        raise ValueError(f'no PDFs in {pdf_dir}')
    for pdf in pdfs:
        require_distinct_paths(db_path, pdf)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=db_path.parent, prefix=db_path.name + '.',
                                     suffix='.tmp', delete=False) as stream:
        tmp = Path(stream.name)
    con = sqlite3.connect(tmp)
    stats = {'docs': 0, 'pages': 0, 'empty_pages': 0, 'chunks': 0}
    failures, source_lines, diagnostics = [], [], []
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        con.execute('CREATE VIRTUAL TABLE chunks USING fts5('
                    'doc_id UNINDEXED, page UNINDEXED, chunk_id UNINDEXED, text)')
        con.execute('CREATE TABLE docs(doc_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, '
                    'pages INTEGER NOT NULL)')
        con.execute('CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        futures = [pool.submit(_extract_pdf, p, strategy, words, overlap) for p in pdfs] if pool else None
        for i, pdf in enumerate(pdfs):
            try:
                digest, parts_by_page, doc_diagnostics = (futures[i].result() if futures
                    else _extract_pdf(pdf, strategy, words, overlap))
                diagnostics.extend({'doc_id': pdf.name, **d} for d in doc_diagnostics)
            except Exception as exc:
                failures.append({'doc_id': pdf.name, 'error': f'{type(exc).__name__}: {exc}'})
                continue
            rows = []
            for page_no, parts in enumerate(parts_by_page, PAGE_BASE):
                stats['empty_pages'] += not parts
                rows += [(pdf.name, page_no, f'{pdf.name}#p{page_no}-c{n}', part)
                         for n, part in enumerate(parts)]
            con.executemany('INSERT INTO chunks VALUES (?, ?, ?, ?)', rows)
            con.execute('INSERT INTO docs VALUES (?, ?, ?)', (pdf.name, digest, len(parts_by_page)))
            source_lines.append(f'{pdf.name} {digest}')
            stats['docs'] += 1
            stats['pages'] += len(parts_by_page)
            stats['chunks'] += len(rows)
            if stats['docs'] % 10 == 0:
                print(f'{strategy}: processed {stats["docs"]}/{len(pdfs)} documents', flush=True)
        if failures:
            raise RuntimeError(f'{len(failures)} unreadable PDF(s); index not built: '
                               + json.dumps(failures))
        meta = {
            'words': words, 'overlap': overlap, 'page_base': PAGE_BASE, 'strategy': strategy,
            'pymupdf': pymupdf.__version__, 'tokenizer': 'fts5 unicode61',
            'sources_sha256': hashlib.sha256('\n'.join(source_lines).encode()).hexdigest(),
        }
        if strategy == 'table':
            meta['max_drawing_items'] = MAX_DRAWING_ITEMS
        con.executemany('INSERT INTO meta VALUES (?, ?)', [(k, str(v)) for k, v in meta.items()])
        con.commit()
        con.close()
        os.replace(tmp, db_path)
    except BaseException:
        con.close()
        tmp.unlink(missing_ok=True)
        raise
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    return {**stats, 'failures': failures, 'diagnostics': diagnostics,
            'extraction_workers': workers, 'db': str(db_path), **meta}


def _connect(db_path):
    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'index not found: {path}')
    return sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True)


def search(db_path: Path, query: str, doc_id: str | None = None, top_k: int = 5) -> list[dict]:
    _check_int('top_k', top_k, 1)
    terms = TERM.findall(query)
    if not terms:
        return []
    match = ' OR '.join(f'"{term}"' for term in terms)
    sql = ('SELECT doc_id, page, chunk_id, text, bm25(chunks) AS score '
           'FROM chunks WHERE chunks MATCH ?')
    args = [match]
    if doc_id is not None:
        sql += ' AND doc_id = ?'
        args.append(doc_id)
    sql += ' ORDER BY score, doc_id, page, chunk_id LIMIT ?'  # bm25: lower is better
    args.append(top_k)
    with closing(_connect(db_path)) as con:
        rows = con.execute(sql, args).fetchall()
    return [{'doc_id': d, 'page': int(p), 'chunk_id': c, 'text': t, 'score': s}
            for d, p, c, t, s in rows]


def score_pages(expected, retrieved) -> dict:
    """recall = fraction of gold pages retrieved; hit = any gold page; mrr = first gold rank."""
    gold = set(expected)
    if not gold:
        raise ValueError('expected pages must not be empty')
    recall = len(gold & set(retrieved)) / len(gold)
    rank = next((i for i, page in enumerate(retrieved, 1) if page in gold), None)
    return {'recall': recall, 'hit': float(recall > 0), 'mrr': 1 / rank if rank else 0.0}


def _literal_list(value, label, kind):
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f'{label} is not a literal list: {value!r}') from exc
    if not isinstance(value, list) or any(
            isinstance(v, bool) or not isinstance(v, kind) for v in value):
        raise ValueError(f'{label} must be a list of {kind.__name__}')
    return value


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _mean(values):
    return sum(values) / len(values) if values else None


def _run_case(row, sample, db_path, page_counts, top_k):
    if not isinstance(sample, dict):
        raise ValueError('sample must be an object')
    missing = {'doc_id', 'question', 'evidence_pages', 'evidence_sources'} - sample.keys()
    if missing:
        raise ValueError(f'missing keys {sorted(missing)}')
    doc_id, question = sample['doc_id'], sample['question']
    if not isinstance(doc_id, str) or doc_id not in page_counts:
        raise ValueError(f'unknown doc_id {doc_id!r}')
    if not isinstance(question, str):
        raise ValueError('question must be a string')
    expected = sorted(set(_literal_list(sample['evidence_pages'], 'evidence_pages', int)))
    _literal_list(sample['evidence_sources'], 'evidence_sources', str)
    bad = [p for p in expected if not PAGE_BASE <= p <= page_counts[doc_id]]
    if bad:
        raise ValueError(f'evidence pages {bad} outside {PAGE_BASE}..{page_counts[doc_id]} '
                         f'for {doc_id}')
    if expected and sample.get('answer') == 'Not answerable':
        raise ValueError('Not answerable sample has evidence pages')
    if not expected and sample.get('answer') != 'Not answerable':
        raise ValueError('answer without page evidence; run the annotation audit first')
    start = time.perf_counter()
    hits = search(db_path, question, doc_id=doc_id, top_k=ALL_CHUNKS)  # question only
    retrieved = list(dict.fromkeys(hit['page'] for hit in hits))[:top_k]
    latency_ms = (time.perf_counter() - start) * 1000
    case = {'row': row, 'source_row': sample.get('source_row', row),
            'doc_id': doc_id, 'question': question, 'answerable': bool(expected),
            'expected_pages': expected, 'retrieved_pages': retrieved,
            'recall': None, 'hit': None, 'mrr': None, 'latency_ms': latency_ms}
    if expected:
        case.update(score_pages(expected, retrieved))
    return case


def evaluate(db_path: Path, samples: list[dict], top_k: int = 5) -> dict:
    """Document-scoped page@k retrieval metrics; answerable and unanswerable are separate."""
    _check_int('top_k', top_k, 1)
    if not isinstance(samples, list):
        raise ValueError('samples must be a list')
    with closing(_connect(db_path)) as con:
        page_counts = dict(con.execute('SELECT doc_id, pages FROM docs'))
        meta = dict(con.execute('SELECT key, value FROM meta'))
    if meta.get('page_base') != str(PAGE_BASE):
        raise ValueError('index page base does not match evaluator')
    cases = []
    for row, sample in enumerate(samples):
        try:
            cases.append(_run_case(row, sample, db_path, page_counts, top_k))
        except ValueError as exc:
            raise ValueError(f'sample {row}: {exc}') from exc
    scored = [c for c in cases if c['answerable']]
    latencies = [c['latency_ms'] for c in cases]
    return {
        'config': {
            'top_k': top_k, 'page_base': PAGE_BASE,
            'retrieval_unit': 'page@k: unique pages ranked by best-scoring chunk (BM25, OR terms)',
            'scope': 'document-scoped', 'index': meta,
        },
        'docs': len({c['doc_id'] for c in cases}),
        'answerable': len(scored),
        'unanswerable': len(cases) - len(scored),
        'unanswerable_note': 'Excluded from retrieval metrics; not evidence of abstention quality.',
        'mean_recall': _mean([c['recall'] for c in scored]),
        'mean_hit': _mean([c['hit'] for c in scored]),
        'mrr': _mean([c['mrr'] for c in scored]),
        'latency_ms': {'p50': _percentile(latencies, 0.5), 'p95': _percentile(latencies, 0.95)},
        'cases': cases,
    }
