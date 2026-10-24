"""Resumable local OCR for pages lacking baseline text; never reads benchmark labels."""
import hashlib
import importlib.metadata
import json
import sqlite3
import time
from pathlib import Path
from .corpus import write_json_atomic as _write_json_atomic


def write_json_atomic(path, value):
    """Allow short Windows/OneDrive sharing locks without losing a checkpoint."""
    for attempt in range(8):
        try:
            return _write_json_atomic(path, value)
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.1 * (attempt + 1))


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare_manifest(baseline, pdf_dir, output):
    import rapidocr
    import pymupdf
    models = Path(rapidocr.__file__).parent / 'models'
    names = {'Det': 'PP-OCRv6_det_small.onnx', 'Cls': 'ch_ppocr_mobile_v2.0_cls_mobile.onnx', 'Rec': 'PP-OCRv6_rec_small.onnx'}
    params = {'EngineConfig.onnxruntime.intra_op_num_threads': 2,
              'EngineConfig.onnxruntime.inter_op_num_threads': 1,
              'Global.max_side_len': 2400, 'Global.log_level': 'error'}
    for kind, name in names.items():
        params[kind + '.model_path'] = str((models / name).resolve())
    settings = {'engine': 'RapidOCR', 'rapidocr': importlib.metadata.version('rapidocr'),
                'onnxruntime': importlib.metadata.version('onnxruntime'), 'pymupdf': pymupdf.__version__,
                'dpi': 150, 'max_render_side': 2400, 'params': params,
                'models': {name: sha256(models / name) for name in names.values()},
                'defaults_sha256': sha256(Path(rapidocr.__file__).parent / 'config.yaml'),
                'text_order': 'RapidOCR returned line order', 'status_blank_meaning': 'No text above engine confidence threshold; not proof visually blank'}
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    pages = []
    with sqlite3.connect(f'{Path(baseline).resolve().as_uri()}?mode=ro', uri=True) as con:
        indexed = set((d, int(p)) for d, p in con.execute('select distinct doc_id,page from chunks'))
        docs = con.execute('select doc_id,sha256,pages from docs order by doc_id').fetchall()
    for doc_id, source_hash, count in docs:
        if sha256(Path(pdf_dir) / doc_id) != source_hash:
            raise ValueError(f'Source differs from baseline: {doc_id}')
        for page in range(1, count + 1):
            if (doc_id, page) not in indexed:
                key = hashlib.sha256(doc_id.encode()).hexdigest()[:20]
                pages.append({'doc_id': doc_id, 'page': page, 'source_sha256': source_hash,
                              'cache_path': f'pages/{key}/p{page:05d}.json'})
    manifest = {'settings': settings, 'settings_sha256': digest, 'baseline_sha256': sha256(baseline),
                'selection': 'All physical PDF pages absent from baseline chunks; no labels used',
                'documents': len(docs), 'pages': pages}
    write_json_atomic(Path(output) / 'manifest.json', manifest)
    return manifest


def valid_cache(path, entry, settings_hash):
    try:
        item = json.loads(Path(path).read_text(encoding='utf-8'))
        return (isinstance(item, dict) and isinstance(item.get('text'), str)
                and ((item.get('status') == 'ok' and bool(item['text'].strip()))
                     or (item.get('status') in ('blank', 'error') and not item['text'].strip()))
                and all(item.get(k) == entry[k] for k in ('doc_id', 'page', 'source_sha256'))
                and item.get('settings_sha256') == settings_hash
                and item.get('status') in ('ok', 'blank', 'error'))
    except (OSError, ValueError):
        return False


_ENGINE = None


def recover_page(entry, settings, settings_hash, pdf_dir, output, pilot=False):
    global _ENGINE
    started = time.monotonic()
    result = {k: entry[k] for k in ('doc_id', 'page', 'source_sha256')}
    result.update(settings_sha256=settings_hash, engine=settings['engine'], text='', status='error')
    try:
        import cv2
        import numpy as np
        import pymupdf
        from rapidocr import RapidOCR
        cv2.setNumThreads(1)
        if _ENGINE is None:
            _ENGINE = RapidOCR(params=settings['params'])
        with pymupdf.open(Path(pdf_dir) / entry['doc_id']) as doc:
            page = doc[entry['page'] - 1]
            scale = min(settings['dpi'] / 72, settings['max_render_side'] / max(page.rect.width, page.rect.height))
            pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False, colorspace=pymupdf.csRGB)
            result['render'] = {'width': pix.width, 'height': pix.height, 'effective_dpi': scale * 72,
                                'page_rect_points': list(page.rect), 'full_page': True}
            if pilot:
                png = Path(output) / 'pilot' / (Path(entry['cache_path']).parent.name + f'-p{entry["page"]}.png')
                png.parent.mkdir(parents=True, exist_ok=True)
                pix.save(png)
                result['pilot_image'] = str(png.resolve())
            pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            ocr = _ENGINE(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
        texts = list(ocr.txts) if ocr.txts is not None else []
        scores = [float(s) for s in ocr.scores] if ocr.scores is not None else []
        result.update(text='\n'.join(texts), status='ok' if any(t.strip() for t in texts) else 'blank',
                      confidence=sum(scores) / len(scores) if scores else None,
                      lines=[{'text': t, 'confidence': s, 'box': b.tolist()} for t, s, b in zip(texts, scores, ocr.boxes if ocr.boxes is not None else [])])
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['seconds'] = round(time.monotonic() - started, 3)
    write_json_atomic(Path(output) / entry['cache_path'], result)
    return result
