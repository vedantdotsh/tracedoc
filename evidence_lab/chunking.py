"""Page-local chunking alternatives using the same PyMuPDF parser.

Heading detection is a font-size heuristic, not a learned layout model. Table
chunks preserve detected rows with column names; detection can miss/merge cells.
"""
import json
import re
from statistics import median

STRATEGIES = ('fixed', 'paragraph', 'heading', 'table')
MAX_DRAWING_ITEMS = 2000  # deterministic compute guard, selected without retrieval labels


def pack_units(units, words=180, overlap=30):
    """Pack whole units; long units fall back to bounded word windows.

    Overlap is used only within a long unit. Whole paragraphs are not repeated.
    """
    from .baseline import chunk_page
    chunk_page('', words, overlap)
    result, current = [], []
    for unit in units:
        tokens = unit.split()
        if not tokens:
            continue
        if len(current) + len(tokens) > words and current:
            result.append(' '.join(current))
            current = []
        if len(tokens) > words:
            result.extend(chunk_page(unit, words, overlap))
        else:
            current.extend(tokens)
    if current:
        result.append(' '.join(current))
    return result


def table_rows(headers, rows):
    """Serialize every row with repeated column headings; no inferred cell values."""
    result = []
    for row in rows:
        cells = [{'column': headers[i] if i < len(headers) and headers[i] else f'column_{i+1}',
                  'value': value} for i, value in enumerate(row)]
        if any(v is not None and str(v).strip() for v in row):
            result.append('Table row: ' + json.dumps(cells, ensure_ascii=False))
    return result


def page_chunks(page, strategy='fixed', words=180, overlap=30, diagnostics=None):
    from .baseline import chunk_page
    if strategy not in STRATEGIES:
        raise ValueError(f'unknown chunk strategy: {strategy}')
    chunk_page('', words, overlap)
    text = page.get_text('text', sort=True)
    if strategy == 'fixed':
        return chunk_page(text, words, overlap)
    if strategy == 'paragraph':
        # Blank lines delimit paragraphs where available; sentence boundaries
        # avoid cutting short sentences on PDFs lacking paragraph separators.
        units = re.split(r'\n\s*\n|(?<=[.!?])\s+(?=[A-Z0-9])', text)
        return pack_units(units, words, overlap)
    blocks = page.get_text('dict', sort=True)['blocks']
    lines = [line for b in blocks if b['type'] == 0 for line in b['lines']]
    sizes = [s['size'] for line in lines for s in line['spans'] if s['text'].strip()]
    typical = median(sizes) if sizes else 0
    if strategy == 'heading':
        result, body, heading = [], [], ''
        def flush():
            result.extend((heading + '\n' + part).strip()
                          for part in pack_units(body, words, overlap))
        for line in lines:
            value = ' '.join(s['text'] for s in line['spans']).strip()
            is_heading = (bool(value) and len(value.split()) <= 25 and
                          max(s['size'] for s in line['spans']) >= typical * 1.2)
            if is_heading:
                flush()
                body = []
                heading = value
                result.append(value)  # preserve headings with no following body
            else:
                body.append(value)
        flush()
        return result
    drawings = page.get_drawings()
    item_count = sum(len(p['items']) for p in drawings)
    if item_count > MAX_DRAWING_ITEMS:
        if diagnostics is not None:
            diagnostics.append({'page': page.number + 1,
                                'reason': 'table_graphics_complexity_limit',
                                'drawing_items': item_count, 'limit': MAX_DRAWING_ITEMS,
                                'fallback': 'fixed_text_chunks'})
        return chunk_page(text, words, overlap)
    tables = page.find_tables(paths=drawings).tables  # other errors propagate
    table_boxes = [t.bbox for t in tables]
    plain = []
    for line in lines:
        x0, y0, x1, y1 = line['bbox']
        inside = any(a <= (x0+x1)/2 <= c and b <= (y0+y1)/2 <= d
                     for a, b, c, d in table_boxes)
        if not inside:
            plain.append(' '.join(s['text'] for s in line['spans']))
    result = pack_units(plain, words, overlap)
    for table in tables:
        rows = table.extract()
        # Header may be external; otherwise extract() includes it as first row.
        body = rows if table.header.external else rows[1:]
        result.extend(table_rows(table.header.names, body))
        if not body and rows:
            result.extend(table_rows([], rows))
    return result
