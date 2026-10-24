"""Conservative typed exact lookup over extracted cells; no reference answers imported."""
from __future__ import annotations
import re


def clean(value):
    return ' '.join(str(value).split())


def lookup(tables, row_id, column, conditions=None, unit=None):
    """Return one exact fact, or abstain. IDs are case sensitive; no unit conversion."""
    conditions = conditions or {}
    facts = [f for t in tables for f in t.get('facts', [])
             if clean(f['row_id']) == clean(row_id)
             and clean(f['column']).casefold() == clean(column).casefold()
             and all(clean(f.get('conditions', {}).get(k, '')) == clean(v) for k, v in conditions.items())
             and (unit is None or f.get('unit') == unit)]
    result = dict(status='abstain', value=None, unit=unit, provenance=None,
                  source_verified=False, reason='missing ID, column, condition or unit')
    if len(facts) != 1:
        if len(facts) > 1:
            result['reason'] = 'ambiguous: multiple matching cells or conditions'
        return result
    fact = facts[0]
    prov = fact.get('provenance') or {}
    if not prov.get('doc_id') or not prov.get('page') or not prov.get('cell_bbox'):
        result['reason'] = 'missing cell provenance'
        return result
    return dict(status='answer', value=fact['value'], unit=fact.get('unit'),
                conditions=fact.get('conditions', {}), provenance=prov,
                source_verified=bool(fact.get('source_verified', False)), reason='one exact matching fact')


def typed_values(column, text):
    """Declared narrow parsers; unsupported fields remain literal source strings."""
    text = clean(text)
    rules = {'Product Weight': (r'^(\d+(?:\.\d+)?)\s*Kg\b', 'kg'),
             'Viewable Size': (r'^(\d+(?:\.\d+)?)[”\"]\s*Diagonal', 'in'),
             'Response Time': (r'^(\d+(?:\.\d+)?)\s*ms\s*\(Gray to Gray\)', 'ms')}
    if column in rules:
        pattern, unit = rules[column]
        m = re.search(pattern, text, re.I)
        if not m:
            return []
        conditions = {'mode': 'Gray to Gray'} if column == 'Response Time' else {}
        return [(float(m[1]), unit, conditions)]
    if column == 'Light Intensity':
        # Both alternatives stay distinct. An unqualified lookup must abstain.
        pattern = r'^(\d+(?:\.\d+)?)\s*cd/m[²2]\s*\(typ\.\)\s*Optional for\s*(\d+(?:\.\d+)?)\s*cd/m[²2]\s*\(typ\.\)$'
        m = re.fullmatch(pattern, text)
        return [(float(m[i]), 'cd/m2', {'option': option, 'rating': 'typ.'})
                for i, option in [(1, 'standard'), (2, 'optional')]] if m else []
    return [(text, None, {})] if text else []


def normalize_docling(document, doc_id):
    """Transpose explicit model headers; preserve every used cell's geometry.

    Headerless continuation tables are deliberately not assigned inferred model IDs.
    """
    tables = []
    for table in document.get('tables', []):
        cells = table['data']['table_cells']
        grid = {}
        for cell in cells:
            for r in range(cell['start_row_offset_idx'], cell['end_row_offset_idx']):
                for c in range(cell['start_col_offset_idx'], cell['end_col_offset_idx']):
                    grid[r, c] = cell
        def val(r,c):
            return clean(grid.get((r,c), {}).get('text', ''))
        tid = table['self_ref']
        prov = table.get('prov', [{}])[0]
        out = dict(table_id=tid, facts=[])
        def add(row_id, column, cell, values, extra=None):
            for value, unit, conditions in values:
                conditions = {**conditions, **(extra or {})}
                out['facts'].append(dict(row_id=row_id, column=column, value=value, unit=unit,
                    conditions=conditions, source_text=cell['text'], source_verified=False,
                    provenance=dict(doc_id=doc_id, page=prov.get('page_no'), table_id=tid,
                        table_bbox=prov.get('bbox'), cell_bbox=cell.get('bbox'),
                        row=cell['start_row_offset_idx'], column=cell['start_col_offset_idx'],
                        row_id=row_id, field=column,
                        coordinate_system='Docling bbox coord_origin retained',
                        row_label_cell=grid.get((cell['start_row_offset_idx'],0)),
                        column_header_cell=grid.get((0,cell['start_col_offset_idx'])),
                        condition_cell=grid.get((cell['start_row_offset_idx'],1)) if extra else None)))
        headers = {c: val(0,c) for c in range(table['data']['num_cols'])}
        model_cols = {c: h for c,h in headers.items() if re.fullmatch(r'M\d+TF-XX', h)}
        for r in range(1, table['data']['num_rows']):
            if model_cols:
                label = val(r,0)
                for c,model in model_cols.items():
                    if (r,c) in grid:
                        add(model, label, grid[r,c], typed_values(label,val(r,c)))
            elif headers.get(0) == 'Resolution' and headers.get(2) == 'SDI Cable':
                if re.fullmatch(r'\d+x\d+', val(r,0)) and (r,2) in grid:
                    add(val(r,0), 'SDI Cable', grid[r,2], [(val(r,2),None,{})],
                        {'Frame Rate & Structure':val(r,1)})
        tables.append(out)
    return tables


def text_lookup(pages, row_id, column, conditions=None, unit=None):
    """Text-only baseline: native PDF line sequence with positional column offset.

    No bounding boxes or Docling table cells are available to this baseline.
    It is a limited deterministic baseline, not a competitive LLM table reader.
    """
    conditions = conditions or {}
    candidates = []
    for page in pages:
        lines = [clean(x) for x in page['text'].splitlines() if clean(x)]
        models = list(dict.fromkeys(x for x in lines if re.fullmatch(r'M\d+TF-XX',x)))
        if row_id in models:
            for i,line in enumerate(lines):
                if line.casefold() == clean(column).casefold():
                    offset = i + 1 + models.index(row_id)
                    if offset < len(lines):
                        candidates.extend((v,u,c,page['page']) for v,u,c in typed_values(column,lines[offset]))
        elif column == 'SDI Cable' and 'SDI Cable' in lines:
            for i,line in enumerate(lines[:-2]):
                if line == row_id:
                    candidates.append((lines[i+2],None,{'Frame Rate & Structure':lines[i+1]},page['page']))
    candidates = [x for x in candidates if (unit is None or x[1]==unit)
                  and all(x[2].get(k)==v for k,v in conditions.items())]
    if len(candidates)!=1:
        return dict(status='abstain',value=None,reason='missing or ambiguous text match',source_verified=False)
    v,u,c,p = candidates[0]
    return dict(status='answer',value=v,unit=u,conditions=c,provenance={'page':p,'method':'native line offsets'},source_verified=False)


def verify_reviewed_answer(answer, reviews):
    """Post-selection comparison with independently transcribed pilot source cells.

    This does not retrieve answers and is not a universal semantic verifier.
    Reviews must bind the original PDF hash, page, ID, field and conditions.
    """
    result = dict(answer)
    result['source_verified'] = False
    result['verification'] = {'status': 'not_reviewed', 'scope': 'reviewed pilot cells only'}
    if answer.get('status') != 'answer':
        return result
    p = answer.get('provenance') or {}
    if not p.get('source_sha256'):
        return result
    matches = [r for r in reviews
               if all(r.get(k) == p.get(k) for k in ['doc_id', 'page', 'source_sha256', 'row_id', 'field'])
               and r.get('conditions', {}) == answer.get('conditions', {})]
    if len(matches) != 1:
        return result
    review = matches[0]
    passed = review['value'] == answer['value'] and review.get('unit') == answer.get('unit')
    result['source_verified'] = passed
    result['verification'] = {'status': 'matches_review' if passed else 'mismatch',
                              'scope': 'reviewed pilot cells only', 'review_id': review['review_id'],
                              'method': review['method']}
    return result
