"""Reproducible notebook helpers. Labels are used only after retrieval."""
import hashlib
import html
import json
from pathlib import Path

from . import baseline
from .corpus import write_json_atomic


def freeze_split(samples, path, seed='document-evidence-v1'):
    """Split whole documents, keeping duplicate questions in the same partition.

    All eligible rows were included in the earlier descriptive baseline; this is
    reserved for future tuning, NOT a previously unseen holdout.
    """
    fingerprint = hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest()
    docs = sorted({s['doc_id'] for s in samples},
                  key=lambda d: hashlib.sha256(f'{seed}:{d}'.encode()).hexdigest())
    cut = max(1, int(len(docs) * 0.7))
    result = {'seed': seed, 'samples_sha256': fingerprint,
              'development_docs': docs[:cut], 'reserved_docs': docs[cut:],
              'caveat': 'Previously evaluated in descriptive baseline; not an unseen test set.'}
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text('utf-8')) != result:
            raise ValueError('Frozen split differs; use a new version, never overwrite it silently')
    else:
        write_json_atomic(path, result)
    return result


def select_samples(samples, split, partition='development'):
    if partition not in ('development', 'reserved'):
        raise ValueError('partition must be development or reserved')
    docs = set(split[f'{partition}_docs'])
    return [s for s in samples if s['doc_id'] in docs]


def inspect_case(db, sample, top_k=5, generation=None):
    """Return an escaped HTML case card; gold is never passed to search/generation."""
    hits = baseline.search(db, sample['question'], sample['doc_id'], top_k=baseline.ALL_CHUNKS)
    pages = list(dict.fromkeys(h['page'] for h in hits))[:top_k]
    expected = baseline._literal_list(sample['evidence_pages'], 'pages', int)
    metrics = baseline.score_pages(expected, pages) if expected else {'unanswerable': True}
    def esc(value):
        return html.escape(str(value))
    parts = [f'<h3>{esc(sample["question"])}</h3>',
             f'<p>Document: {esc(sample["doc_id"])}</p>',
             f'<p>Expected pages: {esc(expected)} · Retrieved pages: {esc(pages)}</p>',
             f'<p>Reference answer (evaluation only): {esc(sample.get("answer"))}</p>',
             f'<pre>{esc(json.dumps(metrics, indent=2))}</pre>',
             '<p>Below: top five chunks. Page metrics above rank five distinct pages; '
             'these are different context budgets.</p>']
    for rank, hit in enumerate(hits[:top_k], 1):
        parts.append(f'<details><summary>Chunk {rank}, physical page {hit["page"]}, '
                     f'BM25 {hit["score"]:.5f}</summary><pre style="white-space:pre-wrap">'
                     f'{esc(hit["text"])}</pre></details>')
    if generation is None:
        parts.append('<p>Generated answer: not run for this case.</p>')
    else:
        parts.append(f'<h4>Generated answer</h4><pre style="white-space:pre-wrap">'
                     f'{esc(json.dumps({k: generation.get(k) for k in ("status", "answer", "citations", "model_reported", "provider", "timings")}, indent=2))}</pre>')
    return '\n'.join(parts)
