"""Frozen candidate budgets; select on development, then measure reserved once."""
import argparse
import json
import re
import time
from pathlib import Path
from . import baseline
from .retrieval import Retriever, METHODS, EMBED_MODEL, RERANK_MODEL
from .experiments import freeze_split, select_samples
from .corpus import write_json_atomic

def evaluate(retriever, samples, method):
    cases = []
    for sample in samples:
        start = time.perf_counter()
        hits = retriever.search(sample['question'], sample['doc_id'], 30, method)
        pages = list(dict.fromkeys(h['page'] for h in hits))[:5]
        expected = baseline._literal_list(sample['evidence_pages'], 'pages', int)
        case = {'source_row': sample['source_row'], 'doc_id': sample['doc_id'],
                'question': sample['question'], 'expected_pages': expected,
                'retrieved_pages': pages, 'latency_ms': (time.perf_counter()-start)*1000,
                'answerable': bool(expected)}
        if expected:
            case.update(baseline.score_pages(expected, pages))
        cases.append(case)
    scored = [c for c in cases if c['answerable']]
    return {'method': method, 'answerable':len(scored), 'unanswerable':len(cases)-len(scored),
            'mean_recall':baseline._mean([c['recall'] for c in scored]),
            'mean_hit':baseline._mean([c['hit'] for c in scored]),
            'mrr':baseline._mean([c['mrr'] for c in scored]),
            'latency_ms':{'p50':baseline._percentile([c['latency_ms'] for c in cases],.5),
                          'p95':baseline._percentile([c['latency_ms'] for c in cases],.95)},
            'cases':cases}

def select_method(reports):
    return max(METHODS, key=lambda m: (reports[m]['mean_recall'], reports[m]['mrr'], -METHODS.index(m)))

def run(root=Path('.'), run_name='retrieval-v1'):
    root = Path(root)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', run_name):
        raise ValueError('run_name must be a directory name')
    output = root/'reports'/run_name
    if output.exists():
        raise ValueError('Run already exists; choose a new run name')
    samples = json.loads((root/'data/benchmark/eligible.json').read_text('utf-8'))
    split = freeze_split(samples, root/'reports/experiment-split-v1.json')
    retriever = Retriever(root/'data/baseline.sqlite', root/'data/dense-v1')
    manifest = retriever.build()
    output.mkdir(parents=True)
    config = {'embedding_model':EMBED_MODEL,'reranker':RERANK_MODEL,'index':manifest,'split':split,
              'candidate_chunks':30,'page_k':5,'rrf_constant':60,'scope':'document-scoped',
              'caveat':split['caveat'], 'negative_rows':'Excluded from retrieval metrics; abstention is measured separately.',
              'control':'Every method returns at most 30 chunks before collapsing to five distinct pages. Historical baseline scans all chunks and is a separate measurement.'}
    write_json_atomic(output/'config.json', config)
    dev = {}
    for method in METHODS:
        print('development '+method, flush=True)
        dev[method] = evaluate(retriever, select_samples(samples,split), method)
        write_json_atomic(output/f'development-{method}.json',dev[method])
    winner = select_method(dev)
    write_json_atomic(output/'selection.json', {'method':winner,'rule':'Maximum development mean recall, then MRR, then simpler method order.', 'reserved_consulted':False})
    for method in dict.fromkeys(('bm25',winner)):
        print('reserved '+method, flush=True)
        report = evaluate(retriever, select_samples(samples,split,'reserved'),method)
        write_json_atomic(output/f'reserved-{method}.json',report)
    return {'selected':winner,'output':str(output)}

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('.')); p.add_argument('--run-name',default='retrieval-v1')
    a=p.parse_args(); print(run(a.root,a.run_name))
