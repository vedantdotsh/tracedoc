"""Run the development-set chunking comparison; no model calls or gold indexing."""
import argparse
import json
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from . import baseline
from .chunking import STRATEGIES
from .corpus import write_json_atomic
from .experiments import freeze_split, select_samples


def run(root, run_name='chunking-v1', resume=False):
    if not re.fullmatch(r'chunking-[a-z0-9-]+', run_name):
        raise ValueError('run name must start chunking- and contain only lowercase letters, digits or hyphens')
    root = Path(root).resolve()
    samples = json.loads((root/'data/benchmark/eligible.json').read_text('utf-8'))
    split = freeze_split(samples, root/'reports/experiment-split-v1.json')
    dev = select_samples(samples, split)
    out = root/'reports'/run_name
    out.mkdir(parents=True, exist_ok=True)
    summary = {'split_sha256': split['samples_sha256'], 'partition': 'development',
               'questions': len(dev), 'top_k_pages': 5, 'methods': {},
               'note': 'Same PDFs/parser/BM25. Structure methods also change serialization and length; not a pure boundary-only ablation.'}
    with closing(baseline._connect(root/'data/baseline.sqlite')) as con:
        expected_meta = dict(con.execute('SELECT key,value FROM meta'))
    for strategy in STRATEGIES:
        target = out/f'{strategy}.json'
        if target.exists():
            if not resume:
                raise ValueError(f'{target} already exists; preserve it and use a new experiment version')
            report = json.loads(target.read_text('utf-8'))
            config = report['config']
            meta = config['index']
            if (meta['sources_sha256'] != expected_meta['sources_sha256']
                    or meta.get('strategy', 'fixed') != strategy
                    or meta.get('words') != '180' or meta.get('overlap') != '30'
                    or config.get('top_k') != 5 or config.get('scope') != 'document-scoped'
                    or [c['source_row'] for c in report['cases']] != [s['source_row'] for s in dev]):
                raise ValueError('Existing report is incompatible with the current corpus/split')
            summary['methods'][strategy] = {k: report[k] for k in
                ('answerable','unanswerable','mean_recall','mean_hit','mrr','latency_ms','build')}
            print(f'{strategy}: retained completed report', flush=True)
            continue
        db = root/'data/baseline.sqlite' if strategy == 'fixed' else root/f'data/{run_name}/{strategy}.sqlite'
        start = time.perf_counter()
        if strategy != 'fixed':
            if strategy == 'table':
                stats = baseline.build_index(root/'data/documents', db, strategy=strategy, workers=3)
            else:
                stats = baseline.build_index(root/'data/documents', db, strategy=strategy)
        else:
            stats = {'note': 'Existing baseline reused; build time not remeasured'}
        build_s = time.perf_counter()-start if strategy != 'fixed' else None
        with closing(baseline._connect(db)) as con:
            meta = dict(con.execute('SELECT key,value FROM meta'))
            if meta['sources_sha256'] != expected_meta['sources_sha256']:
                raise ValueError('Corpus differs between methods')
            lengths = [len(row[0].split()) for row in con.execute('SELECT text FROM chunks')]
        print(f'{strategy}: indexed {len(lengths)} chunks; evaluating {len(dev)} development questions', flush=True)
        report = baseline.evaluate(db, dev)
        report['build'] = {**stats, 'elapsed_s': build_s, 'index_bytes': db.stat().st_size,
                           'mean_chunk_words': sum(lengths)/len(lengths),
                           'max_chunk_words': max(lengths),
                           'chunks_over_180_words': sum(n > 180 for n in lengths)}
        write_json_atomic(target, report)
        summary['methods'][strategy] = {k: report[k] for k in
            ('answerable','unanswerable','mean_recall','mean_hit','mrr','latency_ms','build')}
        print(f'{strategy}: recall={report["mean_recall"]:.4f}, hit={report["mean_hit"]:.4f}', flush=True)
    base = json.loads((out/'fixed.json').read_text('utf-8'))
    for strategy in STRATEGIES[1:]:
        alt = json.loads((out/f'{strategy}.json').read_text('utf-8'))
        pairs = [(a,b) for a,b in zip(base['cases'], alt['cases']) if a['answerable']]
        assert all(a['source_row']==b['source_row'] for a,b in pairs)
        summary['methods'][strategy]['paired_recall_vs_fixed'] = {
            'wins': sum(b['recall'] > a['recall'] for a,b in pairs),
            'losses': sum(b['recall'] < a['recall'] for a,b in pairs),
            'ties': sum(b['recall'] == a['recall'] for a,b in pairs)}
    write_json_atomic(out/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--run-name', default='chunking-v1')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.root, args.run_name, args.resume)
