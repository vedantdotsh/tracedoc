"""Four explicitly authorized OpenRouter requests; no automatic retries."""
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

from . import answering
from .corpus import write_json_atomic
from .chunking import STRATEGIES


def run(root, allow_paid=False):
    if not allow_paid:
        raise ValueError('Pass --allow-paid to authorize up to four OpenRouter requests')
    root = Path(root).resolve()
    load_dotenv(root/'.env', override=False)
    out = root/'reports/chunking-v1'
    samples = json.loads((root/'data/benchmark/eligible.json').read_text('utf-8'))
    baseline = json.loads((out/'fixed.json').read_text('utf-8'))
    allowed = {c['source_row'] for c in baseline['cases'] if c['answerable']}
    sample = next(s for s in samples if s['source_row'] in allowed and 'Table' in s['evidence_sources'])
    target = out/'generation-pilot.json'
    if target.exists():
        raise ValueError('Pilot already exists; do not repeat billed requests automatically')
    pilot = {'source_row': sample['source_row'], 'question': sample['question'],
             'selection': 'First answerable development Table question in source order; not selected by retrieval scores.',
             'model': answering.DEFAULT_MODEL, 'max_requests': 4, 'results': {}}
    # Write before the first call: an interrupted run is never retried silently.
    write_json_atomic(target, pilot)
    for name in STRATEGIES:
        db = root/('data/baseline.sqlite' if name == 'fixed' else f'data/chunking-v1/{name}.sqlite')
        result = answering.answer(db, sample['question'], sample['doc_id'], allow_paid=True)
        pilot['results'][name] = result
        write_json_atomic(target, pilot)
        print(f'{name}: {result["status"]}; cost={result["usage"].get("cost", "unreported")}', flush=True)
        if result['status'] == 'provider_error':
            print('Provider failure: stopping pilot without retry.', flush=True)
            break
    return pilot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--allow-paid', action='store_true')
    args = parser.parse_args()
    run(args.root, args.allow_paid)
