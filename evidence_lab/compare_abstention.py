"""Offline counterfactual replay: identical recorded model answers and evidence.

No new retrieval or generation, and no claim of a new model run. Labels are read
only after check() has produced its decision, for descriptive tradeoff counts.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from . import abstention
from .corpus import write_json_atomic


def replay_case(case):
    baseline = deepcopy(case['result'])
    decision = abstention.check(baseline['question'], baseline['sources'])
    gated = deepcopy(baseline)
    gated['evidence_gate'] = decision
    gated['replay_original_generation'] = True
    # Failed/invalid original calls cannot become successful counterfactual answers.
    if decision['abstain'] and baseline['execution_status'] == 'ok':
        gated.update(status='gate_abstained', execution_status='not_run',
                     abstained=True, abstention_origin='evidence_gate', citations=[],
                     answer='The evidence did not pass the optional evidence checks.')
    else:
        gated['abstention_origin'] = 'model' if baseline['status'] == 'abstained' else None
    return {'method': case['method'], 'source_row': case['source_row'],
            'answerable': case['scores']['answerable'],
            'reference_answer': case.get('reference_answer'),
            'baseline': baseline, 'gated': gated, 'decision': decision}


def summarize(cases):
    results = {}
    for method in dict.fromkeys(c['method'] for c in cases):
        rows = [c for c in cases if c['method'] == method]
        result = {}
        for mode in ('baseline', 'gated'):
            failures = [c for c in rows if c['baseline']['execution_status'] != 'ok']
            valid = [c for c in rows if c not in failures]
            result[mode] = {
                'cases': len(rows), 'original_execution_failures': len(failures),
                'gate_abstentions': sum(c[mode]['status'] == 'gate_abstained' for c in valid),
                'model_abstentions': sum(c[mode]['status'] == 'abstained' for c in valid),
                'answerable_cases': sum(c['answerable'] for c in valid),
                'false_abstentions': sum(c['answerable'] and c[mode]['abstained'] is True for c in valid),
                'negative_cases': sum(not c['answerable'] for c in valid),
                'negative_abstentions': sum(not c['answerable'] and c[mode]['abstained'] is True for c in valid),
                'negative_false_answers': sum(not c['answerable'] and c[mode]['abstained'] is False for c in valid),
            }
        results[method] = result
    return results


def run(root=Path('.')):
    root = Path(root)
    source = root / 'reports/answer-evaluation-v1/report.json'
    raw = source.read_bytes()
    original = json.loads(raw)
    cases = [replay_case(c) for c in original['cases'] if 'result' in c]
    report = {
        'experiment': 'offline counterfactual replay, not a new model run',
        'gate_version': abstention.VERSION, 'provenance': abstention.PROVENANCE,
        'gate_sha256': hashlib.sha256(Path(abstention.__file__).read_bytes()).hexdigest(),
        'baseline_report': 'reports/answer-evaluation-v1/report.json',
        'baseline_sha256': hashlib.sha256(raw).hexdigest(),
        'additional_calls': 0, 'additional_cost_usd': 0,
        'design': 'Initial v1 replay blocked zero cases. Independent review then corrected an overly broad arithmetic exemption: explicit year/identifier scopes remain required. A test-driven publication-date exclusion was also added. V2 is post-review exploratory, not a predeclared held-out experiment. Same recorded sources, prompts, and raw generations; only deterministic evidence decisions differ.',
        'limitations': [
            'Eight records are four questions from one document, not independent accuracy estimates.',
            'Reserved documents were already exposed in an earlier baseline; not an unseen test.',
            'Passing the gate does not prove answer correctness, citation support, or entailment.',
            'Exact-context conflicts only span different documents; same-document ambiguity is missed.',
            'Unrelated numbers may satisfy the presence check; derived questions bypass quantity/conflict checks but retain explicit scopes.',
            'False abstentions use benchmark answerability, which does not prove retrieved evidence suffices.',
            'Replay preserves historical raw responses, timings and usage for audit, not new incurred cost.',
            'Original provider failures remain failures even if a gate would reject their evidence.',
        ], 'cases': cases, 'summary': summarize(cases),
    }
    output = root / 'reports/abstention-v2/report.json'
    if output.exists():
        existing = json.loads(output.read_text('utf-8'))
        if existing != report:
            raise ValueError('Existing comparison differs; use a new version rather than overwrite')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output, report)
    return report


def coverage(root=Path('.')):
    """Gate-only BM25 audit on all reserved questions; no model correctness score."""
    from . import answering, baseline
    root = Path(root)
    samples = json.loads((root / 'data/benchmark/eligible.json').read_text('utf-8'))
    split = json.loads((root / 'reports/experiment-split-v1.json').read_text('utf-8'))
    selected = sorted((s for s in samples if s['doc_id'] in split['reserved_docs']), key=lambda s: s['source_row'])
    rows = []
    for sample in selected:
        sources = answering._build_sources(baseline.search(root / 'data/baseline.sqlite', sample['question'], sample['doc_id'], 5))
        decision = abstention.check(sample['question'], sources)
        # Labels enter only after the question/evidence decision.
        answerable = bool(baseline._literal_list(sample['evidence_pages'], 'pages', int))
        rows.append({'source_row': sample['source_row'], 'question': sample['question'],
                     'sources': sources, 'decision': decision, 'answerable': answerable})
    from collections import Counter
    report = {'experiment': 'gate-only reserved BM25 coverage; no generation',
        'gate_version': abstention.VERSION, 'gate_sha256': hashlib.sha256(Path(abstention.__file__).read_bytes()).hexdigest(),
        'scope': 'document-scoped BM25 top five chunks; 24000 character model context budget',
        'additional_calls': 0, 'additional_cost_usd': 0,
        'limitations': 'Post-review descriptive audit on previously exposed reserved rows. Pass is not an answer or evidence-support score. Benchmark-positive rejection may reflect missing retrieved evidence.',
        'summary': {'questions': len(rows), 'documents': len({s['doc_id'] for s in selected}),
            'answerable': sum(r['answerable'] for r in rows),
            'negative': sum(not r['answerable'] for r in rows),
            'answerable_rejected': sum(bool(r['sources']) and r['answerable'] and r['decision']['abstain'] for r in rows),
            'negative_rejected': sum(bool(r['sources']) and not r['answerable'] and r['decision']['abstain'] for r in rows),
            'no_sources': sum(not r['sources'] for r in rows),
            'answerable_no_sources': sum(r['answerable'] and not r['sources'] for r in rows),
            'negative_no_sources': sum(not r['answerable'] and not r['sources'] for r in rows),
            'reason_counts': dict(Counter(code for r in rows if r['sources'] for code in r['decision']['reasons']))},
        'cases': rows}
    target = root / 'reports/abstention-v2/coverage.json'
    if target.exists() and json.loads(target.read_text('utf-8')) != report:
        raise ValueError('Coverage differs; use a new version')
    if not target.exists():
        write_json_atomic(target, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    args = parser.parse_args()
    print(json.dumps(run(args.root)['summary'], indent=2))
