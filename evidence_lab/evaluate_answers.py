"""Small, predeclared generation comparison; lexical proxies are not correctness.

One immutable run, at most eight requests, no retries. A started request remains
recorded if interrupted; rerunning is refused because it might already be billed.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import urllib.request

from . import answering, baseline
from .corpus import write_json_atomic
from .experiments import freeze_split

MAX_REQUESTS = 8
BUDGET_USD = 0.10
PRICE_CEILINGS = {'prompt': 0.10, 'completion': 0.20}  # USD / million tokens
MODELS_URL = 'https://openrouter.ai/api/v1/models'


def normalize(text):
    """Lowercase, drop punctuation and collapse whitespace; keep articles/numbers."""
    return ' '.join(re.sub(r'[^\w\s]', ' ', str(text).casefold()).split())


def lexical_scores(prediction, reference):
    p, r = normalize(prediction), normalize(reference)
    pt, rt = p.split(), r.split()
    overlap = sum((Counter(pt) & Counter(rt)).values())
    f1 = 2 * overlap / (len(pt) + len(rt)) if pt or rt else 1.0
    return {'exact_normalized': float(p == r), 'token_f1': f1}


def select_cases(samples, split):
    reserved = sorted((s for s in samples if s['doc_id'] in split['reserved_docs']),
                      key=lambda s: s['source_row'])
    positive = [s for s in reserved if baseline._literal_list(s['evidence_pages'], 'pages', int)][:2]
    negative = [s for s in reserved if not baseline._literal_list(s['evidence_pages'], 'pages', int)][:2]
    if len(positive) != 2 or len(negative) != 2:
        raise ValueError('Need at least two answerable and two negative reserved rows')
    return positive + negative


def current_prices():
    """Read public pricing immediately before the run; never send credentials."""
    with urllib.request.urlopen(MODELS_URL, timeout=30) as response:
        models = json.load(response)['data']
    model = next((m for m in models if m['id'] == answering.DEFAULT_MODEL), None)
    if model is None:
        raise ValueError('Requested model has no current pricing')
    pricing = model['pricing']
    for field in ('prompt', 'completion'):
        value = float(pricing[field])
        if not math.isfinite(value) or value < 0 or value * 1e6 > PRICE_CEILINGS[field]:
            raise ValueError('Current model pricing exceeds the fixed ceiling or is unknown')
    # These calls are text-only, with no tools, audio or images. Refuse fixed fees.
    if float(pricing.get('request', 0)) != 0:
        raise ValueError('Per-request charges are not supported by this budget')
    return {'model': model['id'], 'pricing': pricing, 'source': MODELS_URL,
            'checked_utc': datetime.now(timezone.utc).isoformat()}


def request_bound(question, hits):
    messages = answering._messages(question, answering._build_sources(hits))
    # UTF-8 bytes upper-bound byte-level text tokens; generous framing allowance.
    tokens = len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) + 1024
    cost = (tokens * PRICE_CEILINGS['prompt'] + 384 * PRICE_CEILINGS['completion']) / 1e6
    return {'input_token_upper_bound': tokens, 'output_token_limit': 384,
            'cost_upper_bound_usd': cost}


def score_case(sample, result):
    gold = baseline._literal_list(sample['evidence_pages'], 'pages', int)
    ok = result.get('execution_status') == 'ok'
    answerable = bool(gold)
    scores = {'answerable': answerable, 'execution_success': ok,
              'failure': result.get('execution_status') if not ok else None}
    if answerable:
        scores['lexical'] = lexical_scores(result.get('answer') or '', sample['answer']) if ok else None
        scores['abstained'] = result.get('abstained') if ok else None
    else:
        scores['correct_abstention'] = bool(result.get('abstained')) if ok else None
        scores['false_answer'] = not result.get('abstained') if ok else None
    cited = {c['page'] for c in result.get('citations', []) if c['doc_id'] == sample['doc_id']}
    scores['cited_gold_page_fraction'] = len(cited & set(gold)) / len(cited) if cited and gold else None
    scores['citation_support_note'] = 'Gold-page overlap and valid ids are diagnostics only; entailment is not checked.'
    return scores


def summarize(cases):
    summary = {}
    for method in dict.fromkeys(c['method'] for c in cases):
        rows = [c for c in cases if c['method'] == method]
        done = [c for c in rows if 'scores' in c]
        positives = [c['scores'] for c in done if c['scores']['answerable']]
        negatives = [c['scores'] for c in done if not c['scores']['answerable']]
        scored = [s['lexical'] for s in positives if s['lexical'] is not None]
        summary[method] = {
            'attempted': len(rows), 'completed': len(done),
            'execution_failures': sum(not c['scores']['execution_success'] for c in done),
            'answerable_cases': len(positives), 'answerable_lexical_scored': len(scored),
            'answerable_abstentions': sum(s['abstained'] is True for s in positives),
            'mean_exact_normalized': baseline._mean([s['exact_normalized'] for s in scored]) if scored else None,
            'mean_token_f1': baseline._mean([s['token_f1'] for s in scored]) if scored else None,
            'negative_cases': len(negatives),
            'negative_abstentions': sum(s['correct_abstention'] is True for s in negatives),
            'negative_false_answers': sum(s['false_answer'] is True for s in negatives),
            'negative_failures': sum(not s['execution_success'] for s in negatives),
        }
    return summary


def run(root=Path('.'), allow_paid=False, *, retriever=None, answer_fn=None, price_loader=None):
    if not allow_paid:
        raise ValueError('Explicit --allow-paid is required; max eight requests and $0.10')
    root = Path(root)
    output = root / 'reports/answer-evaluation-v1'
    if output.exists():
        raise ValueError('Evaluation already exists; no retry or overwrite of possibly billed requests')
    samples = json.loads((root / 'data/benchmark/eligible.json').read_text('utf-8'))
    split = freeze_split(samples, root / 'reports/experiment-split-v1.json')
    selection = json.loads((root / 'reports/retrieval-v1/selection.json').read_text('utf-8'))
    if selection.get('reserved_consulted') is not False:
        raise ValueError('Retrieval selection must be frozen without consulting reserved results')
    from .retrieval import Retriever, METHODS
    if selection['method'] not in METHODS:
        raise ValueError('Unknown selected method')
    selected = select_cases(samples, split)
    methods = list(dict.fromkeys(('bm25', selection['method'])))
    output.mkdir(parents=True)  # Exclusive run reservation; concurrent callers cannot retry.
    target = output / 'report.json'
    report = {'status': 'predeclared', 'model': answering.DEFAULT_MODEL,
              'selection_rule': 'First two answerable and first two negative reserved rows in source_row order.',
              'source_rows': [s['source_row'] for s in selected], 'methods': methods,
              'documents': list(dict.fromkeys(s['doc_id'] for s in selected)),
              'samples_sha256': split['samples_sha256'],
              'retrieval_selection': selection, 'max_requests': MAX_REQUESTS,
              'budget_usd': BUDGET_USD, 'provider_max_price': PRICE_CEILINGS,
              'context': 'Top five chunks, document-scoped, identical formatting and max_tokens=384.',
              'limitations': [split['caveat'], 'Four cases per method: descriptive pilot only.',
                  'Exact normalized match and token F1 are lexical proxies, not answer correctness.',
                  'Citation existence and gold-page overlap do not establish entailment.',
                  'Failures and missing evidence are not counted as model abstentions.'],
              'cases': [], 'summary': {}}
    write_json_atomic(target, report)
    try:
        report['pricing'] = (price_loader or current_prices)()
        retriever = retriever or Retriever(root / 'data/baseline.sqlite', root / 'data/dense-v1')
        plan = []
        for method in methods:
            for sample in selected:
                hits = retriever.search(sample['question'], sample['doc_id'], 5, method)
                plan.append((method, sample, hits, request_bound(sample['question'], hits)))
        bound = sum(p[3]['cost_upper_bound_usd'] for p in plan)
        if len(plan) > MAX_REQUESTS or bound > BUDGET_USD or answering.OPTIONS['max_tokens'] != 384:
            raise ValueError('Preflight request or cost budget exceeded')
        report.update(status='running', aggregate_cost_upper_bound_usd=bound,
                      request_bounds=[{'method': m, 'source_row': s['source_row'], **b} for m,s,h,b in plan])
        write_json_atomic(target, report)
        for method, sample, hits, budget in plan:
            case = {'method': method, 'source_row': sample['source_row'], 'doc_id': sample['doc_id'],
                    'request_state': 'started', 'budget': budget}
            report['cases'].append(case)
            write_json_atomic(target, report)  # Persist before a request can be billed.
            result = (answer_fn or answering.answer_from_hits)(sample['question'], hits,
                model=answering.DEFAULT_MODEL, allow_paid=True, doc_id=sample['doc_id'], top_k=5,
                provider_max_price=PRICE_CEILINGS.copy())
            case.update(request_state='completed', result=result, scores=score_case(sample, result),
                        reference_answer=sample['answer'], expected_pages=sample['evidence_pages'])
            report['summary'] = summarize(report['cases'])
            costs = [c['result']['usage'].get('cost') for c in report['cases'] if 'result' in c]
            known = [float(c) for c in costs if isinstance(c, (float, int)) and math.isfinite(c)]
            report['reported_cost_usd'] = sum(known)
            report['cost_unreported_cases'] = len(costs) - len(known)
            write_json_atomic(target, report)
            if result['execution_status'] == 'provider_error':
                report['status'] = 'stopped_provider_error'
                break
        else:
            report['status'] = 'completed'
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__)
        write_json_atomic(target, report)
        raise
    write_json_atomic(target, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--allow-paid', action='store_true')
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(args.root / '.env', override=False)
    result = run(args.root, args.allow_paid)
    print(json.dumps({k: result.get(k) for k in ('status', 'summary', 'reported_cost_usd')}, indent=2))
