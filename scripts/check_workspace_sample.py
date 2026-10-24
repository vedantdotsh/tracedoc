"""Explicit live smoke check on the synthetic sample; max $0.05 generation reservation."""
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from evidence_lab.workspace import Workspace
from evidence_lab import answering, evaluate_answers, evidence_eval
from evidence_lab.corpus import write_json_atomic

load_dotenv(override=False)
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--index', default='data/workspace')
p.add_argument('--report', type=Path, default=Path('reports/workspace-sample-live.json'))
p.add_argument('--answers', type=Path, default=Path('data/workspace-answers.json'))
a = p.parse_args()
ws = Workspace(a.index, allow_paid=True)
cases = json.loads(Path('data/workspace-cases.json').read_text())
evaluate_answers.current_prices()
answers, reserved = {}, 0.0
for case in cases:
    hits = ws.search(case['question'], method='hybrid')
    cost = evaluate_answers.request_bound(case['question'], hits)['cost_upper_bound_usd']
    if cost > .01 or reserved + cost > .05:
        raise RuntimeError('Sample generation budget exceeded')
    reserved += cost
    result = answering.answer_from_hits(case['question'], hits, allow_paid=True,
                                        provider_max_price=evaluate_answers.PRICE_CEILINGS.copy())
    result.pop('raw_response', None)
    result.pop('messages', None)
    answers[case['id']] = result
write_json_atomic(a.answers, answers)
class Adapter:
    def search(self, query, doc_id, top_k, method):
        return ws.search(query, doc_id, top_k, method)
report = evidence_eval.evaluate(ws.db, cases, answers=answers, source_dir='data/workspace-sample',
                                retriever=Adapter(), method='hybrid')
report['sample_note'] = 'Synthetic integration sample; not fuel-cell domain validation.'
report['generation_reserved_usd'] = reserved
report['generation_reported_usd'] = sum(r.get('usage', {}).get('cost', 0) or 0 for r in answers.values())
write_json_atomic(a.report, report)
print(json.dumps(report['summary'], indent=2))
print('Reported generation cost:', report['generation_reported_usd'])
