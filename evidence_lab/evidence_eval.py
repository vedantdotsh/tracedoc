"""Located-evidence search and deterministic evaluation in four separate layers.

1. retrieval        gold evidence found in the top-k (recall, hit, MRR)
2. citation validity cited chunk exists, was provided to the answerer, matches its
                    doc/page, and (if source_dir is given) the source hash still matches
3. expected facts   gold value (+unit) appears verbatim in the answer
4. entailment       human verdicts only; never inferred, never part of an automated score
"""
import json
import re
import time
from collections import Counter
from contextlib import closing
from pathlib import Path

from . import baseline
from .sources import sha256_file

VERDICTS = ('supported', 'partial', 'unsupported')


def search(db_path, query, doc_id=None, top_k=5, retriever=None, method='bm25'):
    """baseline.search (or a Retriever) plus source hash, location, system/revision, units."""
    hits = (retriever.search(query, doc_id, top_k, method) if retriever
            else baseline.search(db_path, query, doc_id, top_k))
    with closing(baseline._connect(db_path)) as con:
        for hit in hits:
            row = con.execute('SELECT kind, source_sha256, system, revision, location, extra '
                              'FROM evidence WHERE chunk_id = ?', (hit['chunk_id'],)).fetchone()
            if row is None:
                raise ValueError(f"chunk {hit['chunk_id']} has no evidence metadata")
            kind, sha, system, revision, location, extra = row
            hit['evidence'] = {'kind': kind, 'source_sha256': sha, 'system': system,
                               'revision': revision, 'location': json.loads(location),
                               'details': json.loads(extra), 'support_verified': False}
    return hits


def _matches(location, gold):
    return all(location.get(k) == v for k, v in gold.items())


def fact_in(text, fact):
    pattern = rf'(?<![\w.]){re.escape(fact["value"])}'
    if fact.get('unit'):
        pattern += rf'\s?{re.escape(fact["unit"])}'
    return re.search(pattern + r'(?!\w)', text) is not None


def _check_case(i, case, locations):
    if (not isinstance(case, dict) or not isinstance(case.get('id'), str)
            or not isinstance(case.get('question'), str)):
        raise ValueError(f'case {i}: needs string id and question')
    gold, facts = case.get('gold_evidence', []), case.get('expected_facts', [])
    if not isinstance(gold, list) or not isinstance(facts, list):
        raise ValueError(f"case {case['id']}: gold_evidence and expected_facts must be lists")
    for g in gold:
        if not (isinstance(g, dict) and isinstance(g.get('doc_id'), str)
                and isinstance(g.get('location'), dict) and g['location']):
            raise ValueError(f"case {case['id']}: gold_evidence needs doc_id and a non-empty location")
        if not any(_matches(loc, g['location']) for loc in locations.get(g['doc_id'], [])):
            raise ValueError(f"case {case['id']}: gold evidence matches no indexed evidence: {g}")
    for f in facts:
        if (not isinstance(f, dict) or not isinstance(f.get('value'), str) or not f['value']
                or not isinstance(f.get('unit'), (str, type(None)))):
            raise ValueError(f"case {case['id']}: expected_facts need a string value and optional unit")
    if facts and not gold:
        raise ValueError(f"case {case['id']}: expected_facts on an unanswerable case")
    return case['id']


def _citation(con, cited, provided, source_dir):
    chunk_id = cited.get('chunk_id') if isinstance(cited, dict) else None
    row = con.execute('SELECT c.doc_id, c.page, c.text, e.source_sha256, e.location '
                      'FROM chunks c JOIN evidence e ON e.chunk_id = c.chunk_id '
                      'WHERE c.chunk_id = ?', (chunk_id,)).fetchone()
    out = {'chunk_id': chunk_id, 'exists': row is not None, 'provided': chunk_id in provided}
    if row:
        doc_id, page, text, sha, location = row
        location = json.loads(location)
        expected_page = int(page) if 'page' in location else None
        # Legacy benchmark answers use ordinal pages; mixed answers carry real locators.
        page_ok = cited.get('page', int(page)) == int(page) if 'location' not in cited else cited.get('page') == expected_page
        locator_ok = 'location' not in cited or _matches(cited['location'], location)
        hash_ok = cited.get('source_sha256', sha) == sha
        out.update(consistent=cited.get('doc_id', doc_id) == doc_id
                   and page_ok and locator_ok and hash_ok,
                   location=location, text=text, source_hash_ok=None)
        if source_dir is not None:
            path = Path(source_dir) / doc_id
            out['source_hash_ok'] = path.is_file() and sha256_file(path) == sha
    out['valid'] = bool(row) and out['provided'] and out['consistent'] \
        and out['source_hash_ok'] is not False
    return out


def _check_answer(con, answer, answerable, facts, hits, source_dir):
    abstained = answer.get('abstained')
    if not isinstance(abstained, bool) or answer.get('execution_status') in ('provider_error', 'invalid_response'):
        return {'scored': False, 'abstained': None, 'execution_status': answer.get('execution_status')}
    provided = {s['chunk_id'] for s in answer.get('sources', [])}
    cites = [_citation(con, c, provided, source_dir) for c in answer.get('citations', [])]
    text = answer.get('answer') or ''
    out = {'scored': True, 'abstained': abstained, 'decision_correct': abstained == (not answerable),
           'citations': cites,
           'citations_valid': None if abstained else bool(cites) and all(c['valid'] for c in cites),
           'facts': None, 'facts_ok': None}
    if answerable and not abstained and facts:
        checks = [{**f, 'in_answer': fact_in(text, f),
                   'value_in_retrieved': any(fact_in(h['text'], {'value': f['value']}) for h in hits)}
                  for f in facts]
        out.update(facts=checks, facts_ok=all(c['in_answer'] for c in checks))
    return out


def _run_case(con, db_path, case, top_k, answer, review, source_dir, retriever, method):
    gold, facts = case.get('gold_evidence', []), case.get('expected_facts', [])
    start = time.perf_counter()
    hits = search(db_path, case['question'], case.get('doc_id'), top_k, retriever, method)
    latency_ms = (time.perf_counter() - start) * 1000
    ranks = [next((r for r, h in enumerate(hits, 1) if h['doc_id'] == g['doc_id']
                   and _matches(h['evidence']['location'], g['location'])), None) for g in gold]
    found = [r for r in ranks if r]
    result = {'id': case['id'], 'question': case['question'], 'answerable': bool(gold),
              'latency_ms': latency_ms,
              'retrieved': [{'rank': r, 'chunk_id': h['chunk_id'], 'doc_id': h['doc_id'],
                             'location': h['evidence']['location']} for r, h in enumerate(hits, 1)],
              'retrieval': ({'recall': len(found) / len(gold), 'hit': float(bool(found)),
                             'mrr': 1 / min(found) if found else 0.0} if gold else None),
              'answer': None, 'human_entailment': None}
    if answer is not None:
        result['answer'] = _check_answer(con, answer, bool(gold), facts, hits, source_dir)
    done = result['answer']
    answered = bool(done and done['scored'] and not done['abstained'])
    if review is not None:
        if not answered:
            raise ValueError(f"case {case['id']}: review supplied but the case was not answered")
        if review.get('verdict') not in VERDICTS:
            raise ValueError(f'review verdict must be one of {VERDICTS}')
        if not isinstance(review.get('reviewer'), str) or not review['reviewer']:
            raise ValueError('review needs a reviewer name')
    if answered:
        result['human_entailment'] = ({'status': 'pending'} if review is None else
                                      {'status': 'reviewed', 'verdict': review['verdict'],
                                       'reviewer': review['reviewer']})
    return result


def _summary(results):
    retrieval = [r['retrieval'] for r in results if r['retrieval']]
    supplied = [r['answer'] for r in results if r['answer']]
    scored = [a for a in supplied if a['scored']]
    answered = [a for a in scored if not a['abstained']]
    facts = [a for a in answered if a['facts_ok'] is not None]
    entailment = [r['human_entailment'] for r in results if r['human_entailment']]
    reviewed = [e for e in entailment if e['status'] == 'reviewed']
    return {
        'cases': len(results), 'answerable': len(retrieval),
        'unanswerable': len(results) - len(retrieval),
        'retrieval': {k: baseline._mean([r[k] for r in retrieval]) for k in ('recall', 'hit', 'mrr')}
                     | {'note': 'answerable cases only; unanswerable cases are not retrieval scores'},
        'answers': {'supplied': len(supplied), 'scored': len(scored),
                    'unscored_failures': len(supplied) - len(scored), 'answered': len(answered),
                    'decision_correct': sum(a['decision_correct'] for a in scored)},
        'citations': {'answered': len(answered),
                      'all_valid': sum(a['citations_valid'] is True for a in answered)},
        'facts': {'checked': len(facts), 'passed': sum(a['facts_ok'] for a in facts)},
        'human_entailment': {'required': len(entailment), 'reviewed': len(reviewed),
                             'pending': len(entailment) - len(reviewed),
                             'verdicts': dict(Counter(e['verdict'] for e in reviewed))},
        'latency_ms': {'p50': baseline._percentile([r['latency_ms'] for r in results], .5),
                       'p95': baseline._percentile([r['latency_ms'] for r in results], .95)},
    }


def evaluate(db_path, cases, top_k=5, answers=None, reviews=None, source_dir=None,
             retriever=None, method='bm25') -> dict:
    """answers: {case id: answering-style result}; reviews: {case id: {'verdict','reviewer'}}.

    Never calls a model. Retrieval uses the question text only; gold is never searched.
    """
    baseline._check_int('top_k', top_k, 1)
    if not isinstance(cases, list):
        raise ValueError('cases must be a list')
    answers, reviews = answers or {}, reviews or {}
    with closing(baseline._connect(db_path)) as con:
        locations = {}
        for doc_id, location in con.execute('SELECT doc_id, location FROM evidence'):
            locations.setdefault(doc_id, []).append(json.loads(location))
        meta = dict(con.execute('SELECT key, value FROM meta'))
        ids = [_check_case(i, c, locations) for i, c in enumerate(cases)]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate case ids')
        unknown = sorted((set(answers) | set(reviews)) - set(ids))
        if unknown:
            raise ValueError(f'answers/reviews for unknown case ids: {unknown}')
        for case in cases:
            answer = answers.get(case['id'])
            if answer is not None and answer.get('question', case['question']) != case['question']:
                raise ValueError(f"Saved answer question mismatch: {case['id']}")
        results = [_run_case(con, db_path, c, top_k, answers.get(c['id']), reviews.get(c['id']),
                             source_dir, retriever, method) for c in cases]
    return {'config': {'top_k': top_k, 'method': method, 'retrieval_unit': 'chunk@k (evidence record)',
                       'gold_match': 'doc_id equal and every gold location key/value equal',
                       'index': meta},
            'limits': ['Citation validity is existence/provenance, not support.',
                       'Fact checks are verbatim value(+unit) matches, not correctness.',
                       'Entailment is recorded only from human reviews.'],
            'summary': _summary(results), 'cases': results}
