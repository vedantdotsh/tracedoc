import hashlib
import json

import pytest

from evidence_lab import evidence_eval
from evidence_lab.__main__ import main

CASES = [
    {'id': 'material', 'question': 'material SS316',
     'gold_evidence': [{'doc_id': 'pumps.xlsx', 'location': {'sheet': 'Pumps', 'row': 2}}],
     'expected_facts': [{'value': '3.2', 'unit': 'bar'}]},
    {'id': 'valve', 'question': 'valve close trip signal',
     'gold_evidence': [{'doc_id': 'valves.md', 'location': {'requirement_id': 'SYS-12'}}]},
    {'id': 'missed', 'question': 'material SS316',
     'gold_evidence': [{'doc_id': 'reqs.csv', 'location': {'requirement_id': 'REQ-2'}}]},
    {'id': 'none', 'question': 'helicopter rotor torque'},
]


def answer(chunk_id, text, provided=None):
    return {'abstained': False, 'answer': text,
            'citations': [{'chunk_id': chunk_id, 'doc_id': 'pumps.xlsx', 'page': 1}],
            'sources': [{'chunk_id': c} for c in (provided or [chunk_id])]}


@pytest.fixture
def chunk(evidence_db):
    return evidence_eval.search(evidence_db, 'material SS316')[0]['chunk_id']


def test_search_returns_hash_location_system_revision_and_units(evidence_db, inputs):
    hit = evidence_eval.search(evidence_db, 'material SS316')[0]
    ev = hit['evidence']
    assert (hit['doc_id'], hit['page']) == ('pumps.xlsx', 1)
    assert ev['location'] == {'sheet': 'Pumps', 'row': 2, 'header_row': 1}
    assert (ev['system'], ev['revision']) == ('Pump skid', 'A')
    assert ev['source_sha256'] == hashlib.sha256((inputs / 'pumps.xlsx').read_bytes()).hexdigest()
    assert any(f['unit'] == 'bar' for f in ev['details']['fields'])
    assert ev['support_verified'] is False


def test_retrieval_is_scored_separately_per_case(evidence_db):
    report = evidence_eval.evaluate(evidence_db, CASES, top_k=3)
    by_id = {c['id']: c for c in report['cases']}
    assert by_id['material']['retrieval'] == {'recall': 1.0, 'hit': 1.0, 'mrr': 1.0}
    assert by_id['valve']['retrieval']['mrr'] == 1.0
    assert by_id['missed']['retrieval'] == {'recall': 0.0, 'hit': 0.0, 'mrr': 0.0}
    assert by_id['none']['retrieval'] is None and by_id['none']['answerable'] is False
    s = report['summary']
    assert (s['answerable'], s['unanswerable']) == (3, 1)
    assert s['retrieval']['hit'] == pytest.approx(2 / 3)
    assert s['answers']['supplied'] == 0


def test_gold_matching_nothing_and_duplicate_ids_are_errors(evidence_db):
    bad = {'id': 'x', 'question': 'q',
           'gold_evidence': [{'doc_id': 'pumps.xlsx', 'location': {'sheet': 'Pumps', 'row': 99}}]}
    with pytest.raises(ValueError, match='matches no indexed evidence'):
        evidence_eval.evaluate(evidence_db, [bad])
    with pytest.raises(ValueError, match='duplicate'):
        evidence_eval.evaluate(evidence_db, [CASES[3], CASES[3]])


def test_citation_validity_facts_and_retrieval_are_independent(evidence_db, inputs, tmp_path, chunk):
    def run(ans, **kw):
        return evidence_eval.evaluate(evidence_db, [CASES[0]], 3, {'material': ans}, **kw)['cases'][0]
    good = run(answer(chunk, 'P-101 runs at 3.2 bar.'), source_dir=inputs)
    assert good['answer']['citations_valid'] and good['answer']['facts_ok']
    assert good['answer']['citations'][0]['source_hash_ok'] is True
    wrong = run(answer(chunk, 'P-101 runs at 3.25 bar.'))
    assert wrong['answer']['citations_valid'] is True and wrong['answer']['facts_ok'] is False
    assert wrong['retrieval']['hit'] == 1.0
    fabricated = run(answer('pumps.xlsx#e999', 'P-101 runs at 3.2 bar.'))
    assert fabricated['answer']['citations'][0]['exists'] is False
    assert fabricated['answer']['citations_valid'] is False and fabricated['answer']['facts_ok'] is True
    unprovided = run(answer(chunk, 'P-101 runs at 3.2 bar.', provided=['reqs.csv#e0']))
    assert unprovided['answer']['citations_valid'] is False
    uncited = run({'abstained': False, 'answer': '3.2 bar', 'citations': [], 'sources': []})
    assert uncited['answer']['citations_valid'] is False
    tampered = tmp_path / 'tampered'
    tampered.mkdir()
    (tampered / 'pumps.xlsx').write_bytes((inputs / 'pumps.xlsx').read_bytes() + b'x')
    bad = run(answer(chunk, 'P-101 runs at 3.2 bar.'), source_dir=tampered)
    assert bad['answer']['citations'][0]['source_hash_ok'] is False
    assert bad['answer']['citations_valid'] is False


def test_abstention_and_failed_generation_are_not_conflated(evidence_db):
    def run(ans):
        return evidence_eval.evaluate(evidence_db, [CASES[3]], 3, {'none': ans})
    ok = run({'abstained': True, 'answer': 'No evidence.', 'citations': []})
    assert ok['cases'][0]['answer']['decision_correct'] is True
    false_answer = run({'abstained': False, 'answer': 'x', 'citations': []})
    assert false_answer['cases'][0]['answer']['decision_correct'] is False
    failed = run({'abstained': None, 'execution_status': 'provider_error'})
    assert failed['cases'][0]['answer']['scored'] is False
    assert failed['summary']['answers']['scored'] == 0
    assert failed['summary']['answers']['unscored_failures'] == 1


def test_entailment_comes_only_from_human_reviews(evidence_db, chunk):
    answers = {'material': answer(chunk, 'P-101 runs at 3.2 bar.')}
    pending = evidence_eval.evaluate(evidence_db, [CASES[0]], 3, answers)
    assert pending['cases'][0]['human_entailment'] == {'status': 'pending'}
    assert pending['summary']['human_entailment'] == {'required': 1, 'reviewed': 0, 'pending': 1,
                                                      'verdicts': {}}
    review = {'material': {'verdict': 'supported', 'reviewer': 'reviewer-1'}}
    done = evidence_eval.evaluate(evidence_db, [CASES[0]], 3, answers, review)
    assert done['summary']['human_entailment']['verdicts'] == {'supported': 1}
    for key in ('citations', 'facts', 'retrieval'):  # a review changes no automated metric
        assert done['summary'][key] == pending['summary'][key]
    with pytest.raises(ValueError, match='verdict'):
        evidence_eval.evaluate(evidence_db, [CASES[0]], 3, answers,
                               {'material': {'verdict': 'yes', 'reviewer': 'r'}})
    with pytest.raises(ValueError, match='not answered'):
        evidence_eval.evaluate(evidence_db, [CASES[0]], 3, None, review)


def test_cli_ingest_and_evidence_eval(tmp_path, inputs, capsys):
    db, cases, out = tmp_path / 'cli' / 'e.sqlite', tmp_path / 'cases.json', tmp_path / 'r' / 'eval.json'
    cases.write_text(json.dumps(CASES[:2]), encoding='utf-8')
    ingest = ['ingest', str(inputs / 'pumps.xlsx'), str(inputs / 'valves.md'), '--db', str(db),
              '--system', 'Skid', '--revision', 'A']
    assert main(ingest) == 0
    assert main(ingest) == 1 and 'exists' in capsys.readouterr().err
    assert main(['ingest', str(inputs / 'spec.reqif'), '--db', str(db), '--overwrite',
                 '--system', 'S', '--revision', 'A']) == 1
    assert 'unsupported' in capsys.readouterr().err
    evaluate = ['evidence-eval', '--cases', str(cases), '--db', str(db), '--output', str(out),
                '--top-k', '3']
    assert main(evaluate) == 0
    assert json.loads(out.read_text(encoding='utf-8'))['summary']['retrieval']['hit'] == 1.0
    assert main(evaluate) == 1 and 'exists' in capsys.readouterr().err
    assert main(evaluate + ['--overwrite']) == 0


def test_saved_answer_provenance_is_not_inferred(evidence_db, chunk):
    ans = answer(chunk, '3.2 bar')
    ans['sources'] = []
    result = evidence_eval.evaluate(evidence_db, [CASES[0]], answers={'material': ans})
    assert result['summary']['citations']['all_valid'] == 0
    ans = answer(chunk, '3.2 bar')
    ans['citations'][0]['source_sha256'] = 'stale-source'
    result = evidence_eval.evaluate(evidence_db, [CASES[0]], answers={'material': ans})
    assert result['summary']['citations']['all_valid'] == 0
    ans['question'] = 'Different question'
    with pytest.raises(ValueError, match='question mismatch'):
        evidence_eval.evaluate(evidence_db, [CASES[0]], answers={'material': ans})
