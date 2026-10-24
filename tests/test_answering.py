import json
import sqlite3
import tempfile
import unittest
import urllib.error
from contextlib import closing
from pathlib import Path
from unittest import mock

from evidence_lab import answering

URLOPEN = 'evidence_lab.answering.urllib.request.urlopen'
ROWS = [
    ('a.pdf', 1, 'a.pdf#p1-c0', 'The warranty period is twelve months from delivery.'),
    ('a.pdf', 2, 'a.pdf#p2-c0', 'Shipping uses ground freight only.'),
    ('b.pdf', 1, 'b.pdf#p1-c0', 'SECRETB warranty terms differ for this other document.'),
]


def reply(content, finish_reason='stop'):
    if not isinstance(content, str):
        content = json.dumps(content)
    body = {'model': answering.DEFAULT_MODEL, 'usage': {'total_tokens': 138},
            'choices': [{'finish_reason': finish_reason,
                         'message': {'role': 'assistant', 'content': content}}]}
    resp = mock.MagicMock()
    resp.__enter__.return_value.read.return_value = json.dumps(body).encode('utf-8')
    return resp


class AnsweringTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        key_patch = mock.patch.dict('os.environ', {'OPENROUTER_API_KEY': 'disposable-test-key'})
        key_patch.start()
        self.addCleanup(key_patch.stop)
        self.db = Path(tmp.name) / 'index.sqlite'
        with closing(sqlite3.connect(self.db)) as con:
            con.execute('CREATE VIRTUAL TABLE chunks USING fts5('
                        'doc_id UNINDEXED, page UNINDEXED, chunk_id UNINDEXED, text)')
            con.executemany('INSERT INTO chunks VALUES (?, ?, ?, ?)', ROWS)
            con.commit()

    def ask(self, question='How long is the warranty period?', **kw):
        return answering.answer(self.db, question, doc_id='a.pdf', allow_paid=True, **kw)

    def test_no_hits_skips_generation(self):
        with mock.patch(URLOPEN) as urlopen:
            result = self.ask('zzzunmatched')
        urlopen.assert_not_called()
        self.assertEqual(result['status'], 'no_evidence')
        self.assertTrue(result['abstained'])
        self.assertEqual(result['sources'], [])

    def test_request_contains_only_retrieved_evidence(self):
        content = {'answer': 'Twelve months.', 'citations': ['S1'], 'abstained': False}
        with mock.patch(URLOPEN, return_value=reply(content)) as urlopen:
            result = self.ask()
        req = urlopen.call_args.args[0]
        self.assertEqual(req.full_url, answering.OPENROUTER_URL)
        self.assertEqual(urlopen.call_args.kwargs['timeout'], answering.TIMEOUT_S)
        self.assertEqual(req.get_header('Authorization'), 'Bearer disposable-test-key')
        payload = json.loads(req.data)
        self.assertEqual(payload['model'], answering.DEFAULT_MODEL)
        self.assertIs(payload['stream'], False)
        self.assertEqual(payload['response_format'], {'type': 'json_object'})
        self.assertEqual(payload['temperature'], 0)
        self.assertEqual(payload['max_tokens'], 384)
        prompt = ' '.join(m['content'] for m in payload['messages'])
        self.assertIn('twelve months', prompt)
        self.assertIn('a.pdf#p1-c0', prompt)
        self.assertIn('untrusted', prompt)
        self.assertNotIn('SECRETB', prompt)  # other document, not retrieved
        self.assertNotIn('evidence_pages', prompt)
        self.assertNotIn('Not answerable', prompt)
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['citations'][0]['page'], 1)
        self.assertEqual(result['model_reported'], answering.DEFAULT_MODEL)
        self.assertEqual(result['usage']['total_tokens'], 138)
        self.assertFalse(result['citation_validation']['entailment_checked'])
        self.assertEqual(result['context_chars'], sum(len(s['text']) for s in result['sources']))
        self.assertNotIn('disposable-test-key', json.dumps(result))

    def test_invalid_citations_rejected(self):
        cases = {
            'unknown id': {'answer': 'x', 'citations': ['S9'], 'abstained': False},
            'no citation': {'answer': 'x', 'citations': [], 'abstained': False},
            'empty answer': {'answer': ' ', 'citations': ['S1'], 'abstained': False},
            'not json': 'sorry, I cannot',
        }
        for name, content in cases.items():
            with self.subTest(name), mock.patch(URLOPEN, return_value=reply(content)):
                result = self.ask()
            self.assertEqual(result['execution_status'], 'invalid_response')
            self.assertEqual(result['status'], 'invalid_response')
            self.assertIsNone(result['answer'])
            self.assertIsNone(result['abstained'])
            self.assertTrue(result['error'])
            self.assertIsNotNone(result['raw_response'])

    def test_genuine_refusal_reported(self):
        content = {'answer': 'The evidence does not say.', 'citations': [], 'abstained': True}
        with mock.patch(URLOPEN, return_value=reply(content)):
            result = self.ask()
        self.assertEqual(result['status'], 'abstained')
        self.assertEqual(result['execution_status'], 'ok')
        self.assertTrue(result['abstained'])
        self.assertIsNone(result['error'])

    def test_incomplete_generation_is_not_accepted_as_an_answer(self):
        content = {'answer': 'Twelve months.', 'citations': ['S1'], 'abstained': False}
        for reason in ('length', None):
            with self.subTest(reason), mock.patch(URLOPEN,
                                                  return_value=reply(content, reason)):
                result = self.ask()
            self.assertEqual(result['status'], 'invalid_response')
            self.assertIsNone(result['answer'])

    def test_provider_error_is_not_abstention(self):
        for exc in (urllib.error.URLError('connection refused'), TimeoutError('timed out')):
            with self.subTest(type(exc).__name__), mock.patch(URLOPEN, side_effect=exc):
                result = self.ask()
            self.assertEqual(result['execution_status'], 'provider_error')
            self.assertEqual(result['status'], 'provider_error')
            self.assertIsNot(result['abstained'], True)
            self.assertIsNone(result['answer'])
            self.assertTrue(result['error'])

    def test_context_budget_limits_sent_snippets(self):
        long_rows = [('a.pdf', n, f'a.pdf#p{n}-c0', 'warranty ' + 'x' * 15000)
                     for n in range(1, 4)]
        with closing(sqlite3.connect(self.db)) as con:
            con.executemany('INSERT INTO chunks VALUES (?, ?, ?, ?)', long_rows)
            con.commit()
        content = {'answer': 'n/a', 'citations': [], 'abstained': True}
        with mock.patch(URLOPEN, return_value=reply(content)):
            result = self.ask(top_k=10)
        self.assertLessEqual(result['context_chars'], answering.CONTEXT_CHARS)
        self.assertTrue(any(s['truncated'] for s in result['sources']))
        sent = result['messages'][1]['content']
        for s in result['sources']:
            self.assertIn(s['chunk_id'], sent)


if __name__ == '__main__':
    unittest.main()
