import json

from evidence_lab import answering


def test_excel_locator_survives_prompt_and_citation(monkeypatch):
    location = {'kind': 'excel', 'sheet': 'Limits', 'cell_range': 'A2:C2'}
    hits = [{'doc_id': 'limits.xlsx', 'page': None, 'chunk_id': 'limits-2',
             'text': 'Coolant temperature: 65 C', 'location': location}]
    def respond(messages, model):
        assert 'cell_range' in messages[1]['content']
        return json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content':
            json.dumps({'answer': '65 C', 'citations': ['S1'], 'abstained': False})}}]}).encode()
    monkeypatch.setattr(answering, '_post', respond)
    result = answering.answer_from_hits('Coolant temperature?', hits, allow_paid=True)
    assert result['citations'][0]['location'] == location
