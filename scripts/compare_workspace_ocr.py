"""Compare existing local/API OCR indexes on known synthetic labels, without API calls."""
import json
import sqlite3
from pathlib import Path
from evidence_lab.corpus import write_json_atomic

labels = ['SYNTHETIC DRAWING', 'VALVE V-101', 'PORT DN20']
report = {'note': 'One simple synthetic drawing; not a general OCR quality benchmark.', 'labels': labels}
for name, folder in [('local', 'workspace'), ('openrouter', 'workspace-api')]:
    with sqlite3.connect(f'data/{folder}/evidence.sqlite') as con:
        lines = [r[0] for r in con.execute("SELECT text FROM chunks WHERE doc_id='drawing.png'")]
    text = '\n'.join(lines)
    report[name] = {'lines': lines, 'exact_label_matches': {label: label in text for label in labels}}
for path in Path('data/workspace-api/openrouter-ocr/responses').glob('*.json'):
    cached = json.loads(path.read_text(encoding='utf-8'))
    report['openrouter']['usage'] = cached['usage']
write_json_atomic(Path('reports/workspace-ocr-comparison.json'), report)
print(json.dumps(report, indent=2))
answers = json.loads(Path('data/workspace-api-answers.json').read_text(encoding='utf-8'))
print('Unsupported case:', answers['unsupported']['status'], answers['unsupported'].get('error'))
