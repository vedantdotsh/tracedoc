"""Create clearly synthetic engineering documents, never claimed as fuel-cell specifications."""
from pathlib import Path
import json
import pymupdf
from openpyxl import Workbook

root = Path('data/workspace-sample')
root.mkdir(parents=True, exist_ok=False)
book = Workbook()
sheet = book.active
sheet.title = 'Limits'
sheet.append(['System', 'Revision', 'Coolant inlet maximum [C]', 'Pressure [bar]'])
sheet.append(['DEMO-A', 'R1', 65, 2.5])
sheet.append(['DEMO-B', 'R1', 70, 3.0])
book.save(root / 'limits.xlsx')
(root / 'requirements.csv').write_text(
    'id,text,system,revision,status\nREQ-101,DEMO-A controller shall stop on leak alarm,DEMO-A,R1,draft\n'
    'REQ-102,DEMO-B controller shall record the alarm timestamp,DEMO-B,R1,draft\n', encoding='utf-8')
(root / 'requirements.md').write_text(
    '# Synthetic training requirements\n- SYS-201: DEMO-A valve shall close within 500 ms\n'
    '  after the trip signal.\n', encoding='utf-8')
with pymupdf.open() as doc:
    p = doc.new_page()
    p.insert_text((50, 50), 'SYNTHETIC TRAINING DATA - NOT ENGINEERING GUIDANCE', fontsize=14)
    p.insert_text((50, 100), 'DEMO-A service interval is 120 hours. Revision R1.', fontsize=14)
    p.insert_text((50, 130), 'DEMO-B service interval is 180 hours. Revision R1.', fontsize=14)
    doc.save(root / 'manual.pdf')
with pymupdf.open() as doc:
    p = doc.new_page(width=700, height=400)
    p.insert_text((30, 40), 'SYNTHETIC DRAWING - DEMO-A R1', fontsize=20)
    p.draw_rect(pymupdf.Rect(120, 120, 420, 260), width=2)
    p.insert_text((150, 170), 'VALVE V-101', fontsize=24)
    p.insert_text((150, 220), 'PORT DN20', fontsize=24)
    doc.save(root / 'drawing.pdf')
    p.get_pixmap(dpi=150).save(root / 'drawing.png')
cases = [
    {'id': 'pdf', 'question': 'DEMO-A service interval hours', 'gold_evidence':
        [{'doc_id': 'manual.pdf', 'location': {'page': 1}}], 'expected_facts': [{'value': '120', 'unit': 'hours'}]},
    {'id': 'excel', 'question': 'DEMO-A coolant inlet maximum', 'gold_evidence':
        [{'doc_id': 'limits.xlsx', 'location': {'sheet': 'Limits', 'row': 2}}], 'expected_facts': [{'value': '65'}]},
    {'id': 'requirement', 'question': 'REQ-101 leak alarm', 'gold_evidence':
        [{'doc_id': 'requirements.csv', 'location': {'requirement_id': 'REQ-101'}}]},
    {'id': 'drawing', 'question': 'VALVE V-101 PORT DN20', 'gold_evidence':
        [{'doc_id': 'drawing.pdf', 'location': {'page': 1}}]},
    {'id': 'unsupported', 'question': 'What catalyst mass is specified?'},
]
Path('data/workspace-cases.json').write_text(json.dumps(cases, indent=2), encoding='utf-8')
print(root.resolve())
