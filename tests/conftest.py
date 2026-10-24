import zipfile

import pymupdf
import pytest

MAIN = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
LINES = [{'text': 'DN50 PN16', 'confidence': 0.9, 'box': [[1, 1], [30, 1], [30, 10], [1, 10]]}]


def make_xlsx(path, sheets):
    shared, tabs, rels, files = [], [], [], {}
    for n, (name, rows) in enumerate(sheets.items(), 1):
        body = ''
        for r, row in enumerate(rows, 1):
            cells = ''
            for c, value in enumerate(row):
                ref = f'{"ABCDEF"[c]}{r}'
                if isinstance(value, str):
                    shared.append(value)
                    cells += f'<c r="{ref}" t="s"><v>{len(shared) - 1}</v></c>'
                elif value is not None:
                    cells += f'<c r="{ref}"><v>{value}</v></c>'
            body += f'<row r="{r}">{cells}</row>'
        files[f'xl/worksheets/sheet{n}.xml'] = f'<worksheet xmlns="{MAIN}"><sheetData>{body}</sheetData></worksheet>'
        tabs.append(f'<sheet name="{name}" sheetId="{n}" r:id="rId{n}"/>')
        rels.append(f'<Relationship Id="rId{n}" Target="worksheets/sheet{n}.xml"/>')
    files['xl/workbook.xml'] = f'<workbook xmlns="{MAIN}" xmlns:r="{REL}"><sheets>{"".join(tabs)}</sheets></workbook>'
    files['xl/_rels/workbook.xml.rels'] = ('<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                                           + ''.join(rels) + '</Relationships>')
    files['xl/sharedStrings.xml'] = f'<sst xmlns="{MAIN}">' + ''.join(f'<si><t>{s}</t></si>' for s in shared) + '</sst>'
    with zipfile.ZipFile(path, 'w') as z:
        for name, data in files.items():
            z.writestr(name, data)


@pytest.fixture
def metadata():
    return {'*': {'system': 'Pump skid', 'revision': 'A'}, 'reqs.csv': {'revision': 'C'}}


@pytest.fixture
def fake_ocr():
    return lambda data: LINES


@pytest.fixture
def inputs(tmp_path):
    folder = tmp_path / 'inputs'
    folder.mkdir()
    make_xlsx(folder / 'pumps.xlsx', {'Pumps': [['Tag', 'Pressure [bar]', 'Material'],
                                                ['P-101', 3.2, 'SS316'], ['P-102', 7.5, 'Bronze']],
                                      'Spare': []})
    (folder / 'reqs.csv').write_text(
        'id,text,unit,revision\nREQ-1,Pump P-101 shall deliver 3.2 bar,bar,B\n'
        'REQ-2,Housing shall resist 40 °C ambient,,\n', encoding='utf-8')
    (folder / 'valves.md').write_text(
        '# Valves\n- SYS-12: The valve shall close within 500 ms\n  after trip signal.\nplain prose\n',
        encoding='utf-8')
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), 'Flange OD 120 mm')
        doc.save(folder / 'drawing.pdf')
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 20), False)
    pix.clear_with(255)
    pix.save(folder / 'scan.png')
    return folder


@pytest.fixture
def evidence_db(tmp_path, inputs, metadata, fake_ocr):
    from evidence_lab import sources
    db = tmp_path / 'out' / 'evidence.sqlite'
    sources.build_index(sources.expand([inputs]), db, metadata, fake_ocr)
    return db
