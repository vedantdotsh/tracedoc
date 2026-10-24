"""Execute frozen table pilot. Docling environment required only with --extract."""
import argparse, hashlib, importlib.metadata, json, platform, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evidence_lab.tables import lookup, normalize_docling, text_lookup, verify_reviewed_answer

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def extract(pdf,pages,output,ocr=False):
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    options=PdfPipelineOptions(do_ocr=ocr,do_table_structure=True)
    options.table_structure_options.mode=TableFormerMode.ACCURATE
    options.accelerator_options=AcceleratorOptions(num_threads=4,device=AcceleratorDevice.CPU)
    converter=DocumentConverter(format_options={InputFormat.PDF:PdfFormatOption(pipeline_options=options)})
    started=time.perf_counter()
    result=converter.convert(pdf,page_range=(min(pages),max(pages)))
    document=result.document.export_to_dict()
    dump(output/'docling.json',document)
    tables=normalize_docling(document,pdf.name)
    for table in tables:
        for fact in table['facts']:
            fact['provenance']['source_sha256']=hashlib.sha256(pdf.read_bytes()).hexdigest()
    dump(output/'tables.json',tables)
    versions={k:importlib.metadata.version(k) for k in ['docling','docling-core','docling-ibm-models','docling-parse','torch','transformers']}
    dump(output/'runtime.json',dict(seconds=time.perf_counter()-started,python=sys.version,platform=platform.platform(),versions=versions,status=str(result.status),pages=pages,source_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),tableformer_mode='accurate',ocr=ocr,device='cpu',threads=4))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract',action='store_true')
    parser.add_argument('--ocr',action='store_true',help='Enable Docling OCR for scanned PDFs (additional local OCR models may download)')
    parser.add_argument('--pdf',type=Path)
    parser.add_argument('--pages',type=int,nargs='+',default=[31,32])
    parser.add_argument('--output',type=Path,default=ROOT/'data/tables-v1/extraction')
    args=parser.parse_args()
    if args.pdf:
        if not args.extract: parser.error('--pdf requires --extract; custom extraction does not run frozen evaluation')
        if max(args.pages)-min(args.pages)>9: parser.error('At most ten contiguous pages per bounded extraction')
        extract(args.pdf,args.pages,args.output,ocr=args.ocr)
        return
    folder=ROOT/'data/tables-v1'
    raw=(folder/'spec.json').read_bytes()
    sha=hashlib.sha256(raw).hexdigest()
    if sha!=(folder/'spec.sha256').read_text().strip(): raise ValueError('Frozen specification hash mismatch')
    spec=json.loads(raw)
    pdf=ROOT/'data/documents'/spec['source_doc']
    if hashlib.sha256(pdf.read_bytes()).hexdigest()!=spec['source_sha256']: raise ValueError('Source hash mismatch')
    if args.extract: extract(pdf,spec['pages'],args.output)
    import fitz
    with fitz.open(pdf) as doc:
        pages=[dict(page=n,text=doc[n-1].get_text()) for n in spec['pages']]
    dump(folder/'native-text.json',pages)
    tables=json.loads((args.output/'tables.json').read_text(encoding='utf-8'))
    predictions=[]
    for question in spec['questions']:
        query={k:question[k] for k in ['row_id','column','conditions','unit']}
        for name,predict in [('text_only',lambda:text_lookup(pages,**query)),('structured',lambda:lookup(tables,**query))]:
            answer=predict()
            correct=answer['status']==question['expected_status'] and (answer['status']=='abstain' or answer['value']==question['expected_value'])
            predictions.append(dict(id=question['id'],method=name,query=query,prediction=answer,correct=correct,expected_status=question['expected_status'],expected_value=question['expected_value']))
    metrics={}
    for name in ['text_only','structured']:
        rows=[p for p in predictions if p['method']==name]
        answered=[p for p in rows if p['prediction']['status']=='answer']
        metrics[name]=dict(n=len(rows),correct=sum(p['correct'] for p in rows),answered=len(answered),incorrect_answers=sum(not p['correct'] for p in answered),accuracy=sum(p['correct'] for p in rows)/len(rows),coverage=len(answered)/len(rows))
    report=dict(spec_sha256=sha,metrics=metrics,limitations=['Purposeful two-page one-manual exact lookup pilot, not a held-out generalization claim.','Both methods use hand-declared narrow type parsers fixed before extraction.','Text baseline is native line-offset lookup, not an LLM; no claim of superiority over strong text models.','Source locator is provenance, not verification; all extracted facts remain source_verified=false.','Gold is read only for scoring; neither predictor receives reference values.'])
    reviews=json.loads((folder/'source-review.json').read_text(encoding='utf-8'))['reviews']
    verification=[dict(id=p['id'],result=verify_reviewed_answer(p['prediction'],reviews)) for p in predictions if p['method']=='structured']
    dump(ROOT/'reports/tables-v1/source-verification.json',verification)
    dump(ROOT/'reports/tables-v1/predictions.json',predictions)
    dump(ROOT/'reports/tables-v1/metrics.json',report)
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
