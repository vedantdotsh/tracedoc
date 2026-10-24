"""Loopback evidence browser; generation requires an explicit bounded POST."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from .retrieval import Retriever, METHODS
from . import answering, evaluate_answers

HTML = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>TraceDoc</title>
<style>body{font:16px system-ui;margin:0;background:#f2f5fa;color:#182b40}header{background:#142f46;color:white;padding:28px 5%}main{max-width:1250px;margin:auto;padding:28px}input,select,button{font:inherit;padding:10px;border:1px solid #9aabba;border-radius:6px}input{width:65%}select{max-width:100%;min-width:0}h2,summary{overflow-wrap:anywhere}button{background:#146a75;color:white;cursor:pointer}.controls{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}article{background:white;padding:20px;border-radius:10px;margin-bottom:16px}pre{white-space:pre-wrap;font:15px/1.5 system-ui}img{max-width:100%}.muted{color:#52687b}#status{min-height:28px}@media(max-width:800px){.grid{grid-template-columns:1fr}input{width:90%}}</style>
<header><h1>TraceDoc</h1><p>Find evidence. Inspect the original page. Check the answer.</p></header>
<main><p>135 PDFs · 107 with extractable text chunks · physical page citations · local retrieval. Saved evaluation is document-scoped; selecting All documents searches the entire corpus.</p>
<form id="form"><input id="query" required maxlength="2000" placeholder="Ask a question or search an exact identifier" value="purchasing power 2014"><button>Search evidence</button><div class="controls"><select id="doc"><option value="">All documents</option></select><select id="method"><option>bm25</option><option>dense</option><option selected>hybrid</option><option>reranked</option></select></div></form>
<label><input id="evidence-gate" type="checkbox" style="width:auto"> Use optional evidence checks</label> <button id="ask" type="button">Generate cited answer · paid, max $0.01</button><p class="muted">One OpenRouter request per click. Session ceiling $0.05. No automatic retries.</p><article id="live" hidden></article><p id="status" role="status"></p><div class="grid"><section id="hits"></section><section><article><h2>Original source page</h2><p>Select a document above, then a physical page:</p><input id="page-number" type="number" min="1" value="1" style="width:70px"><button id="direct-page" type="button">View page</button><p class="muted">Choose a citation to inspect tables, diagrams and surrounding context. Page images are for human inspection; the model does not interpret them.</p><img id="page" alt="Select a citation to view its source page"></article><article><h2>Saved answer evaluation</h2><p>Browsing and saved results make no API calls. Citation existence and numeric string matches do not prove that an answer follows from the evidence.</p><div id="answers"></div></article></section></div></main>
<script>const $=id=>document.getElementById(id);function el(tag,text){const e=document.createElement(tag);e.textContent=text;return e;}fetch('/api/docs').then(r=>r.json()).then(ds=>ds.forEach(d=>{const o=el('option',d);o.value=d;$('doc').append(o)}));
fetch('/api/config').then(r=>r.json()).then(c=>{$('method').value=c.method});
fetch('/api/answers').then(r=>r.json()).then(x=>{const box=$('answers');box.append(el('p',x.note||x.status||'Saved diagnostic answers'));(x.cases||[]).forEach(c=>{const d=el('details','');d.append(el('summary',c.method+' · '+c.status+' · '+c.question),el('p',c.answer||'No answer'));const technical=el('details','');technical.append(el('summary','Scoring diagnostics'),el('pre',JSON.stringify(c.scores,null,2)));d.append(technical);box.append(d)});});
$('ask').onclick=async()=>{const b=$('ask');b.disabled=true;$('live').hidden=false;$('live').textContent='Generating with the retrieved evidence…';try{const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({q:$('query').value,doc:$('doc').value,method:$('method').value,allow_paid:true,evidence_gate:$('evidence-gate').checked})});const x=await r.json();if(!r.ok)throw Error(x.error);$('live').replaceChildren(el('h2','Generated answer · '+x.status),el('p',x.answer||x.error||'No answer'),el('p','Reported cost: '+(x.status==='gate_abstained'?'0 (no generation call)':(x.usage.cost??'unreported'))+' USD'));(x.citations||[]).forEach(c=>{const a=el('button',c.source_id+' · '+c.doc_id+' · p'+c.page);a.onclick=()=>{$('page').src='/api/page?'+new URLSearchParams({doc:c.doc_id,page:c.page})};$('live').append(a)});$('live').append(el('p',x.status==='gate_abstained'?'Rule-based abstention: '+x.evidence_gate.reasons.join(', ')+'. No generation request was made.':'Citation IDs are checked only for existence. Entailment has not been verified.')); }catch(e){$('live').textContent=e.message}finally{b.disabled=false}};
$('direct-page').onclick=()=>{if(!$('doc').value){$('status').textContent='Select a document first';return;}$('page').src='/api/page?'+new URLSearchParams({doc:$('doc').value,page:$('page-number').value});};
$('form').onsubmit=async e=>{e.preventDefault();$('status').textContent='Searching… first model load may take a moment.';$('hits').replaceChildren();try{const p=new URLSearchParams({q:$('query').value,doc:$('doc').value,method:$('method').value});const r=await fetch('/api/search?'+p);const x=await r.json();if(!r.ok)throw Error(x.error);$('status').textContent=x.length+' evidence chunks · '+($('doc').value?'document-scoped':'corpus-wide');x.forEach((h,i)=>{const a=el('article','');a.append(el('h2','S'+(i+1)+' · '+h.doc_id));const b=el('button','View physical page '+h.page);b.onclick=()=>{$('page').src='/api/page?'+new URLSearchParams({doc:h.doc_id,page:h.page});$('page').alt=h.doc_id+' physical page '+h.page;$('page').scrollIntoView({behavior:'smooth',block:'start'})};a.append(b,el('p','Score '+h.score.toFixed(5)+' · '+h.chunk_id),el('pre',h.text));$('hits').append(a)});}catch(err){$('status').textContent=err.message;}};</script></html>'''

def create_server(root=Path('.'), port=8765):
    root=Path(root).resolve()
    retriever=Retriever(root/'data/baseline.sqlite', root/'data/dense-v1')
    docs=sorted({r['doc_id'] for r in retriever.rows} | {p.name for p in (root/'data/documents').glob('*.pdf')})
    reserved_spend = 0.0
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            nonlocal reserved_spend
            try:
                size=int(self.headers.get('Content-Length','0'))
            except ValueError:
                return self.json({'error':'Invalid request size'},400)
            if not 0<size<=12000:
                return self.json({'error':'Invalid request size'},400)
            payload=self.rfile.read(size)
            host=f'127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host') != host or self.headers.get('Origin') != 'http://'+host:
                return self.json({'error':'Same-origin loopback request required'},403)
            if self.path != '/api/ask': return self.json({'error':'Not found'},404)
            try:
                body=json.loads(payload)
                if not isinstance(body,dict) or body.get('allow_paid') is not True: raise ValueError('Explicit paid opt-in required')
                query=body.get('q','');doc=body.get('doc') or None;method=body.get('method','hybrid')
                if not isinstance(query,str) or not query.strip() or len(query)>2000: raise ValueError('Invalid question')
                if doc is not None and doc not in docs: raise ValueError('Unknown document')
                if method not in METHODS: raise ValueError('Unknown retrieval method')
                if not isinstance(body.get('evidence_gate',False),bool): raise ValueError('Evidence gate must be a boolean')
                hits=retriever.search(query,doc,5,method)
                from .abstention import check
                sources=answering._build_sources(hits)
                blocked=body.get('evidence_gate',False) and bool(sources) and check(query,sources)['abstain']
                if blocked:
                    result=answering.answer_from_hits(query,hits,allow_paid=True,doc_id=doc,evidence_gate=True)
                    bound={'cost_upper_bound_usd':0.0,'generation_request':False}
                else:
                    quote=evaluate_answers.current_prices()
                    bound=evaluate_answers.request_bound(query,hits)
                    cost=bound['cost_upper_bound_usd']
                    if cost>0.01 or reserved_spend+cost>0.05: raise ValueError('Demo budget reached: $0.01 per request, $0.05 per server session')
                    if answering.OPTIONS['max_tokens'] != 384: raise ValueError('Generation options changed; budget invalid')
                    reserved_spend+=cost  # Never refund uncertain calls or retry them.
                    from dotenv import load_dotenv
                    load_dotenv(root/'.env',override=False)
                    result=answering.answer_from_hits(query,hits,allow_paid=True,doc_id=doc,provider_max_price=evaluate_answers.PRICE_CEILINGS.copy(),evidence_gate=body.get('evidence_gate',False))
                result['budget']=bound; result['session_reserved_usd']=reserved_spend
                from .corpus import write_json_atomic
                import uuid
                write_json_atomic(root/'data/demo-answers'/f'{uuid.uuid4().hex}.json',result)
                result.pop('messages',None);result.pop('raw_response',None)
                return self.json(result)
            except (ValueError,OSError,RuntimeError,KeyError,TypeError) as exc:
                return self.json({'error':str(exc)},400)
        def do_GET(self):
            if self.headers.get('Host') != f'127.0.0.1:{self.server.server_port}':
                return self.json({'error':'Loopback Host required'},403)
            parsed=urlsplit(self.path); args=parse_qs(parsed.query)
            def arg(k,default=''): return args.get(k,[default])[0]
            try:
                if parsed.path == '/':
                    return self.respond(HTML.encode(),'text/html; charset=utf-8')
                if parsed.path == '/api/docs': return self.json(docs)
                if parsed.path == '/api/config':
                    selection=root/'reports/retrieval-v1/selection.json'
                    return self.json(json.loads(selection.read_text('utf-8')) if selection.exists() else {'method':'bm25'})
                if parsed.path == '/api/answers':
                    path=root/'reports/answer-evaluation-v1/report.json'
                    if not path.exists(): return self.json({'status':'No saved answer evaluation available yet.'})
                    report=json.loads(path.read_text('utf-8'))
                    return self.json({'summary':report.get('summary',{}),'note':'Tiny diagnostic sample; lexical overlap is not semantic correctness.',
                        'cases':[{'question':c.get('result',{}).get('question'), 'method':c.get('method'),'answer':c.get('result',{}).get('answer'),'status':c.get('result',{}).get('status'),'scores':c.get('scores')} for c in report.get('cases',[])]})
                if parsed.path in ('/api/search','/api/page'):
                    doc=arg('doc') or None
                    if doc is not None and doc not in docs: raise ValueError('Unknown document')
                    if parsed.path == '/api/search':
                        query=arg('q'); method=arg('method','bm25')
                        if not query.strip() or len(query)>2000: raise ValueError('Enter a question of 1–2000 characters')
                        return self.json(retriever.search(query,doc,5,method))
                    if not doc: raise ValueError('Document required')
                    import pymupdf
                    page=int(arg('page','0'))
                    with pymupdf.open(root/'data/documents'/doc) as pdf:
                        if not 1<=page<=len(pdf): raise ValueError('Page outside document')
                        return self.respond(pdf[page-1].get_pixmap(matrix=pymupdf.Matrix(1.2,1.2)).tobytes('png'),'image/png')
                self.json({'error':'Not found'},404)
            except (ValueError,OSError,RuntimeError) as exc:
                self.json({'error':str(exc)},400)
        def json(self,data,status=200): return self.respond(json.dumps(data,ensure_ascii=False).encode(),'application/json; charset=utf-8',status)
        def respond(self,data,content_type,status=200):
            self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(data)));self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
    return HTTPServer(('127.0.0.1',port),Handler)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--port',type=int,default=8765);a=p.parse_args()
    server=create_server(a.root,a.port);print(f'Evidence demo: http://127.0.0.1:{server.server_port}',flush=True);server.serve_forever()
