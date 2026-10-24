"""Loopback chat for mixed document evidence; no uploads or implicit generation."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from . import answering, evaluate_answers

HTML = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TraceDoc</title>
<style>
:root{--bg:#f7f4ee;--surface:#fffdfa;--ink:#1d2327;--muted:#5f625d;--line:#e3dccf;--field:#cfc6b6;--teal:#0e6b68;--teal-dk:#0a514f;--teal-bg:#e7f1ef;--warn:#7a4d00;--warn-bg:#f8eed8;--err:#9b2c2c}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif}
.top{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 16px;padding:14px 28px;border-bottom:1px solid var(--line)}
h1{margin:0;font:600 1.15rem/1.2 ui-serif,Georgia,"Times New Roman",serif;letter-spacing:.01em}
.lib{margin:0;color:var(--muted);font-size:.9rem}
.layout{max-width:1280px;margin:0 auto;padding:28px;display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:32px;align-items:start}
h2{margin:0 0 10px;font:600 1rem/1.3 ui-serif,Georgia,"Times New Roman",serif}
label{display:block;margin:0 0 6px;font-size:.88rem;font-weight:600}
textarea,select{width:100%;font:inherit;color:inherit;background:#fff;border:1px solid var(--field);border-radius:8px}
textarea{min-height:5.5em;padding:12px 14px;resize:vertical}
select{padding:8px 10px}
.hint{margin:6px 0 0;color:var(--muted);font-size:.82rem}
.actions{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-top:14px}
.spacer{flex:1}
button{font:inherit;cursor:pointer;border-radius:8px;padding:9px 16px;border:1px solid transparent}
.primary{background:var(--teal);color:#fff}
.primary:hover:not(:disabled){background:var(--teal-dk)}
.secondary{background:var(--surface);color:var(--teal);border-color:var(--teal)}
.secondary:hover:not(:disabled){background:var(--teal-bg)}
.link{background:none;border:0;padding:4px 2px;color:var(--teal);text-decoration:underline;text-underline-offset:3px;font-size:.9rem}
.quiet{color:var(--muted)}
button:disabled{opacity:.5;cursor:not-allowed}
:focus-visible{outline:2px solid var(--teal);outline-offset:2px}
details.settings{margin-top:18px;border-top:1px solid var(--line);padding-top:10px}
details.settings summary{cursor:pointer;color:var(--muted);font-size:.9rem;width:max-content}
.fields{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0 10px}
.notes{margin:0;padding-left:1.1em;color:var(--muted);font-size:.85rem}
.notes li{margin:4px 0}
#status{min-height:1.55em;margin:18px 0 0;color:var(--muted);font-size:.9rem}
.error{color:var(--err)}
.empty{color:var(--muted)}
.turn{border-top:1px solid var(--line);padding:22px 0 8px}
.kind{margin:0;color:var(--teal);font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em}
.turn h3{margin:4px 0 12px;font-size:1.05rem;font-weight:600;overflow-wrap:anywhere}
.answer{margin:0 0 8px;padding:12px 16px;background:var(--surface);border-left:3px solid var(--teal);border-radius:0 6px 6px 0;white-space:pre-wrap;overflow-wrap:anywhere}
.meta{margin:4px 0 12px;color:var(--muted);font-size:.85rem}
.hits{list-style:none;margin:0;padding:0}
.hit{margin:10px 0;padding:12px 14px;background:var(--surface);border:1px solid var(--line);border-radius:8px}
.hit.selected{border-color:var(--teal);box-shadow:0 0 0 1px var(--teal)}
.hit-head{display:flex;flex-wrap:wrap;gap:4px 10px;color:var(--muted);font-size:.85rem}
.sid{color:var(--teal);font-weight:600}
.doc{color:var(--ink);font-weight:500;overflow-wrap:anywhere}
pre.excerpt{margin:8px 0;white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;font-size:.93rem;line-height:1.5;max-height:14em;overflow:auto}
pre.full{max-height:none}
.flag{display:inline-block;margin:4px 0;padding:2px 8px;background:var(--warn-bg);color:var(--warn);border-radius:4px;font-size:.8rem}
.source{position:sticky;top:20px;max-height:calc(100vh - 40px);overflow:auto;padding:20px 22px;background:var(--surface);border:1px solid var(--line);border-radius:10px}
.prov{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;margin:0 0 12px;font-size:.85rem}
.prov dt{color:var(--muted)}
.prov dd{margin:0;overflow-wrap:anywhere}
.label{margin:14px 0 0;font-size:.8rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
figure{margin:16px 0 0}
figure img{display:block;max-width:100%;height:auto;border:1px solid var(--line);border-radius:4px;background:#fff}
figcaption{margin-top:6px;color:var(--muted);font-size:.8rem}
@media (max-width:900px){.top{padding:12px 18px}.layout{grid-template-columns:1fr;padding:18px;gap:24px}.source{position:static;max-height:none}}
@media (max-width:520px){.fields{grid-template-columns:1fr}.actions button.primary,.actions button.secondary{flex:1 1 100%}}
</style>
</head>
<body>
<header class="top">
<h1>TraceDoc</h1>
<p class="lib" id="lib">Loading documents…</p>
</header>
<main class="layout">
<div>
<section aria-labelledby="ask-title">
<h2 id="ask-title">Ask the documents</h2>
<form id="composer" novalidate>
<label for="q">Question</label>
<textarea id="q" maxlength="2000" required aria-describedby="q-hint" placeholder="Which temperature limit is specified?"></textarea>
<p class="hint" id="q-hint">Ctrl + Enter finds evidence. Each question is searched on its own.</p>
<p class="hint" id="library-note"></p>
<div class="actions">
<button id="find" class="primary" type="submit">Find evidence</button>
<button id="ask" class="secondary" type="button" hidden disabled>Generate answer · paid, max $0.01</button>
<span class="spacer"></span>
<button id="clear" class="link quiet" type="button">Clear conversation</button>
</div>
<details class="settings">
<summary>Search settings</summary>
<div class="fields">
<div><label for="doc">Documents</label><select id="doc"><option value="">All documents</option></select></div>
<div><label for="method">Search method</label><select id="method"><option value="bm25" selected>Keyword</option><option value="dense" disabled>Semantic</option><option value="hybrid" disabled>Hybrid</option></select></div>
</div>
<ul class="notes">
<li id="cfg-note">Checking server configuration…</li>
<li>Keyword search runs locally at no cost. Semantic and hybrid search send your question to OpenRouter for a paid embedding.</li>
<li>Generate answer runs the same search, then sends the top excerpts to a language model. Each request is capped at $0.01 and the server stops at $0.05 per session. <span id="spent"></span></li>
<li>Each question is standalone: earlier turns stay on this page but are never used as evidence or context.</li>
<li>Citation IDs in answers are checked; whether the excerpts actually support the answer still needs your review. Drawing text is a transcription, not verified geometry.</li>
</ul>
</details>
</form>
<p id="status" role="status" aria-live="polite"></p>
</section>
<section aria-label="Results">
<p id="empty" class="empty">Results will appear here. Ask a question to find matching excerpts with their exact location.</p>
<div id="transcript"></div>
</section>
</div>
<aside class="source" id="source-panel" aria-labelledby="source-title">
<h2 id="source-title" tabindex="-1">Source</h2>
<div id="source"><p class="empty">Choose “Inspect source” on an excerpt to see its full text, location and original image here.</p></div>
</aside>
</main>
<script>
const $=id=>document.getElementById(id);
function el(tag,text,cls){const e=document.createElement(tag);if(text!=null)e.textContent=text;if(cls)e.className=cls;return e}
async function call(url,opts){let r,x=null;try{r=await fetch(url,opts)}catch(e){throw Error('Could not reach the local server.')}try{x=await r.json()}catch(e){x=null}if(!r.ok)throw Error((x&&typeof x.error==='string'&&x.error)||'Request failed ('+r.status+')');if(x===null)throw Error('Unexpected server response.');return x}
const METHODS={bm25:'Keyword',dense:'Semantic',hybrid:'Hybrid'};
const mobile=window.matchMedia('(max-width: 900px)');
let busy=false,genEnabled=false,epoch=0,srcSeq=0,selected=null;
function say(msg,isErr){const s=$('status');s.textContent=msg;s.classList.toggle('error',!!isErr)}
function setBusy(b){busy=b;$('clear').disabled=b;$('find').disabled=b;$('ask').disabled=b||!genEnabled;$('composer').setAttribute('aria-busy',String(b))}
function sourceEmpty(){return el('p','Choose “Inspect source” on an excerpt to see its full text, location and original image here.','empty')}
function locationText(l){return [l.page?'Page '+l.page:'',l.sheet?'Sheet '+l.sheet:'',l.sheet_state&&l.sheet_state!=='visible'?'Hidden sheet':'',l.row?'Row '+l.row:'',l.cells||'',l.requirement_id||'',l.line_start?'Lines '+l.line_start+'–'+l.line_end:'',l.image?'Drawing image':''].filter(Boolean).map(String).join(' · ')}
function unverified(ex){return el('p','Unverified transcription · '+String(ex.model||'local OCR')+' · compare with the original image','flag')}

call('/api/docs').then(ds=>{if(!Array.isArray(ds))throw Error('Unexpected document list.');ds.forEach(d=>{const o=el('option',String(d));o.value=String(d);$('doc').append(o)});$('lib').textContent=ds.length+(ds.length===1?' document':' documents')+' in the local index'}).catch(e=>{$('lib').textContent='Document list unavailable';say('Could not load documents: '+e.message,true)});

call('/api/config').then(c=>{genEnabled=!!c&&c.generation_enabled===true;const emb=!!c&&c.embeddings_enabled===true;Array.from($('method').options).forEach(o=>{if(o.value!=='bm25')o.disabled=!emb});if(emb&&c.default_method==='hybrid')$('method').value='hybrid';$('library-note').textContent=c.library||'';$('ask').hidden=!genEnabled;$('ask').disabled=busy||!genEnabled;$('cfg-note').textContent=(emb?'Semantic and hybrid search are enabled for this session.':'Semantic and hybrid search are off for this session.')+' '+(genEnabled?'Answer generation is enabled.':'Answer generation is off.')}).catch(()=>{genEnabled=false;$('ask').hidden=true;$('ask').disabled=true;$('cfg-note').textContent='Server configuration could not be loaded, so paid features stay off. Keyword search remains available.'});

function hitItem(h,i){const li=el('li',null,'hit');const sid='S'+(i+1);const head=el('div',null,'hit-head');head.append(el('span',sid,'sid'),el('span',String(h.doc_id||'Unknown document'),'doc'));const loc=locationText(h.location||{});if(loc)head.append(el('span',loc));li.append(head,el('pre',String(h.text||''),'excerpt'));if(h.extraction&&h.extraction.source_verified===false)li.append(unverified(h.extraction));const b=el('button','Inspect source','link');b.type='button';b.setAttribute('aria-label','Inspect source for '+sid+', '+String(h.doc_id||''));b.setAttribute('aria-pressed','false');b.onclick=()=>inspect(h,li,b);li.append(b);return li}

function renderTurn(q,paid,x,body){const hits=Array.isArray(x.hits)?x.hits:[];const a=el('article',null,'turn');a.append(el('p',(paid?'Generated answer':'Evidence')+' · '+(body.doc||'All documents')+' · '+METHODS[body.method],'kind'),el('h3',q));
if(paid){if(typeof x.answer==='string'&&x.answer)a.append(el('p',x.answer,'answer'));if(x.error)a.append(el('p','Generation failed: '+String(x.error),'error'));else if(x.status==='no_evidence')a.append(el('p','No evidence was found, so no answer was generated.','empty'));else{const cites=(Array.isArray(x.citations)?x.citations:[]).map(c=>c&&c.source_id).filter(Boolean).map(String);a.append(el('p',(x.status==='abstained'?'The model declined to answer from these excerpts. ':'')+(cites.length?'Cites '+cites.join(', ')+'. ':'')+'Citation IDs checked; factual support still needs your review.','meta'))}if(typeof x.session_reserved_usd==='number')$('spent').textContent='Reserved so far this session: $'+x.session_reserved_usd.toFixed(4)+'.'}
if(hits.length){a.append(el('p',paid?'Excerpts sent to the model':'Top excerpts. Check whether they answer the question.','meta'));const ol=el('ol',null,'hits');hits.forEach((h,i)=>ol.append(hitItem(h,i)));a.append(ol)}else if(!paid||x.status!=='no_evidence')a.append(el('p','No matching excerpts. Try other wording, another document, or a different search method.','empty'));
$('transcript').prepend(a);$('empty').hidden=true;return hits.length}

async function run(paid){if(busy||(paid&&!genEnabled))return;const q=$('q').value;if(!q.trim()){say('Enter a question first.',true);$('q').focus();return}const body={q,doc:$('doc').value,method:$('method').value,allow_paid:paid};const my=epoch;setBusy(true);say(paid?'Searching and generating an answer…':'Searching documents…');
try{const x=await call('/api/'+(paid?'ask':'search'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(my!==epoch)return;const n=renderTurn(q,paid,x,body);if(paid)say(x.error?'Answer generation failed.':x.status==='answered'?'Answer generated from '+n+' excerpts.':x.status==='abstained'?'The model declined to answer.':'No answer generated.',!!x.error);else say(n?n+(n===1?' excerpt found.':' excerpts found.'):'No matching excerpts found.')}
catch(e){if(my===epoch)say(e.message,true)}finally{setBusy(false)}}

function renderSource(x,h){const nodes=[];const dl=el('dl',null,'prov');[['Document',x.doc_id],['Location',locationText(x.location||{})],['System',x.system],['Revision',x.revision]].forEach(p=>{if(p[1])dl.append(el('dt',p[0]),el('dd',String(p[1])))});nodes.push(dl);
if(x.extraction&&x.extraction.source_verified===false)nodes.push(unverified(x.extraction));nodes.push(el('h3','Indexed text','label'),el('pre',String(x.text||''),'excerpt full'));
if(typeof x.image_url==='string'&&x.image_url.startsWith('/api/image?')){const f=el('figure');const img=document.createElement('img');img.alt='Original source for '+String(x.doc_id||'document');img.src=x.image_url;img.onerror=()=>f.replaceChildren(el('p','Original image could not be loaded.','error'));const pg=x.location&&x.location.page;f.append(img,el('figcaption',pg?'Rendered page '+pg+' of the original file':'Original image'));nodes.push(f)}
$('source').replaceChildren(...nodes)}

async function inspect(h,li,b){const my=++srcSeq;if(selected){selected.li.classList.remove('selected');selected.b.setAttribute('aria-pressed','false')}selected={li,b};li.classList.add('selected');b.setAttribute('aria-pressed','true');const s=$('source');s.setAttribute('aria-busy','true');s.replaceChildren(el('p','Loading source…','empty'));
if(mobile.matches){$('source-panel').scrollIntoView({behavior:'smooth',block:'start'});$('source-title').focus({preventScroll:true})}
try{const x=await call('/api/source?'+new URLSearchParams({chunk_id:String(h.chunk_id||'')}));if(my!==srcSeq)return;renderSource(x,h);say('Source loaded: '+String(x.doc_id||'document')+'.')}
catch(e){if(my===srcSeq){s.replaceChildren(el('p','Could not open source: '+e.message,'error'));say('Could not open source.',true)}}finally{if(my===srcSeq)s.removeAttribute('aria-busy')}}

$('composer').onsubmit=e=>{e.preventDefault();run(false)};
$('q').onkeydown=e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();run(false)}};
$('ask').onclick=()=>run(true);
$('clear').onclick=()=>{epoch++;srcSeq++;selected=null;$('transcript').replaceChildren();$('empty').hidden=false;$('source').removeAttribute('aria-busy');$('source').replaceChildren(sourceEmpty());say('Conversation cleared.')};
</script>
</body>
</html>'''


def create_server(backend, port=8766, *, allow_generation=False):
    docs = set(backend.documents())
    reserved = 0.0
    class Handler(BaseHTTPRequestHandler):
        def respond(self, data, mime='application/json; charset=utf-8', status=200):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def same_host(self):
            return self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}'

        def do_GET(self):
            if not self.same_host():
                return self.respond({'error': 'Loopback Host required'}, status=403)
            parsed = urlsplit(self.path)
            args = parse_qs(parsed.query)
            try:
                if parsed.path == '/':
                    return self.respond(HTML.encode(), 'text/html; charset=utf-8')
                if parsed.path == '/api/docs':
                    return self.respond(sorted(docs))
                if parsed.path == '/api/config':
                    return self.respond({'generation_enabled': allow_generation,
                                         'library': getattr(backend, 'library_label', ''),
                                         'default_method': getattr(backend, 'default_method', 'bm25'),
                                         'embeddings_enabled': bool(getattr(backend, 'embedder', None))})
                if parsed.path == '/api/source':
                    return self.respond(backend.source(args.get('chunk_id', [''])[0]))
                if parsed.path == '/api/image':
                    return self.respond(backend.image(args.get('chunk_id', [''])[0]), 'image/png')
                return self.respond({'error': 'Not found'}, status=404)
            except (ValueError, OSError, RuntimeError, KeyError) as exc:
                return self.respond({'error': str(exc)}, status=400)

        def do_POST(self):
            nonlocal reserved
            if not self.same_host() or self.headers.get('Origin') != f'http://127.0.0.1:{self.server.server_port}':
                return self.respond({'error': 'Same-origin loopback request required'}, status=403)
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 12000:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('JSON object required')
                if self.path not in ('/api/search', '/api/ask'):
                    return self.respond({'error': 'Not found'}, status=404)
                paid = self.path == '/api/ask'
                if paid and (not allow_generation or body.get('allow_paid') is not True):
                    return self.respond({'error': 'Explicit generation opt-in required'}, status=403)
                q, doc, method = body.get('q'), body.get('doc') or None, body.get('method', 'bm25')
                if not isinstance(q, str) or not q.strip() or len(q) > 2000:
                    raise ValueError('Question must contain 1–2000 characters')
                if doc is not None and (not isinstance(doc, str) or doc not in docs):
                    raise ValueError('Unknown document')
                if method not in ('bm25', 'dense', 'hybrid'):
                    raise ValueError('Unknown retrieval method')
                hits = backend.search(q, doc_id=doc, limit=5, method=method)
                if not paid:
                    return self.respond({'hits': hits, 'status': 'evidence'})
                if hits:
                    evaluate_answers.current_prices()
                    cost = evaluate_answers.request_bound(q, hits)['cost_upper_bound_usd']
                    if cost > .01 or reserved + cost > .05 or answering.OPTIONS['max_tokens'] != 384:
                        raise ValueError('Generation budget reached or configuration changed')
                    reserved += cost
                result = answering.answer_from_hits(q, hits, allow_paid=True,
                    provider_max_price=evaluate_answers.PRICE_CEILINGS.copy())
                result.pop('messages', None)
                result.pop('raw_response', None)
                result.update(hits=hits, session_reserved_usd=reserved)
                return self.respond(result)
            except (ValueError, OSError, RuntimeError, TypeError, KeyError) as exc:
                return self.respond({'error': str(exc)}, status=400)
    return HTTPServer(('127.0.0.1', port), Handler)


def main():
    from .workspace import Workspace
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, default=Path('data/workspace'))
    p.add_argument('--corpus-root', type=Path, help='Use the full PDF corpus and its saved OpenAI embeddings')
    p.add_argument('--port', type=int, default=8766)
    p.add_argument('--enable-paid-generation', action='store_true')
    p.add_argument('--enable-paid-embeddings', action='store_true')
    a = p.parse_args()
    from dotenv import load_dotenv
    load_dotenv(override=False)
    if a.corpus_root:
        from .corpus_workspace import CorpusWorkspace
        backend = CorpusWorkspace(a.corpus_root, allow_paid=a.enable_paid_embeddings)
    else:
        backend = Workspace(a.index, allow_paid=a.enable_paid_embeddings)
    server = create_server(backend, a.port, allow_generation=a.enable_paid_generation)
    print(f'Workspace chat: http://127.0.0.1:{server.server_port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
