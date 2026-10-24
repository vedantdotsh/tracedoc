"""Capture exact downloaded weights and source chunk token truncation exposure."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
from .corpus import write_json_atomic

def model_files(cache):
    cache=Path(cache)
    paths=[p for p in cache.rglob('*') if p.is_file() and
           ('snapshots' in p.parts or ('reranker' in p.parts and '__MACOSX' not in p.parts))]
    return {p.relative_to(cache).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}

def verify(cache):
    path=Path(cache)/'model-provenance.json'
    if path.exists() and model_files(cache)!=json.loads(path.read_text('utf-8'))['files']:
        raise ValueError('Model files changed since the recorded experiment; use a new cache')

def audit(root=Path('.')):
    from tokenizers import Tokenizer
    from .retrieval import Retriever
    root=Path(root); cache=root/'data/dense-v1'
    verify(cache)
    rows=Retriever(root/'data/baseline.sqlite',cache).rows
    tokenizer_path=next((cache/'models').glob('**/snapshots/*/tokenizer.json'))
    tok=Tokenizer.from_file(str(tokenizer_path));tok.no_truncation();tok.no_padding()
    lengths=[len(tok.encode(r['text']).ids) for r in rows]
    result={'files':model_files(cache),'embedding_hf_revision':tokenizer_path.parent.name,
            'packages':{k:importlib.metadata.version(k) for k in ('fastembed','flashrank','onnxruntime','numpy','tokenizers')},
            'embedding_max_tokens':512,'chunks':len(rows),'chunks_over_512_tokens':sum(n>512 for n in lengths),
            'max_chunk_tokens':max(lengths),'reranker_max_pair_tokens':512,
            'note':'Embedding truncates above 512 tokens; reranker jointly truncates question+chunk. No claim that 180 words always fits. Model checksums identify exact downloaded weights; do not imply upstream immutability.'}
    write_json_atomic(cache/'model-provenance.json',result)
    write_json_atomic(root/'reports/retrieval-v1/model-provenance.json',result)
    return result

if __name__=='__main__':
    r=audit();print(json.dumps({k:v for k,v in r.items() if k!='files'},indent=2))
