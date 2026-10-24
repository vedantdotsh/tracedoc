"""Pinned MMLongBench-Doc corpus downloader (stdlib only)."""
import concurrent.futures as cf
import hashlib
import http.client
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = 'mayubo2333/MMLongBench-Doc'
REVISION = 'd73f0dc0be7e0a2ff6a403d5fe65fcd96461f384'
EXPECTED_PDFS = 135
MAX_WORKERS = 8
TREE_URL = f'https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1'
RAW_URL = f'https://raw.githubusercontent.com/{REPO}/{REVISION}/'
DOC_PREFIX = 'data/documents/'
SAMPLES_PATH = 'data/samples.json'
LICENSE_NOTE = ('The repository license does not imply a license for the source PDFs; '
                'each document keeps its own terms.')
HEADERS = {'User-Agent': 'document-evidence-lab'}


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.',
                                     suffix='.tmp', delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(data)
        except BaseException:
            stream.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def require_distinct_paths(*paths) -> None:
    paths = [Path(p).resolve() for p in paths]
    for i, path in enumerate(paths):
        for other in paths[:i]:
            if path == other or (path.exists() and other.exists() and path.samefile(other)):
                raise ValueError(f'input and output paths must be distinct: {path}')


def write_json_atomic(path: Path, obj) -> None:
    write_atomic(Path(path), json.dumps(obj, indent=2, ensure_ascii=False).encode('utf-8'))


def _retry(fn, tries=4):
    for attempt in range(tries):
        try:
            return fn()
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise
            err = exc
        except (OSError, http.client.HTTPException) as exc:
            err = exc
        if attempt == tries - 1:
            raise err
        time.sleep(2 ** attempt)


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as resp:
        return resp.read()


def _git_hasher(size: int):
    return hashlib.sha1(f'blob {size}\0'.encode())


def _fetch_file(url, dest: Path, size: int, git_sha1: str, pdf: bool) -> str:
    part = dest.with_name(dest.name + '.part')
    git, sha, count = _git_hasher(size), hashlib.sha256(), 0
    try:
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=60) as resp, part.open('wb') as out:
            while chunk := resp.read(1 << 20):
                if count == 0 and pdf and not chunk.startswith(b'%PDF-'):
                    raise ValueError('missing %PDF- header')
                git.update(chunk)
                sha.update(chunk)
                out.write(chunk)
                count += len(chunk)
        if count != size or git.hexdigest() != git_sha1:
            raise ConnectionError(f'size/blob sha1 mismatch ({count} bytes)')
        os.replace(part, dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return sha.hexdigest()


def _verify_existing(dest: Path, size: int, git_sha1: str, pdf: bool):
    if not dest.is_file() or dest.stat().st_size != size:
        return None
    git, sha, first = _git_hasher(size), hashlib.sha256(), True
    with dest.open('rb') as f:
        while chunk := f.read(1 << 20):
            if first and pdf and not chunk.startswith(b'%PDF-'):
                return None
            first = False
            git.update(chunk)
            sha.update(chunk)
    return sha.hexdigest() if git.hexdigest() == git_sha1 else None


def _sync(entry: dict, dest: Path, pdf: bool) -> dict:
    size, git_sha1 = entry['size'], entry['sha']
    url = RAW_URL + urllib.parse.quote(entry['path'])
    sha256 = _verify_existing(dest, size, git_sha1, pdf)
    status = 'existing'
    if sha256 is None:
        status = 'downloaded'
        sha256 = _retry(lambda: _fetch_file(url, dest, size, git_sha1, pdf))
    return {'path': dest.relative_to(dest.parent.parent).as_posix(), 'url': url,
            'git_blob_sha1': git_sha1, 'size': size, 'sha256': sha256, 'status': status}


def _load_tree(root: Path) -> dict:
    cache = root / 'metadata' / 'tree.json'
    tree = None
    if cache.is_file():
        try:
            tree = json.loads(cache.read_text('utf-8'))
        except ValueError:
            tree = None
    if not isinstance(tree, dict) or tree.get('sha') != REVISION or tree.get('truncated'):
        tree = json.loads(_retry(lambda: _get(TREE_URL)))
        if tree.get('sha') != REVISION or tree.get('truncated'):
            raise RuntimeError('GitHub tree is truncated or not the pinned revision')
        write_atomic(cache, json.dumps(tree, indent=2).encode('utf-8'))
    return tree


def _safe_name(name: str) -> bool:
    return (name not in ('', '.', '..') and name == Path(name).name
            and not any(c in name for c in '/\\:'))


def download(root: Path, workers: int = 4) -> dict:
    """Download and verify the pinned corpus under root; failures are recorded, never dropped."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    workers = min(workers, MAX_WORKERS)
    root = Path(root)
    blobs = {e['path']: e for e in _load_tree(root)['tree'] if e.get('type') == 'blob'}
    if SAMPLES_PATH not in blobs:
        raise RuntimeError(f'{SAMPLES_PATH} missing from pinned tree')
    jobs, failures, tree_pdfs = [], [], 0
    for path, entry in sorted(blobs.items()):
        name = path[len(DOC_PREFIX):]
        if not (path.startswith(DOC_PREFIX) and name.lower().endswith('.pdf')):
            continue
        tree_pdfs += 1
        if _safe_name(name):
            jobs.append((name, entry))
        else:
            failures.append({'doc_id': name, 'error': 'unsafe filename'})
    if tree_pdfs != EXPECTED_PDFS:
        failures.append({'doc_id': None,
                         'error': f'expected {EXPECTED_PDFS} PDFs, tree lists {tree_pdfs}'})

    def run(name, entry, dest, pdf):
        try:
            return {'doc_id': name, **_sync(entry, dest, pdf)}
        except Exception as exc:
            return {'doc_id': name, 'error': f'{type(exc).__name__}: {exc}'}

    docs_dir = root / 'documents'
    docs_dir.mkdir(parents=True, exist_ok=True)
    (root / 'benchmark').mkdir(parents=True, exist_ok=True)
    samples = run('samples.json', blobs[SAMPLES_PATH], root / 'benchmark' / 'samples.json', False)
    documents, total = [], len(jobs)
    with cf.ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(run, name, entry, docs_dir / name, True) for name, entry in jobs]
        for done, future in enumerate(cf.as_completed(futures), 1):
            documents.append(future.result())
            if done % 10 == 0 or done == total:
                print(f'{done}/{total} documents', flush=True)
    documents.sort(key=lambda r: r['doc_id'])
    failures += [r for r in documents if 'error' in r]
    if 'error' in samples:
        failures.append(samples)
    manifest = {
        'repository': REPO, 'revision': REVISION, 'tree_url': TREE_URL,
        'retrieved_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'license_note': LICENSE_NOTE, 'expected_pdfs': EXPECTED_PDFS,
        'tree_pdfs': tree_pdfs, 'verified_pdfs': sum('error' not in r for r in documents),
        'samples': samples, 'documents': [r for r in documents if 'error' not in r],
        'failures': failures, 'ok': not failures,
    }
    write_json_atomic(root / 'manifest.json', manifest)
    return manifest
