"""Run bounded OCR workers, checkpoint each page and restart crashed/timed-out workers."""
import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_lab.ocr import prepare_manifest, valid_cache, recover_page, write_json_atomic


def worker(pipe, settings, digest, pdf_dir, output):
    while True:
        entry = pipe.recv()
        if entry is None:
            return
        result = recover_page(entry, settings, digest, pdf_dir, output)
        pipe.send({'status': result['status'], 'seconds': result['seconds']})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', default='data/baseline.sqlite')
    parser.add_argument('--pdf-dir', default='data/documents')
    parser.add_argument('--output', default='data/ocr-v1')
    parser.add_argument('--reports', default='reports/ocr-v1')
    parser.add_argument('--workers', type=int, default=3, choices=range(1, 5))
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--retry-errors', action='store_true', help='Retry cached errors once, preserving prior records')
    args = parser.parse_args()
    out = Path(args.output)
    manifest = prepare_manifest(args.baseline, args.pdf_dir, out)
    pages, settings, digest = manifest['pages'], manifest['settings'], manifest['settings_sha256']
    write_json_atomic(Path(args.reports) / 'provenance.json', {k: v for k, v in manifest.items() if k != 'pages'})
    if args.pilot:
        selected = [pages[round(i * (len(pages) - 1) / 4)] for i in range(5)]
        results = [recover_page(p, settings, digest, args.pdf_dir, out, pilot=True) for p in selected]
        write_json_atomic(Path(args.reports) / 'pilot.json', {'selection': 'five evenly spaced missing-page ordinal positions; labels never read', 'pages': results})
        print(json.dumps([{'doc_id': r['doc_id'], 'page': r['page'], 'status': r['status'], 'seconds': r['seconds'], 'text': r['text'][:300]} for r in results]), flush=True)
        return
    pending = []
    for entry in pages:
        cache = out / entry['cache_path']
        if not valid_cache(cache, entry, digest):
            pending.append(entry)
        elif args.retry_errors:
            prior = json.loads(cache.read_text(encoding='utf-8'))
            if prior['status'] == 'error':
                history = out / 'error-history' / Path(entry['cache_path']).parent.name / (cache.stem + f'-{time.time_ns()}.json')
                write_json_atomic(history, prior)
                pending.append(entry)
    done = len(pages) - len(pending)
    reused_pages = done
    started = time.monotonic()
    slots = []
    def spawn():
        parent, child = mp.Pipe()
        proc = mp.Process(target=worker, args=(child, settings, digest, args.pdf_dir, out))
        proc.start()
        child.close()
        return {'proc': proc, 'pipe': parent, 'entry': None}
    try:
        slots = [spawn() for _ in range(args.workers)]
        while pending or any(s['entry'] for s in slots):
            for i, slot in enumerate(slots):
                if slot['entry'] and slot['pipe'].poll() and slot['proc'].is_alive():
                    try:
                        slot['pipe'].recv()
                        slot['entry'] = None
                        done += 1
                    except EOFError:
                        pass
                if slot['entry'] and (not slot['proc'].is_alive() or time.monotonic() - slot['start'] > args.timeout):
                    entry = slot['entry']
                    error = 'worker native crash' if not slot['proc'].is_alive() else f'page exceeded {args.timeout}s timeout'
                    slot['proc'].terminate()
                    slot['proc'].join(10)
                    slot['pipe'].close()
                    if not valid_cache(out / entry['cache_path'], entry, digest):
                        write_json_atomic(out / entry['cache_path'], {**entry, 'settings_sha256': digest, 'engine': settings['engine'], 'status': 'error', 'text': '', 'seconds': time.monotonic() - slot['start'], 'error': error})
                    done += 1
                    slot = slots[i] = spawn()
                if slot['entry'] is None and pending:
                    slot['entry'] = pending.pop(0)
                    slot['start'] = time.monotonic()
                    slot['pipe'].send(slot['entry'])
            progress = {'completed': done, 'total': len(pages), 'remaining': len(pages) - done, 'wall_seconds': round(time.monotonic() - started, 2)}
            if done != getattr(main, 'last', None) or time.monotonic() - getattr(main, 'last_write', 0) >= 5:
                write_json_atomic(Path(args.reports) / 'progress.json', progress)
                main.last_write = time.monotonic()
            if done != getattr(main, 'last', None):
                print(json.dumps(progress), flush=True)
                main.last = done
            time.sleep(.25)
    finally:
        for slot in slots:
            if slot['proc'].is_alive():
                slot['pipe'].send(None)
                slot['proc'].join(5)
                if slot['proc'].is_alive():
                    slot['proc'].terminate()
            slot['pipe'].close()
    results = [json.loads((out / p['cache_path']).read_text(encoding='utf-8')) for p in pages]
    counts = Counter(r['status'] for r in results)
    report = {'selected_pages': len(pages), 'documents_with_missing_pages': len({p['doc_id'] for p in pages}),
              'status_counts': dict(counts), 'recovered_characters': sum(len(r['text']) for r in results),
              'page_seconds_sum': sum(r['seconds'] for r in results), 'run_wall_seconds': time.monotonic() - started,
              'workers': args.workers, 'reused_pages': reused_pages,
              'timing_note': 'run_wall_seconds covers this invocation only; prior checkpoints and pilot time are excluded. page_seconds_sum sums final page artifacts, not end-to-end wall time.',
              'settings_sha256': digest, 'errors': [{k: r.get(k) for k in ('doc_id', 'page', 'error')} for r in results if r['status'] == 'error'],
              'accuracy': 'Not measured: no transcription gold. Confidence is an engine score, not calibrated accuracy.'}
    write_json_atomic(Path(args.reports) / 'coverage.json', report)
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
