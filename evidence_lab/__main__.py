"""Command line: python -m evidence_lab {download,index,search,evaluate}."""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

from . import baseline, corpus, audit, answering
from . import sources, evidence_eval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m evidence_lab')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('download', help='download and verify the pinned PDFs')
    p.add_argument('--root', type=Path, default=Path('data'))
    p.add_argument('--workers', type=int, default=4)

    p = sub.add_parser('index', help='build the FTS5 index from PDFs')
    p.add_argument('--pdf-dir', type=Path, default=Path('data/documents'))
    p.add_argument('--db', type=Path, default=Path('data/baseline.sqlite'))
    p.add_argument('--words', type=int, default=180)
    p.add_argument('--overlap', type=int, default=30)
    p.add_argument('--strategy', choices=['fixed', 'paragraph', 'heading', 'table'], default='fixed')

    p = sub.add_parser('search', help='BM25 search over the index')
    p.add_argument('query')
    p.add_argument('--db', type=Path, default=Path('data/baseline.sqlite'))
    p.add_argument('--doc-id')
    p.add_argument('--top-k', type=int, default=5)

    p = sub.add_parser('evaluate', help='measure page retrieval on benchmark samples')
    p.add_argument('--samples', type=Path, default=Path('data/benchmark/samples.json'))
    p.add_argument('--db', type=Path, default=Path('data/baseline.sqlite'))
    p.add_argument('--output', type=Path, default=Path('reports/baseline.json'))
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--overwrite', action='store_true')

    p = sub.add_parser('audit', help='record incompatible annotations without changing raw data')
    p.add_argument('--samples', type=Path, default=Path('data/benchmark/samples.json'))
    p.add_argument('--db', type=Path, default=Path('data/baseline.sqlite'))
    p.add_argument('--output', type=Path, default=Path('reports/annotation-audit.json'))
    p.add_argument('--eligible', type=Path, default=Path('data/benchmark/eligible.json'))
    p.add_argument('--overwrite', action='store_true')

    p = sub.add_parser('ask', help='generate a cited answer using OpenRouter')
    p.add_argument('query')
    p.add_argument('--db', type=Path, default=Path('data/baseline.sqlite'))
    p.add_argument('--doc-id')
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--model', default=answering.DEFAULT_MODEL)
    p.add_argument('--allow-paid', action='store_true', help='explicitly opt in to an OpenRouter request')
    p.add_argument('--output', type=Path)
    p = sub.add_parser('ingest', help='index PDF/XLSX/requirements/drawing files with located evidence')
    p.add_argument('paths', nargs='+', type=Path)
    p.add_argument('--db', type=Path, default=Path('data/evidence-index/evidence.sqlite'))
    p.add_argument('--system')
    p.add_argument('--revision')
    p.add_argument('--meta', type=Path, help="JSON {'*': {...}, '<file>': {'system','revision'}}")
    p.add_argument('--ocr', action='store_true', help='OCR images and text-less PDF pages (RapidOCR)')
    p.add_argument('--openrouter-ocr', action='store_true', help='Paid OpenRouter vision transcription of images and text-less PDF pages')
    p.add_argument('--overwrite', action='store_true')

    p = sub.add_parser('evidence-search', help='BM25 search with source hash, location and units')
    p.add_argument('query')
    p.add_argument('--db', type=Path, default=Path('data/evidence-index/evidence.sqlite'))
    p.add_argument('--doc-id')
    p.add_argument('--top-k', type=int, default=5)

    p = sub.add_parser('evidence-eval', help='retrieval, citation, fact and entailment-review report')
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--db', type=Path, default=Path('data/evidence-index/evidence.sqlite'))
    p.add_argument('--output', type=Path, default=Path('reports/evidence-eval.json'))
    p.add_argument('--answers', type=Path, help='JSON {case id: answering result}')
    p.add_argument('--reviews', type=Path, help='JSON {case id: {verdict, reviewer}}')
    p.add_argument('--source-dir', type=Path, help='re-hash cited sources against the index')
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--overwrite', action='store_true')
    return parser


def run(args) -> int:
    if args.command == 'download':
        manifest = corpus.download(args.root, args.workers)
        print(f"verified {manifest['verified_pdfs']} of {manifest['tree_pdfs']} PDFs "
              f"(expected {manifest['expected_pdfs']}); failures: {len(manifest['failures'])}")
        for failure in manifest['failures']:
            print(json.dumps(failure), file=sys.stderr)
        return 0 if manifest['ok'] else 1
    if args.command == 'index':
        result = baseline.build_index(args.pdf_dir, args.db, args.words, args.overlap, args.strategy)
        print(json.dumps(result, indent=2))
    elif args.command == 'search':
        hits = baseline.search(args.db, args.query, args.doc_id, args.top_k)
        print(json.dumps(hits, indent=2, ensure_ascii=False))
    elif args.command == 'audit':
        corpus.require_distinct_paths(args.samples, args.db, args.output, args.eligible)
        if not args.overwrite and any(p.exists() for p in (args.output, args.eligible)):
            raise ValueError('audit output exists; pass --overwrite to replace it')
        samples = json.loads(args.samples.read_text(encoding='utf-8'))
        with baseline._connect(args.db) as con:
            page_counts = dict(con.execute('SELECT doc_id, pages FROM docs'))
        con.close()
        eligible, report = audit.audit_samples(samples, page_counts)
        corpus.write_json_atomic(args.output, report)
        corpus.write_json_atomic(args.eligible, eligible)
        print(f"{len(eligible)}/{len(samples)} eligible; {report['quarantined_rows']} quarantined; "
              f"see {args.output}")
    elif args.command == 'ask':
        if args.output:
            corpus.require_distinct_paths(args.db, args.output)
        if args.output and args.output.exists():
            raise ValueError(f'{args.output} already exists; choose a new output file')
        from dotenv import load_dotenv
        load_dotenv(Path('.env'), override=False)
        result = answering.answer(args.db, args.query, args.doc_id, args.top_k, args.model,
                                  args.allow_paid)
        if args.output:
            corpus.write_json_atomic(args.output, result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result['execution_status'] in ('ok', 'not_run') else 1
    elif args.command == 'evaluate':
        corpus.require_distinct_paths(args.samples, args.db, args.output)
        if args.output.exists() and not args.overwrite:
            raise ValueError(f'{args.output} exists; pass --overwrite to replace it')
        samples = json.loads(args.samples.read_text(encoding='utf-8'))
        if not isinstance(samples, list):
            raise ValueError('samples file must contain a JSON list')
        report = baseline.evaluate(args.db, samples, args.top_k)
        corpus.write_json_atomic(args.output, report)
        print(f"wrote {args.output}: {report['answerable']} answerable, "
              f"{report['unanswerable']} unanswerable, mean_recall={report['mean_recall']}, "
              f"hit={report['mean_hit']}, mrr={report['mrr']}")
    elif args.command == 'ingest':
        if args.db.exists() and not args.overwrite:
            raise ValueError(f'{args.db} exists; pass --overwrite to replace it')
        metadata = json.loads(args.meta.read_text(encoding='utf-8')) if args.meta else {}
        metadata.setdefault('*', {}).update({k: v for k, v in (
            ('system', args.system), ('revision', args.revision)) if v})
        if args.ocr and args.openrouter_ocr:
            raise ValueError('Choose local --ocr or --openrouter-ocr, not both')
        ocr = sources.rapidocr_lines if args.ocr else None
        if args.openrouter_ocr:
            from dotenv import load_dotenv
            from .api_ocr import OpenRouterOCR
            load_dotenv(override=False)
            ocr = OpenRouterOCR(args.db.parent / 'openrouter-ocr')
        result = sources.build_index(sources.expand(args.paths), args.db, metadata, ocr)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == 'evidence-search':
        hits = evidence_eval.search(args.db, args.query, args.doc_id, args.top_k)
        print(json.dumps(hits, indent=2, ensure_ascii=False))
    elif args.command == 'evidence-eval':
        corpus.require_distinct_paths(args.cases, args.db, args.output,
                                      *[p for p in (args.answers, args.reviews) if p])
        if args.output.exists() and not args.overwrite:
            raise ValueError(f'{args.output} exists; pass --overwrite to replace it')
        load = lambda p: json.loads(p.read_text(encoding='utf-8')) if p else None
        report = evidence_eval.evaluate(args.db, load(args.cases), args.top_k, load(args.answers),
                                        load(args.reviews), args.source_dir)
        corpus.write_json_atomic(args.output, report)
        s = report['summary']
        print(f"wrote {args.output}: {s['answerable']} answerable, {s['unanswerable']} unanswerable, "
              f"hit={s['retrieval']['hit']}, citations valid {s['citations']['all_valid']}/"
              f"{s['citations']['answered']}, human entailment pending {s['human_entailment']['pending']}")
    return 0


def main(argv=None) -> int:
    # Windows redirected terminals may default to a legacy code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
