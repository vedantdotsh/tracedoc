# TraceDoc

A Python workspace for searching documents, asking questions with cited sources, and inspecting the original evidence.

## Features

- Import PDFs, Excel workbooks, requirements files and drawing images.
- Search with SQLite BM25, semantic embeddings or reciprocal-rank hybrid fusion.
- Keep source locations: PDF pages, spreadsheet cells and requirement identifiers.
- Browse retrieved excerpts alongside original PDF pages and images.
- Generate cited answers through OpenRouter with explicit opt-in and request budgets.
- Use local OCR or optional API transcription for scanned documents and drawing labels.

## Quick start

Python 3.11 or newer is required. These commands use PowerShell; on macOS/Linux, use `.venv/bin/python` in place of `.venv\Scripts\python.exe`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[workspace,dev]" "openpyxl>=3.1,<4"
.\.venv\Scripts\python.exe scripts/create_workspace_sample.py
.\.venv\Scripts\python.exe -m evidence_lab ingest data/workspace-sample/manual.pdf data/workspace-sample/drawing.pdf data/workspace-sample/limits.xlsx data/workspace-sample/requirements.csv data/workspace-sample/requirements.md --db data/workspace/evidence.sqlite --system SYNTHETIC-DEMO --revision R1
.\.venv\Scripts\python.exe -m evidence_lab.workspace_chat --index data/workspace
```

Open the printed local address. Try **“DEMO-A service interval hours”**, then select **Inspect source**. The sample is synthetic and keyword search needs no API key. The generator creates its own sample files and will not overwrite an existing sample folder.

See [the workspace guide](docs/WORKSPACE.md) for your own documents, OCR and optional semantic search and generation.

## Evaluation

Retrieval was evaluated on the [MMLongBench-Doc](https://github.com/mayubo2333/MMLongBench-Doc) benchmark: 135 PDFs, 6,529 pages and 837 questions with annotated evidence pages. hit@5 is the share of questions for which at least one annotated evidence page is among the top five retrieved pages. "Whole library" searches all 135 PDFs at once and counts a result only when both the document and the page are correct.

| Index and embedding model | Whole-library hybrid hit@5 | Single-document hybrid hit@5 |
|---|---:|---:|
| Text embedded in the PDFs, BGE-small | 41.9% (351/837) | 58.8% (492/837) |
| Plus OCR text for pages without embedded text, BGE-small | 55.8% (467/837) | 76.3% (639/837) |
| Same OCR index, OpenAI `text-embedding-3-small` | 60.9% (510/837) | 76.0% (636/837) |

OCR recovery and OpenAI embeddings raised whole-library hit@5 from 41.9% to 60.9%. OCR raised the number of pages with indexed text from 4,911 to 6,510 of 6,529.

These figures measure retrieval of annotated evidence pages, not the accuracy of generated answers. The method, all six scope/method results and the limits of the evaluation are in [docs/EVALUATION.md](docs/EVALUATION.md). Per-run summaries, per-document metrics and per-question results are in [reports/corpus-audit-v1](reports/corpus-audit-v1/), [reports/corpus-audit-ocr-v1](reports/corpus-audit-ocr-v1/) and [reports/corpus-audit-openai-v1](reports/corpus-audit-openai-v1/).

```powershell
.\.venv\Scripts\python.exe scripts/verify_reported_metrics.py
```

This recomputes every figure above from the per-question results. The benchmark PDFs and question text are not redistributed here; `python -m evidence_lab download` fetches them for a full re-run.

## Development

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,notebook,retrieval,workspace]"
.\.venv\Scripts\python.exe -m pytest -q
```

The test suite has 221 tests and runs without the benchmark, model downloads or network access.

The source includes document-ingestion, retrieval, citation and evaluation utilities. Local documents, indexes, caches, credentials and experiment outputs other than the evaluation reports above are excluded from the repository.

## Usage notes

This is a local research application. Citation checks confirm references to supplied evidence; inspect the original sources when factual support matters. Drawing ingestion extracts labels rather than CAD geometry. STEP and legacy `.xls` files are not supported.

API features send questions, selected excerpts or images to OpenRouter only when enabled. Keep `OPENROUTER_API_KEY` in a local `.env` file; never commit it. The server binds to loopback and is intended for local use.

The optional benchmark utilities use [MMLongBench-Doc](https://github.com/mayubo2333/MMLongBench-Doc). External documents retain their original terms and are not bundled here.
