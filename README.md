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

## Development

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,notebook,retrieval,workspace]"
.\.venv\Scripts\python.exe -m pytest -q
```

The source includes document-ingestion, retrieval, citation and evaluation utilities. Local documents, indexes, caches, experiment outputs and credentials are excluded from the repository.

## Usage notes

This is a local research application. Citation checks confirm references to supplied evidence; inspect the original sources when factual support matters. Drawing ingestion extracts labels rather than CAD geometry. STEP and legacy `.xls` files are not supported.

API features send questions, selected excerpts or images to OpenRouter only when enabled. Keep `OPENROUTER_API_KEY` in a local `.env` file; never commit it. The server binds to loopback and is intended for local use.

The optional benchmark utilities use [MMLongBench-Doc](https://github.com/mayubo2333/MMLongBench-Doc). External documents retain their original terms and are not bundled here.
