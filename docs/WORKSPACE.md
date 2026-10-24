# Document workspace

## Import your own files

Run from the repository root after the quickstart installation:

```powershell
.\.venv\Scripts\python.exe -m evidence_lab ingest C:\path\to\documents --db data/my-workspace/evidence.sqlite --system SYSTEM-A --revision R1
.\.venv\Scripts\python.exe -m evidence_lab.workspace_chat --index data/my-workspace
```

Directory ingestion is nonrecursive and filenames must be unique. Use `--meta metadata.json` for per-file system/revision metadata. CSV records may supply their own values. Replacing an existing index requires `--overwrite`; ingestion stages the new database before replacement.

| Format | Source information retained |
|---|---|
| PDF | Physical page, text blocks and coordinates |
| XLSX | Sheet, row and cells; first nonempty row as headers |
| Requirements CSV | `id,text` columns; optional system, revision, status and unit |
| Requirements TXT/MD | Identifiers such as `REQ-101: text` and indented continuation lines |
| Drawing PNG/JPEG/PDF | Text labels, source image/page and available OCR coordinates |

XLSX formulas are not calculated; saved results are used where present. Complex headers and drawings should be checked in the source viewer. STEP, ReqIF and legacy XLS need conversion outside this application.

## OCR

Install `.[ocr]` and add `--ocr` to ingestion for local RapidOCR. Model weights may download on first use. OCR processes image files and PDF pages without native text. Pages with both native text and embedded images may need separate preparation.

Alternatively, `--openrouter-ocr` sends image bytes for paid transcription through OpenRouter. It requires `OPENROUTER_API_KEY` in the local ignored `.env`. API transcription uses a durable cache budget and records source/page provenance. Use one OCR option per ingestion run.

The quickstart omits the sample's duplicate drawing PNG so it needs no OCR installation. To include it, ingest the full generated sample directory with `--ocr` into a new index.

## Semantic search and cited answers

Configure `OPENROUTER_API_KEY` in `.env`, then build the document embeddings explicitly:

```powershell
.\.venv\Scripts\python.exe -m evidence_lab.workspace --index data/my-workspace --embed --allow-paid
.\.venv\Scripts\python.exe -m evidence_lab.workspace_chat --index data/my-workspace --enable-paid-embeddings --enable-paid-generation
```

Embeddings use OpenAI `text-embedding-3-small` through OpenRouter, with normalized NumPy vectors. Hybrid search combines semantic and BM25 ranks. The chat server never embeds the whole collection implicitly.

Semantic queries may incur an embedding request. Answer generation is a separate button and sends selected excerpts to the answer model. Generation is capped at $0.01 per request and $0.05 reserved per server session; embedding caches default to a $0.10 reservation budget. Each question is independent of earlier turns. Source inspection checks file hashes before showing the indexed evidence.

## Local assets

Source documents, databases, model/vector caches, generated notebooks and saved experiment results are local assets. They are not included in Git. The optional `--corpus-root` adapter expects a separately populated OCR corpus and matching saved vectors; use the mixed-document workspace above for a new checkout.

Historical benchmark runners in `scripts/` may require matching local inputs and prior result artifacts. They are research utilities, separate from the quickstart.
