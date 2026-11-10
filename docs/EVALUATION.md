# Retrieval evaluation

This page describes how the retrieval figures in the README were measured and how to check them.

## Dataset

The evaluation uses [MMLongBench-Doc](https://github.com/mayubo2333/MMLongBench-Doc) at revision `d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384`: 135 PDFs with 6,529 pages and 1,082 annotated questions. `python -m evidence_lab download` fetches the PDFs and checks each file against a pinned SHA-256.

`python -m evidence_lab audit` keeps the 1,052 questions whose page annotations can be scored against physical PDF pages and sets aside 30 (12 answers without page evidence, 9 with incompatible page coordinates, 7 unanswerable questions that still list evidence pages, 2 conflicting duplicates). Of the 1,052, **837 list at least one evidence page** and are the questions scored below. The other 215 are unanswerable by design, have no page to retrieve, and are not counted.

## Method

Each PDF is split into 180-word chunks with a 30-word overlap. A question is searched three ways:

- **BM25**: SQLite FTS5 keyword ranking.
- **Dense**: cosine similarity over chunk embeddings, exact NumPy search.
- **Hybrid**: reciprocal-rank fusion of the BM25 and dense rankings (constant 60).

Every search takes 30 candidate chunks and keeps the first five distinct pages. Two scopes are measured:

- **Whole library** (`global_*`): all 135 PDFs are searched. A result counts only if both the document and the page match the annotation.
- **Single document** (`document_*`): the search is restricted to the PDF the question was written for.

Three metrics are reported over the 837 questions:

- **hit@5**: share of questions with at least one annotated evidence page among the five pages.
- **recall@5**: fraction of a question's annotated pages among the five, averaged over questions. 35 questions have more than five evidence pages, so the maximum reachable value is 98.4%.
- **MRR@5**: reciprocal rank of the first annotated page, averaged over questions.

## Runs

| Report | Index text | Embedding model | Pages with text | Chunks |
|---|---|---|---:|---:|
| [`corpus-audit-v1`](../reports/corpus-audit-v1/) | Text embedded in the PDFs | `BAAI/bge-small-en-v1.5` | 4,911 of 6,529 | 12,920 |
| [`corpus-audit-ocr-v1`](../reports/corpus-audit-ocr-v1/) | As above, plus local RapidOCR on the 1,618 pages with no embedded text | `BAAI/bge-small-en-v1.5` | 6,510 of 6,529 | 14,756 |
| [`corpus-audit-openai-v1`](../reports/corpus-audit-openai-v1/) | Same chunks as the OCR run | `openai/text-embedding-3-small` via OpenRouter | 6,510 of 6,529 | 14,756 |

OCR pages were selected by missing text only, without reference to the questions or annotations. Pages are rendered at 150 DPI, capped at 2,400 pixels on the long side. The second and third runs share identical chunks, so their BM25 results are identical and the difference between them is the embedding model alone.

## Results

All values are over the same 837 questions.

| Scope and method | Embedded text, BGE-small | With OCR, BGE-small | With OCR, text-embedding-3-small |
|---|---:|---:|---:|
| **Whole library, hybrid hit@5** | **41.9%** (351) | **55.8%** (467) | **60.9%** (510) |
| Whole library, hybrid recall@5 | 36.0% | 48.5% | 53.7% |
| Whole library, hybrid MRR@5 | 0.326 | 0.451 | 0.478 |
| Whole library, BM25 hit@5 | 41.1% (344) | 54.7% (458) | 54.7% (458) |
| Whole library, dense hit@5 | 38.0% (318) | 50.9% (426) | 56.3% (471) |
| Single document, hybrid hit@5 | 58.8% (492) | 76.3% (639) | 76.0% (636) |
| Single document, hybrid recall@5 | 49.2% | 64.4% | 66.1% |
| Single document, BM25 hit@5 | 57.6% (482) | 74.9% (627) | 74.9% (627) |
| Single document, dense hit@5 | 56.5% (473) | 73.4% (614) | 74.4% (623) |

Paired changes in whole-library hybrid hit@5:

- Adding OCR text: 121 questions gained a hit and 5 lost one.
- Changing the embedding model on the OCR index: 53 gained a hit and 10 lost one.

Before OCR, 202 of the 837 questions had no indexed text on any of their evidence pages; after OCR, 1 did.

## Report files

Each report folder contains:

- `summary.json`: counts, index and model identifiers, metric definitions, the six scope/method results, and a breakdown by annotated evidence type (chart, figure, table, layout, plain text).
- `documents.csv`: one row per PDF with its page count, pages with indexed text, question counts and per-document metrics.
- `cases.jsonl`: one line per eligible question with the document, annotated evidence pages, and for each scope/method the five retrieved pages and the resulting recall, hit and MRR. `source_row` is the question's zero-based position in the benchmark's `data/samples.json`.

## Checking the figures

```powershell
.\.venv\Scripts\python.exe scripts/verify_reported_metrics.py
```

This needs only the repository. It re-scores every question from its retrieved pages and annotated pages, recomputes each aggregate, recounts documents, pages and questions, and exits with an error if anything differs from the `summary.json` files.

## Re-running the evaluation

The reports were produced by the commands below. The audit scripts do not overwrite existing output, so move the published `reports/corpus-audit-*` folders aside first.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,retrieval,workspace]"
.\.venv\Scripts\python.exe -m evidence_lab download
.\.venv\Scripts\python.exe -m evidence_lab index
.\.venv\Scripts\python.exe -m evidence_lab audit
.\.venv\Scripts\python.exe scripts/audit_corpus_accuracy.py

.\.venv\Scripts\python.exe scripts/run_ocr_corpus.py --workers 4
.\.venv\Scripts\python.exe -m evidence_lab.ocr_index
.\.venv\Scripts\python.exe scripts/audit_corpus_accuracy.py --index-variant ocr

.\.venv\Scripts\python.exe scripts/run_openai_corpus.py --preflight
.\.venv\Scripts\python.exe scripts/run_openai_corpus.py --execute
```

The download is about 636 MiB. The OCR step needs the `ocr` extra and processes 1,618 pages on CPU. The last command needs `OPENROUTER_API_KEY` in a local `.env` and makes paid embedding requests under a $0.10 cap; the published run reserved $0.06.

## Scope of these results

- The figures measure whether an annotated evidence page is retrieved. They do not measure the accuracy of generated answers or of OCR transcription.
- The published `cases.jsonl` files omit question text, and the PDFs and indexes are not in this repository, because both belong to the benchmark and its original publishers. `verify_reported_metrics.py` therefore confirms that the reported figures follow from the saved per-question results; confirming the retrieved pages themselves requires downloading the benchmark and re-running the commands above.
- MMLongBench-Doc is a public benchmark. The figures describe this corpus and are not an estimate for other document collections.
- BGE-small truncates inputs at 512 tokens; some 180-word chunks exceed that.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The suite has 221 tests. They use synthetic fixtures and run without the benchmark, model downloads or network access.
