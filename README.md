# AI Data Analyst Agent

Upload a CSV and get what a junior analyst would hand you on day one: key metrics, a data-health score, the patterns that stand out, ready-made charts, and answers to plain-English questions. When you're done, export a PDF report. You can also upload PDF, Word and text documents and ask questions about them with retrieval-augmented generation (RAG).

The core rule of the project: **the LLM never invents numbers.** It only chooses *which* validated analysis tool to run. Every figure comes from a deterministic Pandas or SQL calculation, and you can inspect the raw result under each answer.

![Overview](docs/screenshots/overview.png)

| Ask in plain English | Human-approved data cleaning |
|---|---|
| ![Ask](docs/screenshots/ask.png) | ![Quality](docs/screenshots/quality.png) |

## Features

- **Instant overview.** Detects column roles (metric, category, date, identifier). Shows KPIs and findings such as the top segment, concentration, trend, distribution shape, correlations, missing data and outliers. Suggests charts automatically.
- **Data health score (0–100).** Built from completeness, uniqueness, validity and consistency, plus a penalty for each detected issue.
- **Ask questions.**
  - With an OpenRouter key, an LLM picks a tool (`analyze_group_by`, `sql_top_n`, `rag_ask`…) with schema-validated arguments, and a second call writes an answer grounded in the verified result.
  - Without a key, a built-in rule-based planner answers questions such as "top 5 products by revenue", "average rating by category" or "how many orders per channel".
  - If the LLM fails, the app falls back to the planner automatically.
- **Chart builder.** Bar, line, doughnut, pie, histogram or scatter, for any column, with sum, average, median, min, max or count.
- **Data quality workflow.** Issues are detected, recommendations generated, and a human approves or rejects each one. Approved fixes are applied to a *new* cleaned file, which opens automatically. The original upload is never modified.
- **Report export.** A styled PDF (KPIs, health, findings, charts, recommendations, column overview) or a Markdown report.
- **Document Q&A (RAG).** Text extraction from PDF, DOCX and TXT, deterministic chunking, embeddings, vector search (in memory or PostgreSQL with pgvector), and answers with citations.
- **SQL backend.** CSVs can be ingested into PostgreSQL, and the agent chooses between Pandas and SQL tools.
- **Safety.** Filename and path-traversal protection, size limits, content-type checks, UUID file IDs with content fingerprints, read-only analysis, and standardized error codes.

## Quick start

Requires Python 3.11+.

**Windows (PowerShell)**

```powershell
.\run.ps1
```

**macOS / Linux**

```bash
./run.sh
```

Open **http://127.0.0.1:8000**, then click **Try the sample retail dataset** or drop in your own CSV. Interactive API docs are at http://127.0.0.1:8000/docs.

To run it manually:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

### Enable the LLM features (optional)

```bash
cp .env.example .env
```

Then set these values in `.env`:

| Variable | Enables |
|---|---|
| `OPENROUTER_API_KEY` | LLM tool-calling agent and grounded answers (default model `openai/gpt-4o-mini`) |
| `EMBEDDING_API_KEY` | Document indexing and Q&A (OpenAI-compatible embeddings) |
| `POSTGRES_*` | SQL ingestion and analysis, plus a durable pgvector RAG index |

The sidebar shows which mode is active.

### Docker (API + web app + PostgreSQL/pgvector)

```bash
docker compose up --build
```

## Architecture

```
frontend/ (vanilla JS + Chart.js, served by FastAPI at /app)
        │  fetch
        ▼
FastAPI  backend/app/main.py
 ├─ api/routes/        upload · metadata · profile · data-quality · analyze · agent
 │                     documents (RAG) · database · insights (overview, charts, report, assistant)
 ├─ ingestion/         safe uploads, file registry, robust CSV reader, profiler
 ├─ quality/           detection → recommendations → approval → cleaning
 ├─ analysis/          deterministic Pandas + SQL analysis services
 ├─ insights/          column roles, health score, findings, charts, offline planner, PDF/MD report
 ├─ tools/             tool registry, schemas, LLM tool selection with validation
 ├─ llm/               OpenRouter client, planner, grounded answer generation
 ├─ services/          document extraction, chunking, embeddings, retrieval, QA
 └─ repositories/      vector stores (memory / pgvector)
```

The question-answering flow looks like this:

```
question ─► LLM picks tool + args ─► args validated against real columns ─► Pandas/SQL runs
         └─(no key / failure)─► rule-based planner ─────────────────────────┘
                                                          result ─► grounded answer + chart
```

## Main endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/upload` | Upload CSV, PDF, DOCX or TXT (≤ 10 MB) and receive a `file_id` |
| GET | `/files` | List uploaded files |
| GET | `/insights?file_id=` | KPIs, health score, findings, charts, column summary, preview |
| POST | `/charts` | Build a chart spec from any columns |
| POST | `/assistant/ask` | Ask a question (LLM agent with offline fallback) |
| GET | `/report?file_id=&format=pdf\|md` | Download the analysis report |
| GET | `/data-quality/recommendations?file_id=` | Quality recommendations |
| POST | `/data-quality/recommendations/{id}/approve` · `/reject` | Human review |
| POST | `/data-quality/apply` | Apply an approved fix and register the cleaned file |
| POST | `/documents/index/{file_id}` · `/documents/ask` | RAG indexing and Q&A |
| POST | `/database/ingest/{file_id}` · `/database/analyze` | PostgreSQL ingestion and SQL analysis |
| GET | `/status` | Which AI features are configured |

## Tests

```bash
python -m pytest -q
```

The suite has 396 tests. They run against an isolated temporary upload directory and need no API keys or database; the LLM and embedding providers are mocked.

## Persistence

| Data | Where it is stored | Survives an API restart |
|---|---|---|
| Uploaded files | `backend/app/uploads/` | Yes |
| Upload records (`file_id`) | `backend/app/uploads/.upload_registry.json` | Yes, if the file still exists unchanged |
| SQL ingestion registry | `backend/app/uploads/.database_ingestion_registry.json` | Yes, after the table is re-verified in the database |
| Ingested SQL tables | PostgreSQL | Yes |
| RAG index, `EMBEDDING_VECTOR_STORE=postgres` | PostgreSQL with pgvector | Yes |
| RAG index, `EMBEDDING_VECTOR_STORE=memory` | API process memory | **No.** Re-index after each restart. |

Quality approval status is held in memory and resets on restart. The registry files support a single API process, so do not run multiple uvicorn workers against the same upload directory.

## Tech stack

Python · FastAPI · Pandas · SQLAlchemy · PostgreSQL + pgvector · OpenRouter (LLM) · OpenAI-compatible embeddings · ReportLab · Chart.js · pytest · Docker

## Author

Thirumurugan Kumar, MSc Data Science & Analytics, EPITA Paris.
