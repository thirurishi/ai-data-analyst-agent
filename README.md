# AI Data Analyst Agent

A learning-focused portfolio project for exploring how an AI-powered data analyst agent can support data analysis workflows, reporting, and insight generation.

## Main Goal

Build a progressively structured project that teaches the design, planning, and implementation of an AI data analyst agent, starting with the project foundation and evolving toward more advanced capabilities.

## Planned Capabilities

- Natural-language data exploration
- Data cleaning and validation workflows
- Summary and trend analysis
- Visualization support
- Insight and recommendation generation
- Performance reporting and stakeholder-friendly outputs

## Persistence

| Data | Where it is stored | Survives an API restart |
|---|---|---|
| Uploaded files | `backend/app/uploads/` | Yes |
| Upload records (`file_id`) | `backend/app/uploads/.upload_registry.json` | Yes, if the file still exists unchanged |
| SQL ingestion registry | `backend/app/uploads/.database_ingestion_registry.json` | Yes, after the table is re-verified in the database |
| Ingested SQL tables | PostgreSQL | Yes |
| RAG index, `EMBEDDING_VECTOR_STORE=postgres` | PostgreSQL with pgvector | Yes |
| RAG index, `EMBEDDING_VECTOR_STORE=memory` | API process memory | **No.** Local development only; re-index after each restart. |

- Registries are reloaded at startup. Entries whose file is missing or changed are skipped, and an
  ingested table is only reused after its existence, column count, and row count are checked.
- Re-indexing a document replaces its chunks for the current embedding model. Changing
  `EMBEDDING_MODEL` requires re-indexing; a PostgreSQL table created with different
  `EMBEDDING_DIMENSIONS` is reported as a configuration error.
- The registry files support a single API process. Do not run multiple uvicorn workers against
  the same upload directory.

## Project Status

The project is currently in the foundation/setup stage.
