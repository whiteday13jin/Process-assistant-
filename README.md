# Process Decision Assistant

Process Decision Assistant is a process-support system for PCB/FPCB scenarios.  
It combines structured anomaly diagnosis, Excel-based bottleneck analysis, and document Q&A with dual RAG pipelines.

![Role Workspace](docs/screenshots/02-role.png)

## Current Snapshot

1. Web workflow is available end-to-end with login, role hub, diagnosis, optimization, RAG Q&A, and report history.
2. UI pages have been aligned to one industrial backend style baseline (MES/QMS-oriented visual language).
3. The backend keeps rule/knowledge-driven decision logic as the primary path, with LLM usage limited to RAG-related tasks.
4. Reports are persisted locally under `reports/` and can be reviewed from the history pages.

## Core Capabilities

1. Structured diagnosis  
   Rule matching, case similarity, and feedback-weighted ranking for traceable troubleshooting suggestions.
2. Process optimization  
   Excel ingestion and bottleneck detection based on cycle time, defect rate, capacity, and takt-related heuristics.
3. Process knowledge Q&A  
   Two implementations are maintained in parallel:
   1. Custom RAG pipeline (main path).
   2. LangChain demo pipeline (comparison baseline).

## Quick Start

```bash
pip install -r requirements.txt
copy .env.example .env
python web_ui.py
```

Open [http://127.0.0.1:5050](http://127.0.0.1:5050).

Default login:

1. Username: `user`
2. Password: `123`

For RAG features, configure these variables in `.env`:

1. `PROCESS_ASSISTANT_API_KEY`
2. `PROCESS_ASSISTANT_BASE_URL`
3. `PROCESS_ASSISTANT_CHAT_MODEL`
4. `PROCESS_ASSISTANT_EMBED_MODEL`

Diagnosis and optimization modules can still run without model credentials.

## CLI Examples

```bash
python -m process_assistant.cli diagnose --input examples/diagnosis_request.json
python -m process_assistant.cli optimize --excel reference/your_file.xlsx
python -m process_assistant.cli rag-build --docs-dir data/rag_docs --index-dir data/rag_index
python -m process_assistant.cli rag-ask --index-dir data/rag_index --question "What should I check first after weak soldering?"
python -m process_assistant.cli rag-build-lc --docs-dir data/rag_docs --index-dir data/rag_index_langchain
python -m process_assistant.cli rag-ask-lc --index-dir data/rag_index_langchain --question "What factors usually cause lamination misalignment?"
```

Run `python -m process_assistant.cli --help` for the full command list.

## Repository Layout

1. `web_ui.py`: Flask entry and web routes.
2. `process_assistant/`: diagnosis, optimization, RAG, indexing, and CLI modules.
3. `templates/` and `static/`: web pages and shared theme assets.
4. `data/`: local knowledge data and RAG documents/indexes.
5. `examples/`: sample request and evaluation payloads.
6. `reports/`: generated diagnosis, optimization, and Q&A outputs.

