# Diabetes Research AI — HybridRAG local chat

Flask chat interface for diabetes research. It reuses the existing `HybridRAG.search` pipeline (BM25, vector search, Neo4j graph search and RRF) and can call a local Ollama model. Conversation messages are stored in SQLite; retrieval metadata remains private and is not returned by chat/history APIs.

## Requirements

- Windows and Python 3.13 or a compatible Python version.
- Ollama running locally, with `qwen2.5:1.5b` already available.
- Existing Neo4j database and the configured `.env` values used by HybridRAG.
- Existing diabetes chunks, normalized entities and vector file in `data/`.

The application does not crawl, re-chunk, re-embed, import graph data, or download Ollama models.

## Configuration

Copy `.env.example` to `.env` only if you do not already have a `.env`. If one exists, preserve its Neo4j values and add or update only the `OLLAMA_*` settings:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:1.5b
OLLAMA_TIMEOUT=120
OLLAMA_NUM_CTX=4096
OLLAMA_KEEP_ALIVE=10m
OLLAMA_TEMPERATURE=0.2
OLLAMA_TOP_P=0.85
OLLAMA_REPEAT_PENALTY=1.15
OLLAMA_REPEAT_LAST_N=128
  OLLAMA_NUM_PREDICT=96
```

`CHAT_DATABASE` is optional and defaults to `data/chat_history.sqlite3`. Keep `.env` local; do not commit Neo4j credentials.

## Install Python dependencies

Activate the existing virtual environment if the project has one. To create one from scratch, run these commands from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The integration uses `requests`, which is already a project dependency. SQLite and SSE use Python/Flask standard capabilities; no agent framework is required.


The Ollama client checks `done` and `done_reason`, logs request and generation timing/token counters, and rejects repetitive, non-Vietnamese or incomplete outputs before they can be shown. Rejected output uses question-specific extractive answers only when retrieved evidence supports them; otherwise the system abstains. It does not make an extra retry call. Retrieval context sent to agents is capped at 5,200 characters and chat retrieval uses up to eight ranked, article-diversified chunks; the BM25/vector/graph/RRF algorithms and source data remain intact.

## Start local services

1. Start Ollama and check the model. If `ollama` is not in PATH, use its usual Windows install location:

   ```powershell
   $ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
   if (-not $ollama) { $ollama = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe' }
   & $ollama list
   ```

   Confirm `qwen2.5:1.5b` appears in the list. The model is configured to run locally. The application does not call paid or cloud LLM APIs or download a model.

2. Start Neo4j Desktop and ensure the existing database is running. No import script is needed for normal chat.

3. Start Flask:

   ```powershell
   python app.py
   ```

4. Open `http://127.0.0.1:5035` in a browser.

Flask does not contact Ollama or initialize the embedding model at import/startup. The existing HybridRAG instance is initialized lazily on the first in-scope retrieval request. Hugging Face is set to offline mode for that initialization so query embeddings must use the already cached embedding model; no embedding weights are fetched. Chat can still start if Ollama is offline; the request then uses the existing extractive fallback when retrieved evidence supports it.

The right side of the chat includes a dataset-based risk screening form. It trains a local 1D CNN from `data/diabetes_prediction_dataset.csv` when no saved model exists, then caches the CNN and feature transformer under `data/` for reuse. The UI shows either “Nguy cơ cao” or “Không thuộc nhóm nguy cơ cao”; the latter does not mean zero risk. This screening result is not a diagnosis or an individually calibrated probability. If HbA1c is at least 6.5%, or fasting glucose is at least 126 mg/dL, the result is marked high and advises clinical confirmation. No entered values are stored in conversation history.

## API

- `GET /api/health`: Flask and Ollama status.
- `POST /api/conversations`: create a conversation.
- `GET /api/conversations`: list conversations.
- `GET /api/conversations/<id>`: return user-facing messages, without internal retrieval metadata.
- `DELETE /api/conversations/<id>`: delete the conversation, messages and their private metadata.
- `POST /api/chat`: SSE response with genuine Supervisor/retrieval/agent status events and one final answer event.
- `POST /api/predict-risk`: local dataset-based estimate from the left-side form; does not call Ollama or store submitted values.
- `POST /api/ask`: retained backward-compatible HybridRAG endpoint.

The chat API returns `answer`, `agents_used`, and `status`. Source IDs, URLs, retrieved chunks, internal agent results and provider metadata are saved separately in SQLite and omitted from ordinary history responses.

## Agent behavior

The Supervisor routes with lightweight rules. Medical, Nutrition and Knowledge agents share one retrieval result and make local Ollama calls sequentially. Data Analysis only processes a user-supplied numeric table and computes basic descriptive statistics. The Supervisor combines agent responses in Python; it does not spend an extra model call on routine synthesis. A single-specialist request uses at most one normal Qwen call; a combined request uses at most one per specialist (currently at most three agents). If a model response contains CJK text or internal source markers, it is rejected/sanitized and the retrieval-based fallback is used where possible.

## Checks

Run the isolated integration tests without Neo4j or Ollama calls:

```powershell
python -m unittest discover -s tests -v
```

The legacy `/api/ask` endpoint and retrieval/evaluation scripts remain available.
