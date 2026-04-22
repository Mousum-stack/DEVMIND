# DevMind — AI Code Assistant

> Ask questions about any codebase in plain English. Get cited, accurate answers in seconds.

---

## What Is DevMind?

DevMind is a production-grade AI engineering project that turns any GitHub repository into a queryable knowledge base. Point it at a codebase, ask questions, and get answers grounded in the actual source files — with exact citations, latency metrics, and full observability.

Built entirely with a free stack. Zero API costs at inference time.

---

## Live Demo

```
How does FastAPI handle dependency injection?

DevMind: FastAPI handles dependency injection using the Depends() 
function [dependencies/index.md]. You declare dependencies as function 
parameters and FastAPI executes them before the route handler runs, 
injecting the results automatically [fastapi/routing.py].

retrieval 0.99s  |  generation 1.24s  |  total 2.23s  |  5 chunks
Sources: index.md  routing.py  param_functions.py  settings.md
```

---

## Features

- Ask any question about a codebase in plain English
- Hybrid search combining vector similarity and BM25 keyword matching
- Every answer cites exact source files
- Add any GitHub repo directly from the UI — cloned and indexed automatically
- Switch between multiple codebases from a dropdown
- Full observability — retrieval time, generation time, cost per request tracked in Langfuse
- CI/CD quality gate that blocks merges when Ragas faithfulness drops below 0.80
- Works completely offline using local models via Ollama
- 100% free stack — no paid APIs required

---

## Architecture

```
User question (web UI or voice)
        │
        ▼
┌─────────────────────────────────┐
│         Hybrid Retriever         │
│  BM25 keyword search (50%)      │
│  Vector similarity search (50%) │
│  ChromaDB — 17,326 chunks       │
└────────────────┬────────────────┘
                 │ top 5 chunks
                 ▼
┌─────────────────────────────────┐
│           LLM Layer              │
│  Cloud: Groq llama-3.1-8b       │
│  Local: Ollama tinyllama         │
│  Prompt: cite sources, grounded  │
└────────────────┬────────────────┘
                 │
                 ▼
        Answer + Citations
        + Performance Metrics
        + Langfuse Trace
```

---

## Evaluation Results

| Metric | Score | Target |
|--------|-------|--------|
| Faithfulness | 0.856 | > 0.80 ✅ |
| Answer Relevancy | 0.987 | > 0.80 ✅ |
| Avg Retrieval Time | 1.0s | — |
| Avg Generation Time | 1.3s | — |
| Total Cost | $0.00 | — |

---

## Local Model Benchmark (8GB RAM, no GPU)

| Model | Size | Avg Response Time | Verdict |
|-------|------|------------------|---------|
| tinyllama | 637MB | ~19s | Best for local use |
| phi3 | 2.2GB | ~260s | Too slow on CPU |
| qwen3.5 | 6.6GB | 40+ min | Not viable without GPU |

Conclusion: On 8GB RAM without a GPU, only sub-1GB quantized models are practical for local RAG inference.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Embeddings | HuggingFace all-MiniLM-L6-v2 (local, free) |
| Vector database | ChromaDB |
| Keyword search | BM25 |
| LLM — cloud | Groq API — llama-3.1-8b-instant (free tier) |
| LLM — local | Ollama — tinyllama, phi3, qwen3.5 |
| Web framework | FastAPI |
| Evaluation | Ragas |
| Observability | Langfuse |
| CI/CD | GitHub Actions |

---

## Project Structure

```
devmind/
├── rag/
│   ├── ingest.py          # Index a repo into ChromaDB
│   ├── query.py           # Hybrid search + cited answers
│   ├── evaluate.py        # Ragas evaluation suite
│   └── traced_query.py    # Langfuse traced pipeline
├── models/
│   ├── benchmark.py       # Benchmark 3 local models
│   └── local_query.py     # RAG pipeline using Ollama
├── api/
│   ├── main.py            # FastAPI backend + multi-codebase support
│   └── index.html         # Web UI with sidebar history
├── monitoring/
│   └── ci_eval.py         # CI quality gate script
├── .github/
│   └── workflows/
│       └── quality-gate.yml  # GitHub Actions eval on every PR
├── data/
│   └── eval_results.json  # Ragas scores history
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.9+
- Git
- Ollama (for local model mode)

### Installation

```bash
# Clone the repo
git clone https://github.com/Mousum-stack/DEVMIND.git
cd DEVMIND

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your keys
```

### Environment Variables

```bash
# Required
GROQ_API_KEY=gsk_...              # Free at console.groq.com

# Optional — for Langfuse observability
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com

# Recommended
TOKENIZERS_PARALLELISM=false
```

### Index a Codebase

```bash
# Clone the target repo
git clone https://github.com/tiangolo/fastapi ./data/fastapi

# Index it
python rag/ingest.py
```

### Run

```bash
# Start the web UI
uvicorn api.main:app --reload --port 8000

# Open browser
open http://127.0.0.1:8000
```

### Command Line Mode

```bash
# Ask a question
python rag/query.py -q "How does FastAPI handle dependency injection?"

# Local model mode (no internet)
python models/local_query.py -q "What is APIRouter?"

# Run evaluation
python rag/evaluate.py
```

---

## Adding a New Codebase

From the web UI click **+ Add codebase**, paste any GitHub URL and give it a name. DevMind clones and indexes it automatically in the background. When ready it appears in the project dropdown — no restart needed.

---

## CI/CD Quality Gate

Every pull request automatically runs the Ragas evaluation suite via GitHub Actions. If the faithfulness score drops below 0.80 the merge is blocked.

```yaml
# .github/workflows/quality-gate.yml
# Runs on every PR — blocks merge if quality regresses
```

To use this in your fork, add your `GROQ_API_KEY` as a GitHub Actions secret.

---

## What I Learned

This project covers the full spectrum of production AI engineering across 5 phases:

**Phase 1 — RAG core:** Chunking strategies, vector embeddings, hybrid search, citation enforcement, Ragas evaluation

**Phase 2 — Local inference:** Quantization (GGUF format), model benchmarking on real hardware, quality vs speed tradeoffs

**Phase 3 — Observability:** Distributed tracing with Langfuse, CI/CD quality gating, p50/p95 latency tracking

**Phase 4 — Multi-codebase:** Dynamic index management, background indexing, project routing architecture

**Phase 5 — Production UI:** FastAPI backend, real-time observability integration, collapsible history sidebar

---

## Skills Demonstrated

LLM Engineering · RAG Pipelines · Vector Databases · Hybrid Search · Local Inference · Model Benchmarking · MLOps · CI/CD · Observability · FastAPI · Production System Design

---

## Author

Mousum Rajgogoi — [GitHub](https://github.com/Mousum-stack)

Built as a learning project to understand production AI engineering from end to end.
