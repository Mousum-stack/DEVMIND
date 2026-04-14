# DevMind — AI Code Assistant

A production-grade AI code assistant built across 5 phases.
Ask questions about any codebase. Get cited, accurate answers.

## What it does (final state)
- Indexes a GitHub repo and answers questions with file citations
- Runs locally on your hardware (no API costs at inference time)
- Traces every request: latency, cost, quality score
- Uses a fine-tuned model trained on coding tasks
- Accepts voice input and streams spoken answers

---

## Project Structure

```
devmind/
├── rag/              # Phase 1 — RAG pipeline
│   ├── ingest.py     # Index a repo into ChromaDB
│   ├── query.py      # Hybrid search + reranking + cited answers
│   └── evaluate.py   # Ragas evaluation suite
├── models/           # Phase 2 — Local model benchmarking
├── monitoring/       # Phase 3 — Langfuse tracing + CI evals
├── finetuning/       # Phase 4 — QLoRA + DPO fine-tuning
├── voice/            # Phase 5 — Pipecat voice pipeline
├── data/             # Repo clones, vector DB, eval results
├── tests/            # Unit + integration tests
├── .env.example      # API key template
└── requirements.txt  # Python dependencies
```

---

## Phase 1 — RAG Core (Weeks 1–3)

### Setup

```bash
# 1. Clone this repo
git clone <your-repo> devmind && cd devmind

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and COHERE_API_KEY

# 5. Clone the target codebase to index
git clone https://github.com/tiangolo/fastapi ./data/fastapi
```

### Run

```bash
# Step 1: Index the repo (run once)
python rag/ingest.py

# Step 2: Ask a question
python rag/query.py --question "How does FastAPI handle dependency injection?"

# Step 3: Run the evaluation suite
python rag/evaluate.py
```

### Phase 1 Milestone
Ragas faithfulness score > **0.80** on the test question set.

---

## Evaluation Results

| Phase | Model | Faithfulness | Answer Relevancy | Latency |
|-------|-------|-------------|-----------------|---------|
| 1 — RAG | gpt-4o-mini | TBD | TBD | TBD |
| 2 — Local | CodeLlama | TBD | TBD | TBD |
| 2 — Local | DeepSeek-Coder | TBD | TBD | TBD |
| 4 — Fine-tuned | CodeLlama-7B-FT | TBD | TBD | TBD |

*(Fill in as you complete each phase)*

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| RAG | LangChain · ChromaDB · Cohere Rerank · Ragas |
| Local Inference | Ollama · CodeLlama · DeepSeek · Mistral |
| API | FastAPI · Instructor · Pydantic |
| Observability | Langfuse · GitHub Actions |
| Fine-tuning | Unsloth · QLoRA · DPO · HuggingFace |
| Voice | Pipecat · Deepgram · ElevenLabs |

---

## Cost Estimate (Phase 1)

| Item | Cost |
|------|------|
| Embedding FastAPI repo (~2k chunks) | ~$0.002 |
| Per question (retrieval + GPT-4o-mini) | ~$0.001 |
| Evaluation run (8 questions) | ~$0.01 |

Total for all of Phase 1: **under $1**
# DevMind Test
