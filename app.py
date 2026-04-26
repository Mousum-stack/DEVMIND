import time
import os
import re
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langfuse.decorators import observe, langfuse_context
from langfuse import Langfuse

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
)

app = FastAPI()
TOP_K_FINAL = 5

SYSTEM_PROMPT = """You are DevMind, an expert code assistant.

IMPORTANT RULES:
- ALWAYS answer in English only.
- Use ONLY the provided context to answer.
- Cite the source file after every fact using [filename].
- Format your answer in clear bullet points or numbered lists.
- Use markdown headers (##) to organize long answers.
- Format code in markdown code blocks.
- Keep answers concise and structured like a professional assistant.
- If context does not contain the answer say: I don't have enough context.

Context:
{context}
"""

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
retrievers = {}
indexing_status = {}

# ── Intent Detection ──────────────────────────────────────────────────────────

GREETINGS = ["hi", "hello", "hey", "hii", "howdy", "sup",
             "what's up", "good morning", "good evening", "good afternoon"]

def detect_intent(question: str) -> str:
    q = question.lower().strip()
    if any(q == g or q.startswith(g + " ") or q.startswith(g + "!") for g in GREETINGS):
        return "greeting"
    if len(q.split()) <= 3 and not any(c in q for c in ["(", ".", "_", "/"]):
        return "general"
    return "rag"

# ── Retriever ─────────────────────────────────────────────────────────────────

def load_retriever(project_name: str, db_path: str):
    if not Path(db_path).exists():
        return None
    print(f"Loading {project_name}...")
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name="devmind_code",
    )
    bm25_docs = []
    offset = 0
    while True:
        batch = vectorstore.get(limit=5000, offset=offset)
        if not batch["documents"]:
            break
        for text, meta in zip(batch["documents"], batch["metadatas"]):
            bm25_docs.append(Document(page_content=text, metadata=meta))
        offset += 5000
    if not bm25_docs:
        return None
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = 10
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    print(f"Loaded {project_name} — {len(bm25_docs)} chunks")
    return ensemble


def index_repo(github_url: str, project_name: str):
    try:
        indexing_status[project_name] = "cloning"
        clone_path = f"./data/{project_name}_source"
        db_path    = f"./data/{project_name}_db"
        if not Path(clone_path).exists():
            subprocess.run(
                ["git", "clone", "--depth=1", github_url, clone_path],
                check=True, capture_output=True
            )
        indexing_status[project_name] = "indexing"
        repo = Path(clone_path)
        docs = []
        for ext in [".py", ".md"]:
            for f in repo.rglob(f"*{ext}"):
                try:
                    loader = TextLoader(str(f), encoding="utf-8")
                    d = loader.load()
                    for doc in d:
                        doc.metadata["source_file"] = str(f.relative_to(repo))
                        doc.metadata["project"]     = project_name
                    docs.extend(d)
                except:
                    pass
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks   = splitter.split_documents(docs)
        Chroma.from_documents(
            chunks, embeddings,
            persist_directory=db_path,
            collection_name="devmind_code"
        )
        r = load_retriever(project_name, db_path)
        if r:
            retrievers[project_name] = r
        indexing_status[project_name] = "ready"
        print(f"Done {project_name} — {len(chunks)} chunks")
    except Exception as e:
        indexing_status[project_name] = f"error: {str(e)}"
        print(f"Error: {e}")


print("Loading existing projects...")
data_dir = Path("./data")
for item in data_dir.iterdir():
    if item.is_dir() and item.name.endswith("_db"):
        name = item.name.replace("_db", "")
        r    = load_retriever(name, str(item))
        if r:
            retrievers[name]      = r
            indexing_status[name] = "ready"
print(f"Loaded: {list(retrievers.keys())}")


# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    project:  str = "chroma"

class AddRepoRequest(BaseModel):
    github_url:   str
    project_name: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def serve_ui():
    return FileResponse("api/index.html")


@app.get("/projects")
def get_projects():
    return {
        "projects": [
            {"name": k, "status": indexing_status.get(k, "ready")}
            for k in retrievers.keys()
        ]
    }


@app.get("/status/{project_name}")
def get_status(project_name: str):
    return {
        "project": project_name,
        "status":  indexing_status.get(project_name, "not_found")
    }


@app.post("/add-codebase")
def add_codebase(req: AddRepoRequest, background_tasks: BackgroundTasks):
    name = req.project_name.lower().replace(" ", "_")
    if name in retrievers:
        return {"message": f"{name} already exists", "project": name}
    indexing_status[name] = "queued"
    background_tasks.add_task(index_repo, req.github_url, name)
    return {"message": f"Indexing {name} started", "project": name}


@app.post("/query")
@observe(name="devmind-query")
def query(req: QueryRequest):
    question = req.question.strip()
    project  = req.project
    intent   = detect_intent(question)

    langfuse_context.update_current_trace(
        metadata={"project": project, "intent": intent}
    )

    # 1. Greeting → direct LLM
    if intent == "greeting":
        llm    = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
        answer = llm.invoke(
        "You are DevMind, a code assistant. "
        "The user said: " + question +
        ". Reply in 1-2 short friendly sentences only. Ask what codebase they need help with."
        ).content
        langfuse.flush()
        return {
            "answer":  answer,
            "sources": [],
            "metrics": {"mode": "greeting", "retrieval_time": 0,
                       "generation_time": 0, "total_time": 0, "chunks_used": 0}
        }

    # 2. General → LLM fallback
    if intent == "general":
        t0     = time.time()
        llm    = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        answer = llm.invoke(question).content
        total  = round(time.time() - t0, 2)
        langfuse.flush()
        return {
            "answer":  answer,
            "sources": ["general_knowledge"],
            "metrics": {"mode": "fallback", "retrieval_time": 0,
                       "generation_time": total, "total_time": total, "chunks_used": 0}
        }

    # 3. RAG → full pipeline
    if project not in retrievers:
        return {
            "answer":  f"Project '{project}' not found or still indexing.",
            "sources": [],
            "metrics": {"mode": "error", "retrieval_time": 0,
                       "generation_time": 0, "total_time": 0, "chunks_used": 0}
        }

    t0   = time.time()
    docs = retrievers[project].invoke(question)[:TOP_K_FINAL]
    retrieval_time = round(time.time() - t0, 2)

    cited_files   = []
    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        if source not in cited_files:
            cited_files.append(source)
        context_parts.append(f"[{source}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    t1      = time.time()
    llm     = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    full_prompt = SYSTEM_PROMPT.format(context=context) + f"\nQuestion: {question}\nAnswer:"
    answer  = llm.invoke(full_prompt).content
    generation_time = round(time.time() - t1, 2)

    langfuse.flush()

    return {
        "answer":  answer,
        "sources": cited_files,
        "metrics": {
            "mode":            "rag",
            "retrieval_time":  retrieval_time,
            "generation_time": generation_time,
            "total_time":      round(retrieval_time + generation_time, 2),
            "chunks_used":     len(docs)
        }
    }


app.mount("/static", StaticFiles(directory="api"), name="static")