from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import time
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI(title="DevMind API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_PATH = "./data/chroma_db"
TOP_K = 5

SYSTEM_PROMPT = """You are DevMind, an expert code assistant.

FORMAT YOUR RESPONSE WITH:
- Use **bold** for key concepts
- Use numbered lists (1. 2. 3.)
- Use dashes for bullet points
- Add ### Headings for sections
- Use ```python code blocks

Use ONLY the provided context. If insufficient, say "I don't have enough context."

Context:
{context}
"""

class QueryRequest(BaseModel):
    question: str
    model: str = "llama-3.3-70b-versatile"

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    metrics: dict
    success: bool

retriever = None
chunk_count = 0

def load_retriever():
    global chunk_count
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name="devmind_code",
        )
        chunk_count = vectorstore._collection.count()
        print(f"✅ Retriever loaded with {chunk_count} chunks")
        return vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    global retriever
    print("🚀 Starting DevMind API...")
    retriever = load_retriever()

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not loaded")
    
    try:
        retrieval_start = time.time()
        docs = retriever.invoke(request.question)
        retrieval_time = time.time() - retrieval_start

        sources = []
        context_parts = []
        for doc in docs:
            source_file = doc.metadata.get("source_file", "unknown")
            if source_file not in sources:
                sources.append(source_file)
            context_parts.append(f"[{source_file}]\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)

        generation_start = time.time()
        llm = ChatGroq(model=request.model, temperature=0)
        prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        chain = prompt | llm

        result = chain.invoke({
            "question": request.question,
            "context": context
        })

        generation_time = time.time() - generation_start
        total_time = retrieval_time + generation_time

        return QueryResponse(
            question=request.question,
            answer=result.content,
            sources=sources,
            metrics={
                "retrieval_time": round(retrieval_time, 2),
                "generation_time": round(generation_time, 2),
                "total_time": round(total_time, 2),
                "chunks_used": len(docs),
                "model": request.model
            },
            success=True
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy" if retriever else "degraded",
        "chunks_indexed": chunk_count,
    }

@app.get("/")
async def root():
    return FileResponse("public/index.html")

app.mount("/static", StaticFiles(directory="public"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
