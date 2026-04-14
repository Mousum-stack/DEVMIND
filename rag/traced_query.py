"""
DevMind - Phase 3: Traced Query
File: rag/traced_query.py

What this does:
  1. Same as query.py but with Langfuse tracing
  2. Every step gets tracked — retrieval time, LLM time, total time
  3. Results show up in Langfuse dashboard in real-time
  4. Track quality + performance over time

Run with:
  python rag/traced_query.py -q "Your question here"
"""

import argparse
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langfuse.decorators import observe
from langfuse import Langfuse

load_dotenv()

# ── Langfuse Setup ─────────────────────────────────────────────────────────
langfuse_client = Langfuse()

CHROMA_PATH  = "./data/chroma_db"
TOP_K_VECTOR = 10
TOP_K_BM25   = 10
TOP_K_FINAL  = 5

SYSTEM_PROMPT = """You are DevMind, an expert code assistant.
Answer the user question using ONLY the provided context chunks.
After every fact you state, add [filename] in brackets showing which file it came from.
If the context does not contain the answer, say "I don't have enough context."
Format code in markdown code blocks.

Context:
{context}
"""

# ─────────────────────────────────────────────────────────────────────────

def load_retriever():
    """Load hybrid retriever (same as query.py)."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="devmind_code",
    )
    count = vectorstore._collection.count()
    print(f"  Loaded vector store: {count} chunks indexed")

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_VECTOR}
    )

    print("  Loading BM25...")
    all_docs = vectorstore.get()
    bm25_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = TOP_K_BM25

    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble


@observe()
def trace_retrieval(question, retriever):
    """Trace retrieval step with Langfuse."""
    start = time.time()
    docs = retriever.invoke(question)
    docs = docs[:TOP_K_FINAL]
    retrieval_time = time.time() - start

    context_parts = []
    cited_files = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        if source not in cited_files:
            cited_files.append(source)
        context_parts.append(f"[{source}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)

    return {
        "context": context,
        "cited_files": cited_files,
        "num_chunks": len(docs),
        "retrieval_time": retrieval_time,
    }


@observe()
def trace_generation(question, context, llm):
    """Trace LLM generation step with Langfuse."""
    start = time.time()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    generation_time = time.time() - start

    return {
        "answer": response.content,
        "generation_time": generation_time,
    }


@observe()
def trace_ask(question, retriever, llm):
    """Main traced function — orchestrates retrieval + generation."""
    print(f"\n🔍 Retrieving context...")
    retrieval_result = trace_retrieval(question, retriever)

    print(f"🧠 Generating answer...")
    generation_result = trace_generation(
        question,
        retrieval_result["context"],
        llm
    )

    total_time = retrieval_result["retrieval_time"] + generation_result["generation_time"]

    return {
        "question": question,
        "answer": generation_result["answer"],
        "cited_files": retrieval_result["cited_files"],
        "retrieval_time": retrieval_result["retrieval_time"],
        "generation_time": generation_result["generation_time"],
        "total_time": total_time,
        "num_chunks": retrieval_result["num_chunks"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", "-q", type=str,
        default="How does FastAPI handle dependency injection?")
    parser.add_argument("--model", "-m", type=str,
        default="llama-3.3-70b-versatile",
        help="Groq model to use")
    args = parser.parse_args()

    print("=" * 70)
    print("  DevMind - Phase 3: Traced Query with Langfuse")
    print("=" * 70)

    print("\n[1/2] Loading retriever...")
    retriever = load_retriever()

    print(f"\n[2/2] Running traced RAG pipeline...")
    llm = ChatGroq(model=args.model, temperature=0)

    result = trace_ask(args.question, retriever, llm)

    print("\n" + "=" * 70)
    print("  RESULT")
    print("=" * 70)
    print(f"\n❓ QUESTION: {result['question']}\n")
    print(f"📝 ANSWER:\n{result['answer']}\n")
    print(f"📁 SOURCES:")
    for f in result["cited_files"]:
        print(f"   - {f}\n")

    print("=" * 70)
    print("  PERFORMANCE METRICS")
    print("=" * 70)
    print(f"  🔍 Retrieval time: {result['retrieval_time']:.2f}s")
    print(f"  🧠 Generation time: {result['generation_time']:.2f}s")
    print(f"  ⏱️  Total time:     {result['total_time']:.2f}s")
    print(f"  📊 Chunks used:    {result['num_chunks']}")
    print("\n  📊 View full trace at: https://us.cloud.langfuse.com")
    print("=" * 70)

    # Flush Langfuse to send all traces
    langfuse_client.flush()


if __name__ == "__main__":
    main()
