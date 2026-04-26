"""
DevMind - Phase 1: RAG Query
File: rag/query.py

What this does:
  1. Takes a user question
  2. Retrieves relevant chunks using HYBRID search (vector + BM25)
  3. Reranks results with Cohere cross-encoder
  4. Forces the LLM to cite file sources in its answer
  5. Returns answer + list of cited files

Usage:
  python rag/query.py
  python rag/query.py --question "How does FastAPI handle dependency injection?"
"""

import os
import argparse
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH  = "./data/chroma_db"
EMBED_MODEL  = "text-embedding-3-small"
LLM_MODEL    = "gpt-4o-mini"          # cheap + fast for dev
TOP_K_VECTOR = 10                      # how many vector results to fetch
TOP_K_BM25   = 10                      # how many BM25 results to fetch
TOP_K_RERANK = 5                       # how many to keep after reranking
# ─────────────────────────────────────────────────────────────────────────────


# ── Prompt with citation enforcement ─────────────────────────────────────────
SYSTEM_PROMPT = """You are DevMind, an expert code assistant.

Answer the user's question using ONLY the provided context chunks.
You MUST cite your sources. After every claim, add [filename] in brackets.

Rules:
- If the context does not contain the answer, say "I don't have enough context to answer this."
- Always cite the source file for every fact you state.
- Be concise but complete.
- Format code in markdown code blocks.

Context:
{context}
"""

HUMAN_PROMPT = "{question}"
# ─────────────────────────────────────────────────────────────────────────────


def load_vectorstore() -> Chroma:
    """Load the existing ChromaDB from disk."""
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="devmind_code",
    )
    count = vectorstore._collection.count()
    print(f"  Loaded vector store: {count} chunks indexed")
    return vectorstore


def build_hybrid_retriever(vectorstore: Chroma) -> ContextualCompressionRetriever:
    """
    Build a hybrid retriever:
      - Vector search: finds semantically similar chunks
      - BM25: finds chunks with exact keyword matches
      - Ensemble: combines both (0.5/0.5 weight)
      - Cohere Rerank: re-scores the combined results
    
    Why hybrid? Vector search is great at meaning, bad at exact terms.
    BM25 is great at exact terms, bad at meaning. Together they beat either alone.
    """

    # 1. Vector retriever
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_VECTOR}
    )

    # 2. BM25 retriever — needs all docs loaded into memory
    #    (fine for a single repo; use Elasticsearch for production scale)
    print("  Loading docs for BM25...")
    all_docs = vectorstore.get()
    bm25_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = TOP_K_BM25

    # 3. Ensemble: 50% vector + 50% BM25
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    # 4. Cohere reranker on top
    #    Cross-encoder scores each (question, chunk) pair together
    #    — much more accurate than embedding similarity alone
    cohere_reranker = CohereRerank(
        model="rerank-english-v3.0",
        top_n=TOP_K_RERANK
    )

    hybrid_retriever = ContextualCompressionRetriever(
        base_compressor=cohere_reranker,
        base_retriever=ensemble_retriever
    )

    return hybrid_retriever


def format_context(docs: list[Document]) -> tuple[str, list[str]]:
    """Format retrieved docs into a context string + list of cited files."""
    context_parts = []
    cited_files = []

    for i, doc in enumerate(docs):
        source = doc.metadata.get("source_file", "unknown")
        if source not in cited_files:
            cited_files.append(source)

        context_parts.append(
            f"[{source}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts), cited_files


def ask(question: str, retriever, llm) -> dict:
    """Run the full RAG pipeline for one question."""
    print(f"\n  Retrieving context...")
    docs = retriever.invoke(question)
    print(f"  Got {len(docs)} chunks after reranking")

    context, cited_files = format_context(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  HUMAN_PROMPT),
    ])

    chain = prompt | llm

    print("  Generating answer...")
    response = chain.invoke({
        "context":  context,
        "question": question,
    })

    return {
        "question":    question,
        "answer":      response.content,
        "cited_files": cited_files,
        "num_chunks":  len(docs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question", "-q",
        type=str,
        default="How does FastAPI handle dependency injection?",
        help="Question to ask DevMind"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  DevMind — Phase 1: Query Pipeline")
    print("=" * 55)

    print("\n[1/3] Loading vector store...")
    vectorstore = load_vectorstore()

    print("\n[2/3] Building hybrid retriever...")
    retriever = build_hybrid_retriever(vectorstore)

    print("\n[3/3] Running RAG pipeline...")
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    result = ask(args.question, retriever, llm)

    print("\n" + "=" * 55)
    print(f"  QUESTION: {result['question']}")
    print("=" * 55)
    print(f"\n{result['answer']}")
    print("\n" + "-" * 55)
    print(f"  Sources used ({result['num_chunks']} chunks):")
    for f in result["cited_files"]:
        print(f"    - {f}")
    print("-" * 55 + "\n")


if __name__ == "__main__":
    main()
