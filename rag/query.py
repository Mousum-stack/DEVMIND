import argparse
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

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

def load_retriever():
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

def ask(question, retriever, llm):
    print(f"\n  Retrieving context...")
    docs = retriever.invoke(question)
    docs = docs[:TOP_K_FINAL]
    print(f"  Got {len(docs)} chunks")

    context_parts = []
    cited_files = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        if source not in cited_files:
            cited_files.append(source)
        context_parts.append(f"[{source}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    chain = prompt | llm
    print("  Generating answer...")
    response = chain.invoke({"context": context, "question": question})

    return {
        "question":    question,
        "answer":      response.content,
        "cited_files": cited_files,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", "-q", type=str,
        default="How does FastAPI handle dependency injection?")
    args = parser.parse_args()

    print("=" * 55)
    print("  DevMind - Ask Your Codebase")
    print("=" * 55)
    print("\n[1/2] Loading retriever...")
    retriever = load_retriever()
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    print("\n[2/2] Running RAG pipeline...")
    result = ask(args.question, retriever, llm)

    print("\n" + "=" * 55)
    print(f"QUESTION: {result['question']}")
    print("=" * 55)
    print(f"\n{result['answer']}")
    print("\n--- Sources ---")
    for f in result["cited_files"]:
        print(f"  - {f}")

if __name__ == "__main__":
    main()
