import json
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
import numpy as np

load_dotenv()

CHROMA_PATH = "./data/chroma_db"
TOP_K_FINAL = 5

SYSTEM_PROMPT = """You are DevMind, an expert code assistant.
Answer the user question using ONLY the provided context chunks.
After every fact you state, add [filename] in brackets.
If the context does not contain the answer, say "I don't have enough context."
Format code in markdown code blocks.

Context:
{context}
"""

TEST_QUESTIONS = [
    "How does FastAPI handle dependency injection?",
    "What is the difference between APIRouter and FastAPI?",
    "How do you define path parameters in FastAPI?",
    "How does FastAPI validate request bodies?",
    "How do you add middleware in FastAPI?",
]

def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="devmind_code",
    )
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    all_docs = vectorstore.get()
    bm25_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = 10
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble, embeddings

def safe_score(val):
    if isinstance(val, list):
        clean = [v for v in val if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(clean)) if clean else 0.0
    if val is None:
        return 0.0
    return float(val)

def run_evaluation():
    print("=" * 55)
    print("  DevMind - Phase 1: Ragas Evaluation")
    print("=" * 55)

    print("\nLoading pipeline...")
    retriever, embeddings = load_retriever()
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    chain = prompt | llm

    results = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    print(f"\nRunning {len(TEST_QUESTIONS)} test questions...\n")

    for i, question in enumerate(TEST_QUESTIONS):
        print(f"  [{i+1}/{len(TEST_QUESTIONS)}] {question[:55]}...")
        docs = retriever.invoke(question)
        docs = docs[:TOP_K_FINAL]
        context_parts = [
            f"[{doc.metadata.get('source_file', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        ]
        context = "\n\n---\n\n".join(context_parts)
        response = chain.invoke({"context": context, "question": question})
        results["question"].append(question)
        results["answer"].append(response.content)
        results["contexts"].append([doc.page_content for doc in docs])
        results["ground_truth"].append("")

    dataset = Dataset.from_dict(results)

    print("\nScoring with Ragas using Groq...")
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    faith_score = safe_score(scores["faithfulness"])
    relevancy_score = safe_score(scores["answer_relevancy"])

    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    print(f"\n  Faithfulness:     {faith_score:.3f}  (target: > 0.80)")
    print(f"  Answer Relevancy: {relevancy_score:.3f}")

    if faith_score >= 0.8:
        print(f"\n  PHASE 1 MILESTONE PASSED!")
        print(f"  Faithfulness {faith_score:.3f} >= 0.80")
    else:
        print(f"\n  Score: {faith_score:.3f} — still good for first run!")

    output = {
        "faithfulness": faith_score,
        "answer_relevancy": relevancy_score,
    }
    with open("./data/eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to ./data/eval_results.json\n")

if __name__ == "__main__":
    run_evaluation()
