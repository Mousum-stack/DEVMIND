"""
DevMind - Phase 1: Evaluation with Ragas
File: rag/evaluate.py

What this does:
  Runs your RAG pipeline against a test set and scores it on:
  - Faithfulness:       Does the answer stick to the retrieved context?
  - Answer Relevancy:   Does the answer actually address the question?
  - Context Precision:  Are the retrieved chunks actually relevant?

Your Phase 1 milestone: faithfulness score > 0.8

Usage:
  python rag/evaluate.py
"""

import os
import json
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.integrations.langchain import EvaluatorChain
from datasets import Dataset

from query import load_vectorstore, build_hybrid_retriever, ask

load_dotenv()

# ── Test questions (your evaluation set) ─────────────────────────────────────
# These are real questions about the FastAPI codebase.
# Add more as you learn what users actually ask.
TEST_QUESTIONS = [
    "How does FastAPI handle dependency injection?",
    "What is the difference between APIRouter and FastAPI?",
    "How do you define path parameters in FastAPI?",
    "How does FastAPI validate request bodies?",
    "What is the purpose of the lifespan parameter in FastAPI?",
    "How do you add middleware in FastAPI?",
    "What is the difference between sync and async routes in FastAPI?",
    "How does FastAPI generate OpenAPI documentation?",
]

# Ground truth answers (optional but improves score accuracy)
# If you don't have these, Ragas uses the context to evaluate
GROUND_TRUTHS = [
    "FastAPI uses Depends() to inject dependencies into route functions...",
    "APIRouter groups related routes; FastAPI is the main application...",
    # Add more or leave empty — Ragas still works without them
]
# ─────────────────────────────────────────────────────────────────────────────


def run_evaluation():
    print("=" * 55)
    print("  DevMind — Phase 1: Ragas Evaluation")
    print("=" * 55)

    print("\nLoading pipeline...")
    vectorstore = load_vectorstore()
    retriever   = build_hybrid_retriever(vectorstore)
    llm         = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Collect answers for all test questions
    results = {
        "question":        [],
        "answer":          [],
        "contexts":        [],
        "ground_truth":    [],
    }

    print(f"\nRunning {len(TEST_QUESTIONS)} test questions...\n")

    for i, question in enumerate(TEST_QUESTIONS):
        print(f"  [{i+1}/{len(TEST_QUESTIONS)}] {question[:60]}...")

        from langchain_community.vectorstores import Chroma
        docs = retriever.invoke(question)
        from query import format_context, SYSTEM_PROMPT, HUMAN_PROMPT
        from langchain_core.prompts import ChatPromptTemplate

        context_str, _ = format_context(docs)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human",  HUMAN_PROMPT),
        ])
        chain    = prompt | llm
        response = chain.invoke({"context": context_str, "question": question})

        results["question"].append(question)
        results["answer"].append(response.content)
        results["contexts"].append([doc.page_content for doc in docs])
        results["ground_truth"].append(
            GROUND_TRUTHS[i] if i < len(GROUND_TRUTHS) else ""
        )

    # Convert to Ragas Dataset format
    dataset = Dataset.from_dict(results)

    print("\nScoring with Ragas...")
    metrics = [faithfulness, answer_relevancy, context_precision]
    scores  = evaluate(dataset, metrics=metrics)

    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    print(f"\n  Faithfulness:      {scores['faithfulness']:.3f}  (target: > 0.80)")
    print(f"  Answer Relevancy:  {scores['answer_relevancy']:.3f}")
    print(f"  Context Precision: {scores['context_precision']:.3f}")

    # Phase 1 milestone check
    faith_score = scores["faithfulness"]
    if faith_score >= 0.8:
        print(f"\n  ✓ MILESTONE PASSED — faithfulness {faith_score:.3f} >= 0.80")
    else:
        print(f"\n  ✗ Not there yet — faithfulness {faith_score:.3f} < 0.80")
        print("  Tips to improve:")
        print("    - Increase TOP_K_RERANK in query.py")
        print("    - Try smaller CHUNK_SIZE in ingest.py")
        print("    - Check that your prompt isn't hallucinating")

    # Save results to file for later comparison
    output = {
        "scores": {
            "faithfulness":      float(scores["faithfulness"]),
            "answer_relevancy":  float(scores["answer_relevancy"]),
            "context_precision": float(scores["context_precision"]),
        },
        "num_questions": len(TEST_QUESTIONS),
    }
    with open("./data/eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to ./data/eval_results.json\n")


if __name__ == "__main__":
    run_evaluation()
