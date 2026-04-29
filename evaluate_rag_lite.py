"""
Lightweight RAG Evaluation Tool
Bypasses TensorFlow/protobuf issues by using existing vectorstore
"""

import os
import json
import time
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Constants
CHROMA_PATH = "./data/chroma_db"
TOP_K = 5
PROJECT = "booking_sports"

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

# Test questions for evaluation
TEST_QUESTIONS = [
    "What payment methods are supported?",
    "How does the checkout process work?",
    "What are the available API endpoints?",
    "How to handle errors in requests?",
    "What dependencies are required?",
]

class LightweightRAGEvaluator:
    def __init__(self):
        """Initialize evaluator with Groq LLM and loaded Chroma DB"""
        from sentence_transformers import SentenceTransformer

        print("🚀 Initializing RAG Evaluator...")

        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Embedding model loaded")
        except Exception as e:
            print(f"⚠️  Embedding model error: {e}")
            self.embedding_model = None

        # Initialize LLM
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        print("✅ LLM initialized")

        # Load vectorstore
        self.vectorstore = self._load_vectorstore()

    def _load_vectorstore(self):
        """Load the Chroma vectorstore"""
        if not Path(CHROMA_PATH).exists():
            print(f"❌ Chroma database not found at {CHROMA_PATH}")
            return None

        print(f"📚 Loading vectorstore from {CHROMA_PATH}...")
        try:
            # For offline use, we need to load embeddings differently
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embeddings,
                collection_name="devmind_code",
            )
            print(f"✅ Vectorstore loaded with {vectorstore._collection.count()} chunks")
            return vectorstore
        except Exception as e:
            print(f"❌ Error loading vectorstore: {e}")
            return None

    def retrieve_context(self, question: str) -> tuple:
        """Retrieve context for a question"""
        if self.vectorstore is None:
            return [], ""

        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": TOP_K})
            docs = retriever.invoke(question)

            context_parts = []
            sources = []

            for doc in docs:
                source = doc.metadata.get("source", "unknown")
                if source not in sources:
                    sources.append(source)
                context_parts.append(f"[{source}]\n{doc.page_content}")

            context = "\n\n---\n\n".join(context_parts)
            return sources, context
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return [], ""

    def generate_answer(self, question: str, context: str) -> str:
        """Generate an answer using the LLM"""
        try:
            prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
            chain = prompt | self.llm

            result = chain.invoke({
                "question": question,
                "context": context
            })
            return result.content
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return ""

    def evaluate_single(self, question: str) -> Dict:
        """Evaluate a single question"""
        print(f"  🔍 Evaluating: {question[:50]}...")

        start = time.time()
        sources, context = self.retrieve_context(question)
        retrieval_time = time.time() - start

        start = time.time()
        answer = self.generate_answer(question, context)
        generation_time = time.time() - start

        # Basic quality metrics
        answer_length = len(answer)
        has_context = len(context) > 0
        has_sources = len(sources) > 0

        return {
            "question": question,
            "answer": answer[:150] + "..." if len(answer) > 150 else answer,
            "full_answer": answer,
            "context_length": len(context),
            "contexts": [context] if context else [],
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": retrieval_time + generation_time,
            "sources": sources,
            "answer_length": answer_length,
            "has_context": has_context,
            "has_sources": has_sources,
            "num_sources": len(sources),
        }

    def evaluate_batch(self, questions: List[str] = None) -> List[Dict]:
        """Evaluate multiple questions"""
        if questions is None:
            questions = TEST_QUESTIONS

        results = []
        print(f"\n{'='*70}")
        print(f"🚀 Starting RAG Evaluation on {len(questions)} questions")
        print(f"{'='*70}\n")

        for i, question in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}]", end=" ")
            result = self.evaluate_single(question)
            results.append(result)
            print(f"✅ {result['total_time']:.2f}s")

        return results

    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute aggregate metrics"""
        if not results:
            return {}

        return {
            "total_questions": len(results),
            "avg_retrieval_time": sum(r["retrieval_time"] for r in results) / len(results),
            "avg_generation_time": sum(r["generation_time"] for r in results) / len(results),
            "avg_total_time": sum(r["total_time"] for r in results) / len(results),
            "avg_answer_length": sum(r["answer_length"] for r in results) / len(results),
            "avg_num_sources": sum(r["num_sources"] for r in results) / len(results),
            "with_context_pct": (sum(1 for r in results if r["has_context"]) / len(results)) * 100,
            "with_sources_pct": (sum(1 for r in results if r["has_sources"]) / len(results)) * 100,
        }

    def print_results(self, results: List[Dict], metrics: Dict = None):
        """Pretty print evaluation results"""
        print(f"\n{'='*70}")
        print("📊 EVALUATION RESULTS")
        print(f"{'='*70}\n")

        # Individual results
        for i, result in enumerate(results, 1):
            print(f"{i}. Q: {result['question']}")
            print(f"   A: {result['answer']}")
            print(f"   ⏱️  Retrieval: {result['retrieval_time']:.2f}s | Generation: {result['generation_time']:.2f}s | Total: {result['total_time']:.2f}s")
            print(f"   📁 Sources: {len(result['sources'])} files | Answer: {result['answer_length']} chars")
            if result['sources']:
                print(f"      Files: {', '.join(result['sources'][:2])}")
            print()

        # Aggregate metrics
        if metrics:
            print(f"\n{'='*70}")
            print("🎯 PERFORMANCE METRICS")
            print(f"{'='*70}\n")

            print(f"Total Questions: {int(metrics['total_questions'])}")
            print(f"Avg Retrieval Time: {metrics['avg_retrieval_time']:.2f}s")
            print(f"Avg Generation Time: {metrics['avg_generation_time']:.2f}s")
            print(f"Avg Total Time: {metrics['avg_total_time']:.2f}s")
            print(f"Avg Answer Length: {int(metrics['avg_answer_length'])} chars")
            print(f"Avg Sources per Query: {metrics['avg_num_sources']:.1f}")
            print(f"Answers with Context: {metrics['with_context_pct']:.0f}%")
            print(f"Answers with Sources: {metrics['with_sources_pct']:.0f}%")

def main():
    """Main evaluation function"""
    try:
        evaluator = LightweightRAGEvaluator()

        if evaluator.vectorstore is None:
            print("❌ Cannot run evaluation without vectorstore")
            return

        # Run batch evaluation
        results = evaluator.evaluate_batch()

        # Compute metrics
        metrics = evaluator.compute_metrics(results)

        # Print results
        evaluator.print_results(results, metrics)

        # Save results to file
        output_file = "evaluation_results.json"
        with open(output_file, "w") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": [
                    {k: v for k, v in r.items() if k != "full_answer"}
                    for r in results
                ],
                "metrics": metrics,
            }, f, indent=2, default=str)

        print(f"\n✅ Results saved to {output_file}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
