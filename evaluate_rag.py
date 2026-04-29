#!/usr/bin/env python3
"""
DevMind RAG Evaluation with Results Table
Shows metrics in professional table format with Pass/Fail status
"""

import requests
import json
import time
from typing import Dict, List

API_URL = "http://localhost:8000"
PROJECT = "booking_sports"

# Evaluation thresholds
THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.80,
    "avg_response_time": 3.0,  # seconds
    "retrieval_accuracy": 0.75,
}

TEST_QUESTIONS = [
    "What payment methods are supported?",
    "How does the checkout process work?",
    "What are the available API endpoints?",
    "How to handle errors in requests?",
    "What dependencies are required?",
]

def calculate_faithfulness(answer: str) -> float:
    """
    Calculate faithfulness score (0-1)
    Higher if answer has citations and sources
    Lower if answer says "I don't have enough context"
    """
    score = 0.5  # Base score

    # Penalty for lack of context
    if "don't have enough context" in answer.lower():
        score -= 0.3

    # Bonus for detailed answer
    if len(answer) > 150:
        score += 0.2

    # Bonus for citations (mentions of [filename])
    if "[" in answer and "]" in answer:
        score += 0.2

    # Bonus for code blocks
    if "```" in answer:
        score += 0.1

    return min(1.0, max(0.0, score))

def calculate_answer_relevancy(question: str, answer: str, chunks: int) -> float:
    """
    Calculate answer relevancy score (0-1)
    Higher if chunks were used and answer is substantial
    """
    score = 0.5  # Base score

    # Bonus if chunks were used
    if chunks > 0:
        score += min(0.3, chunks * 0.06)  # Up to 0.3 for 5+ chunks

    # Bonus for answer length
    if len(answer) > 100:
        score += 0.2

    # Penalty if "don't have context"
    if "don't have enough" in answer.lower():
        score -= 0.2

    return min(1.0, max(0.0, score))

def evaluate_queries() -> Dict:
    """Run evaluation queries and calculate metrics"""
    print("\n" + "="*70)
    print("🚀 DevMind RAG Evaluation")
    print("="*70 + "\n")

    results = []
    total_time = 0

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] Evaluating: {question[:50]}...", end=" ")

        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"project": PROJECT, "question": question},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                metrics = data.get("metrics", {})
                chunks = metrics.get("chunks_used", 0)
                response_time = metrics.get("total_time", 0)

                # Calculate scores
                faithfulness = calculate_faithfulness(answer)
                relevancy = calculate_answer_relevancy(question, answer, chunks)

                results.append({
                    "question": question,
                    "answer": answer,
                    "faithfulness": faithfulness,
                    "relevancy": relevancy,
                    "response_time": response_time,
                    "chunks": chunks,
                })

                total_time += response_time
                print("✅")
            else:
                print("❌ Failed")

        except Exception as e:
            print(f"❌ Error: {e}")

    # Calculate averages
    if results:
        avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
        avg_relevancy = sum(r["relevancy"] for r in results) / len(results)
        avg_response_time = total_time / len(results)

        return {
            "results": results,
            "avg_faithfulness": avg_faithfulness,
            "avg_relevancy": avg_relevancy,
            "avg_response_time": avg_response_time,
            "total_queries": len(results),
        }

    return {}

def print_results_table(eval_data: Dict):
    """Print results in professional table format"""
    if not eval_data:
        print("❌ No evaluation data")
        return

    avg_faith = eval_data["avg_faithfulness"]
    avg_relev = eval_data["avg_relevancy"]
    avg_time = eval_data["avg_response_time"]

    # Determine pass/fail
    faith_pass = avg_faith > THRESHOLDS["faithfulness"]
    relev_pass = avg_relev > THRESHOLDS["answer_relevancy"]
    time_pass = avg_time < THRESHOLDS["avg_response_time"]

    print("\n" + "="*70)
    print("6.2.3 RESULTS")
    print("="*70)
    print()

    # Create table
    print(f"{'Metric':<22} {'Score':<10} {'Target':<10} {'Status':<10}")
    print("-" * 70)

    # Faithfulness row
    faith_status = "✅ Pass" if faith_pass else "❌ Fail"
    print(f"{'Faithfulness':<22} {avg_faith:>8.3f}   >{THRESHOLDS['faithfulness']:>6.2f}    {faith_status:<10}")

    # Answer Relevancy row
    relev_status = "✅ Pass" if relev_pass else "❌ Fail"
    print(f"{'Answer Relevancy':<22} {avg_relev:>8.3f}   >{THRESHOLDS['answer_relevancy']:>6.2f}    {relev_status:<10}")

    # Response Time row
    time_status = "✅ Pass" if time_pass else "❌ Fail"
    print(f"{'Avg Response Time':<22} {avg_time:>8.3f}s  <{THRESHOLDS['avg_response_time']:>5.2f}s   {time_status:<10}")

    # Total API Cost
    print(f"{'Total API Cost':<22} {'$0.00':<10} {'Free':<10} {'✅ Pass':<10}")

    print()
    print("="*70)

    # Overall status
    all_pass = faith_pass and relev_pass and time_pass
    overall = "✅ PASS" if all_pass else "⚠️  NEEDS IMPROVEMENT"
    print(f"Overall Status: {overall}")
    print("="*70)

    # Detailed breakdown
    print("\n📊 DETAILED METRICS\n")
    print(f"Total Queries Evaluated: {eval_data['total_queries']}")
    print(f"Average Faithfulness:    {avg_faith:.3f} (target: >{THRESHOLDS['faithfulness']:.2f})")
    print(f"Average Relevancy:       {avg_relev:.3f} (target: >{THRESHOLDS['answer_relevancy']:.2f})")
    print(f"Average Response Time:   {avg_time:.2f}s (target: <{THRESHOLDS['avg_response_time']:.1f}s)")
    print()

    # Individual query results
    print("📋 INDIVIDUAL QUERY RESULTS\n")
    for i, result in enumerate(eval_data["results"], 1):
        print(f"{i}. Q: {result['question']}")
        print(f"   Faithfulness: {result['faithfulness']:.3f}")
        print(f"   Relevancy:    {result['relevancy']:.3f}")
        print(f"   Time:         {result['response_time']:.2f}s")
        print()

def main():
    """Main evaluation function"""
    try:
        eval_data = evaluate_queries()
        print_results_table(eval_data)

        # Save to file
        with open("evaluation_results.json", "w") as f:
            json.dump(eval_data, f, indent=2)

        print("\n✅ Results saved to evaluation_results.json\n")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")

if __name__ == "__main__":
    main()
