#!/bin/bash
# Manual RAG Evaluation using API calls
# Run: chmod +x evaluate_api.sh && ./evaluate_api.sh

API_URL="http://localhost:8000"
PROJECT="booking_sports"

echo "=========================================="
echo "🚀 DevMind RAG Evaluation"
echo "=========================================="
echo ""

# Test questions
declare -a QUESTIONS=(
    "What payment methods are supported?"
    "How does the checkout process work?"
    "What are the available API endpoints?"
    "How to handle errors in requests?"
    "What dependencies are required?"
)

TOTAL_RETRIEVAL=0
TOTAL_GENERATION=0
TOTAL_TIME=0
COUNT=0

for i in "${!QUESTIONS[@]}"; do
    Q="${QUESTIONS[$i]}"
    NUM=$((i+1))

    echo "[$NUM/${#QUESTIONS[@]}] Q: $Q"

    # Make API call and capture response
    RESPONSE=$(curl -s -X POST "$API_URL/query" \
        -H "Content-Type: application/json" \
        -d "{
            \"project\": \"$PROJECT\",
            \"question\": \"$Q\"
        }")

    # Extract metrics
    ANSWER=$(echo "$RESPONSE" | jq -r '.answer' | head -c 100)
    RETRIEVAL=$(echo "$RESPONSE" | jq -r '.metrics.retrieval_time // 0')
    GENERATION=$(echo "$RESPONSE" | jq -r '.metrics.generation_time // 0')
    TOTAL=$(echo "$RESPONSE" | jq -r '.metrics.total_time // 0')
    CHUNKS=$(echo "$RESPONSE" | jq -r '.metrics.chunks_used // 0')
    SOURCES=$(echo "$RESPONSE" | jq -r '.sources | length')

    echo "   A: $ANSWER..."
    echo "   ⏱️  Retrieval: ${RETRIEVAL}s | Generation: ${GENERATION}s | Total: ${TOTAL}s"
    echo "   📊 Chunks: $CHUNKS | Sources: $SOURCES"
    echo ""

    # Accumulate totals
    TOTAL_RETRIEVAL=$(echo "$TOTAL_RETRIEVAL + $RETRIEVAL" | bc)
    TOTAL_GENERATION=$(echo "$TOTAL_GENERATION + $GENERATION" | bc)
    TOTAL_TIME=$(echo "$TOTAL_TIME + $TOTAL" | bc)
    COUNT=$((COUNT + 1))
done

# Calculate averages
if [ $COUNT -gt 0 ]; then
    AVG_RETRIEVAL=$(echo "scale=2; $TOTAL_RETRIEVAL / $COUNT" | bc)
    AVG_GENERATION=$(echo "scale=2; $TOTAL_GENERATION / $COUNT" | bc)
    AVG_TOTAL=$(echo "scale=2; $TOTAL_TIME / $COUNT" | bc)

    echo "=========================================="
    echo "📊 EVALUATION METRICS"
    echo "=========================================="
    echo "Total Queries: $COUNT"
    echo "Avg Retrieval Time: ${AVG_RETRIEVAL}s"
    echo "Avg Generation Time: ${AVG_GENERATION}s"
    echo "Avg Total Time: ${AVG_TOTAL}s"
    echo "=========================================="
fi
