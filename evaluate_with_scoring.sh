#!/bin/bash
# RAG Evaluation with Simple Quality Scoring
# Shows: Response Time + Basic Quality Metrics

API_URL="http://localhost:8000"
PROJECT="booking_sports"

echo "=========================================="
echo "🚀 DevMind RAG Evaluation with Scoring"
echo "=========================================="
echo ""

# Test questions with expected quality levels
declare -a QUESTIONS=(
    "What payment methods are supported?"
    "How does the checkout process work?"
    "What are the available API endpoints?"
    "How to handle errors in requests?"
    "What dependencies are required?"
)

TOTAL_TIME=0
HIGH_QUALITY=0
MED_QUALITY=0
LOW_QUALITY=0

for i in "${!QUESTIONS[@]}"; do
    Q="${QUESTIONS[$i]}"
    NUM=$((i+1))

    echo "[$NUM/${#QUESTIONS[@]}] Q: $Q"

    # Make API call
    RESPONSE=$(curl -s -X POST "$API_URL/query" \
        -H "Content-Type: application/json" \
        -d "{
            \"project\": \"$PROJECT\",
            \"question\": \"$Q\"
        }")

    # Extract data
    ANSWER=$(echo "$RESPONSE" | jq -r '.answer')
    RETRIEVAL=$(echo "$RESPONSE" | jq -r '.metrics.retrieval_time // 0')
    GENERATION=$(echo "$RESPONSE" | jq -r '.metrics.generation_time // 0')
    TOTAL=$(echo "$RESPONSE" | jq -r '.metrics.total_time // 0')
    CHUNKS=$(echo "$RESPONSE" | jq -r '.metrics.chunks_used // 0')
    SOURCES=$(echo "$RESPONSE" | jq -r '.sources | length')

    # Simple quality scoring (0-1)
    QUALITY=0

    # Check 1: Is there actual content? (not just "I don't have enough context")
    if ! echo "$ANSWER" | grep -q "don't have enough"; then
        QUALITY=$(echo "$QUALITY + 0.4" | bc)
    fi

    # Check 2: Multiple sources cited?
    if [ "$SOURCES" -ge 3 ]; then
        QUALITY=$(echo "$QUALITY + 0.3" | bc)
    elif [ "$SOURCES" -ge 1 ]; then
        QUALITY=$(echo "$QUALITY + 0.15" | bc)
    fi

    # Check 3: Good response time? (<2s is good)
    if (( $(echo "$TOTAL < 2" | bc -l) )); then
        QUALITY=$(echo "$QUALITY + 0.3" | bc)
    elif (( $(echo "$TOTAL < 4" | bc -l) )); then
        QUALITY=$(echo "$QUALITY + 0.15" | bc)
    fi

    # Categorize quality
    if (( $(echo "$QUALITY >= 0.8" | bc -l) )); then
        QUALITY_LABEL="🟢 HIGH"
        HIGH_QUALITY=$((HIGH_QUALITY + 1))
    elif (( $(echo "$QUALITY >= 0.5" | bc -l) )); then
        QUALITY_LABEL="🟡 MEDIUM"
        MED_QUALITY=$((MED_QUALITY + 1))
    else
        QUALITY_LABEL="🔴 LOW"
        LOW_QUALITY=$((LOW_QUALITY + 1))
    fi

    # Display answer preview
    PREVIEW=$(echo "$ANSWER" | head -c 80)
    echo "   A: $PREVIEW..."
    echo "   ⏱️  Total: ${TOTAL}s | Retrieval: ${RETRIEVAL}s | Generation: ${GENERATION}s"
    echo "   📊 Chunks: $CHUNKS | Sources: $SOURCES"
    echo "   ⭐ Quality Score: $(printf "%.2f" $QUALITY)/1.0  $QUALITY_LABEL"
    echo ""

    TOTAL_TIME=$(echo "$TOTAL_TIME + $TOTAL" | bc)
done

COUNT=${#QUESTIONS[@]}
AVG_TOTAL=$(echo "scale=2; $TOTAL_TIME / $COUNT" | bc)

echo "=========================================="
echo "📊 EVALUATION SUMMARY"
echo "=========================================="
echo "Total Queries: $COUNT"
echo "Avg Response Time: ${AVG_TOTAL}s"
echo ""
echo "Quality Distribution:"
echo "  🟢 HIGH (>0.8):    $HIGH_QUALITY queries"
echo "  🟡 MEDIUM (0.5-0.8): $MED_QUALITY queries"
echo "  🔴 LOW (<0.5):    $LOW_QUALITY queries"
echo ""
echo "💡 Quality Score = Content Quality (40%) + Source Coverage (30%) + Speed (30%)"
echo "=========================================="
