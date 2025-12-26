#!/usr/bin/env bash
# Integration test runner - requires Ollama running locally
set -e

PATTERNS=(
    prompt_chaining
    routing
    parallelization
    reflection
    tool_use
    planning
    multi_agent
    memory
    learning
    human_in_loop
    knowledge_retrieval
    resource_aware
    guardrails
    evaluation
    prioritization
)

PASSED=0
FAILED=0
FAILED_PATTERNS=()

echo "=== Integration Tests ==="
echo "Requires: Ollama running at localhost:11434"
echo ""

for pattern in "${PATTERNS[@]}"; do
    printf "%-25s" "$pattern"
    if timeout 120 .venv/bin/python -m "agentic_patterns.$pattern" > /dev/null 2>&1; then
        echo "✓ PASS"
        ((PASSED++))
    else
        echo "✗ FAIL"
        ((FAILED++))
        FAILED_PATTERNS+=("$pattern")
    fi
done

echo ""
echo "=== Results ==="
echo "Passed: $PASSED"
echo "Failed: $FAILED"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed patterns:"
    for p in "${FAILED_PATTERNS[@]}"; do
        echo "  - $p"
    done
    exit 1
fi

echo ""
echo "All integration tests passed!"
