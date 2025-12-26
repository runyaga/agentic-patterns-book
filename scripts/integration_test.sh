#!/usr/bin/env bash
# Integration test runner - requires Ollama running locally

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
REQUIRED_MODEL="${REQUIRED_MODEL:-gpt-oss:20b}"
RETRY_COUNT="${RETRY_COUNT:-2}"
TIMEOUT_SECS="${TIMEOUT_SECS:-120}"

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

# Results tracking
PASSED=0
FAILED=0
RETRIED=0
RESULTS=""

# === Preflight Checks ===
echo "=== Preflight Checks ==="

# Check if Ollama is running
printf "Ollama running at %s... " "$OLLAMA_URL"
if ! curl -s "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
    echo "FAIL"
    echo ""
    echo "ERROR: Ollama is not running at $OLLAMA_URL"
    echo "Start Ollama with: ollama serve"
    exit 1
fi
echo "OK"

# Check if required model exists
printf "Model '%s' available... " "$REQUIRED_MODEL"
if ! curl -s "$OLLAMA_URL/api/tags" | grep -q "\"name\":\"$REQUIRED_MODEL\""; then
    echo "FAIL"
    echo ""
    echo "ERROR: Model '$REQUIRED_MODEL' not found"
    echo "Pull it with: ollama pull $REQUIRED_MODEL"
    echo ""
    echo "Available models:"
    curl -s "$OLLAMA_URL/api/tags" | python3 -c "import sys,json; [print(f'  - {m[\"name\"]}') for m in json.load(sys.stdin).get('models',[])]"
    exit 1
fi
echo "OK"

# Check if .venv exists
printf "Virtual environment... "
if [ ! -f ".venv/bin/python" ]; then
    echo "FAIL"
    echo ""
    echo "ERROR: .venv not found. Run: uv venv && uv pip install -e '.[dev]'"
    exit 1
fi
echo "OK"

# Check if LOGFIRE_TOKEN is set
printf "LOGFIRE_TOKEN set... "
if [ -z "$LOGFIRE_TOKEN" ]; then
    echo "WARN"
    echo ""
    echo "WARNING: LOGFIRE_TOKEN not set. Observability disabled."
    echo "Set it with: export LOGFIRE_TOKEN=your_token"
    echo ""
else
    echo "OK"
    # Get project URL from logfire (it prints to stderr on configure)
    LOGFIRE_URL=$(.venv/bin/python -c "
import logfire
logfire.configure(send_to_logfire='if-token-present')
" 2>&1 | grep -o 'https://[^ ]*' | head -1)
    echo ""
    echo "Logfire dashboard: ${LOGFIRE_URL:-https://logfire.pydantic.dev/}"
fi

echo ""
echo "Configuration: retries=$RETRY_COUNT, timeout=${TIMEOUT_SECS}s"
echo ""

# === Run Tests ===
# Portable timeout function (works on macOS and Linux)
run_with_timeout() {
    local timeout=$1
    shift

    # Run command in background
    "$@" &
    local pid=$!

    # Start a timer in background that will kill the process
    (
        sleep "$timeout"
        kill -9 $pid 2>/dev/null
    ) &
    local timer_pid=$!

    # Wait for the command to finish
    wait $pid 2>/dev/null
    local exit_code=$?

    # Kill the timer if command finished before timeout
    kill $timer_pid 2>/dev/null
    wait $timer_pid 2>/dev/null

    return $exit_code
}

run_pattern() {
    local pattern=$1
    run_with_timeout "$TIMEOUT_SECS" .venv/bin/python -m "agentic_patterns.$pattern" > /dev/null 2>&1
    return $?
}

echo "=== Integration Tests ==="
echo ""

FIRST_FAILED=""

for pattern in "${PATTERNS[@]}"; do
    attempt=1
    success=false
    attempts_needed=0

    while [ $attempt -le $((RETRY_COUNT + 1)) ]; do
        if [ $attempt -eq 1 ]; then
            printf "%-25s" "$pattern"
        else
            printf "%-25s" "  (retry $((attempt-1)))"
            RETRIED=$((RETRIED + 1))
        fi

        if run_pattern "$pattern"; then
            echo "PASS"
            success=true
            attempts_needed=$attempt
            break
        else
            if [ $attempt -le $RETRY_COUNT ]; then
                echo "FAIL - retrying..."
            else
                echo "FAIL"
            fi
        fi
        attempt=$((attempt + 1))
    done

    if $success; then
        PASSED=$((PASSED + 1))
        if [ $attempts_needed -gt 1 ]; then
            RESULTS="${RESULTS}PASS:${pattern}:${attempts_needed}\n"
        else
            RESULTS="${RESULTS}PASS:${pattern}:1\n"
        fi
    else
        FAILED=$((FAILED + 1))
        RESULTS="${RESULTS}FAIL:${pattern}:$((RETRY_COUNT + 1))\n"
        if [ -z "$FIRST_FAILED" ]; then
            FIRST_FAILED="$pattern"
        fi
    fi
done

# === Report ===
echo ""
echo "============================================================"
echo "                    TEST REPORT"
echo "============================================================"
echo ""
printf "%-20s %s\n" "Total patterns:" "${#PATTERNS[@]}"
printf "%-20s %s\n" "Passed:" "$PASSED"
printf "%-20s %s\n" "Failed:" "$FAILED"
printf "%-20s %s\n" "Total retries:" "$RETRIED"
echo ""

# Show passed patterns
echo "Passed patterns:"
echo -e "$RESULTS" | grep "^PASS:" | while IFS=: read -r status pattern attempts; do
    if [ "$attempts" -gt 1 ]; then
        printf "  [PASS] %-25s (after %d attempts)\n" "$pattern" "$attempts"
    else
        printf "  [PASS] %s\n" "$pattern"
    fi
done
echo ""

# Show failed patterns
if [ $FAILED -gt 0 ]; then
    echo "Failed patterns (after $((RETRY_COUNT + 1)) attempts each):"
    echo -e "$RESULTS" | grep "^FAIL:" | while IFS=: read -r status pattern attempts; do
        printf "  [FAIL] %s\n" "$pattern"
    done
    echo ""
    echo "To debug a failing pattern, run:"
    echo "  .venv/bin/python -m agentic_patterns.$FIRST_FAILED"
    echo ""
fi

echo "============================================================"

if [ $FAILED -gt 0 ]; then
    echo "Result: FAILED ($FAILED/${#PATTERNS[@]} patterns failed)"
    exit 1
else
    echo "Result: ALL TESTS PASSED"
    exit 0
fi
