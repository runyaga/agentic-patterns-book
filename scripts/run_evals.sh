#!/bin/bash
# Run pydantic-evals evaluations and generate reports
#
# Usage:
#   ./scripts/run_evals.sh routing           # Evaluate routing pattern
#   ./scripts/run_evals.sh multi_agent       # Evaluate multi-agent pattern
#   ./scripts/run_evals.sh routing --llm     # Include LLM judge
#   ./scripts/run_evals.sh all               # Evaluate all patterns
#
# Environment:
#   LOGFIRE_TOKEN - Set to enable Logfire tracing
#   OLLAMA_URL    - Ollama server URL (default: http://localhost:11434)

set -e

PATTERN=${1:-routing}
USE_LLM_JUDGE=false
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="reports"

# Parse flags
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --llm)
            USE_LLM_JUDGE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$REPORT_DIR"

run_eval() {
    local pattern=$1
    local report_file="${REPORT_DIR}/eval_${pattern}_${TIMESTAMP}.txt"

    echo "============================================================"
    echo "Running evaluation: $pattern"
    echo "Timestamp: $(date)"
    echo "LLM Judge: $USE_LLM_JUDGE"
    echo "============================================================"

    .venv/bin/python << PYTHON_SCRIPT | tee "$report_file"
import asyncio
from datetime import datetime

from agentic_patterns.evaluation import (
    configure_logfire_for_evals,
    evaluate_pattern,
)

# Import the actual pattern functions
from agentic_patterns.routing import route_query
from agentic_patterns.multi_agent import run_collaborative_task

# Configure logfire with scrubbing disabled
configure_logfire_for_evals(
    service_name="agentic-patterns-evals",
    environment="development",
)

PATTERN_FUNCTIONS = {
    "routing": route_query,
    "multi_agent": run_collaborative_task,
}

async def main():
    pattern = "${pattern}"
    use_llm_judge = ${USE_LLM_JUDGE}

    if pattern not in PATTERN_FUNCTIONS:
        print(f"Unknown pattern: {pattern}")
        print(f"Available: {list(PATTERN_FUNCTIONS.keys())}")
        return

    task_fn = PATTERN_FUNCTIONS[pattern]

    print(f"Pattern: {pattern}")
    print(f"Function: {task_fn.__name__}")
    print(f"LLM Judge: {use_llm_judge}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    report = await evaluate_pattern(
        pattern,
        task_fn,
        include_llm_judge=use_llm_judge,
        metadata={
            "run_id": "${TIMESTAMP}",
        },
    )

    # Print full report
    report.print(
        include_input=True,
        include_output=True,
        include_durations=True,
    )

    print()
    print("=" * 60)
    print("Logfire Dashboard")
    print("=" * 60)
    print("View traces: https://logfire.pydantic.dev/")
    print()

    # Summary stats
    print("Summary:")
    total_cases = len(report.cases)
    print(f"  Total cases: {total_cases}")

    # Count assertions passed
    passed = sum(
        1 for case in report.cases
        for a in case.assertions.values()
        if a.value is True
    )
    total_assertions = sum(len(case.assertions) for case in report.cases)
    if total_assertions > 0:
        print(f"  Assertions: {passed}/{total_assertions} passed")

asyncio.run(main())
PYTHON_SCRIPT

    echo ""
    echo "Report saved to: $report_file"
    echo ""
}

if [ "$PATTERN" = "all" ]; then
    for p in routing multi_agent; do
        run_eval "$p"
    done
else
    run_eval "$PATTERN"
fi

echo "============================================================"
echo "Evaluation complete!"
echo "Reports in: $REPORT_DIR/"
echo "============================================================"
