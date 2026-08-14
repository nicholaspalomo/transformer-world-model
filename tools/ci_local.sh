#!/usr/bin/env bash
# ==============================================================================
# Local CI Emulator — verifies all CI checks locally before pushing
# ==============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

echo "=========================================================="
echo "🚀 Running Local CI Emulator (GitHub Actions Validation)"
echo "=========================================================="

# 1. IFTTT and Google3 Linting Validation
echo ""
echo "=== Step 1: Running Google IFTTT Directives & Code Linters ==="
python3 tools/hooks/check_ifttt.py

if [ -f "${SCRIPT_DIR}/venv/bin/python" ]; then
    PYTHON_BIN="${SCRIPT_DIR}/venv/bin/python"
elif [ -f "/home/nico-palomo/workspace/venv/bin/python" ]; then
    PYTHON_BIN="/home/nico-palomo/workspace/venv/bin/python"
else
    PYTHON_BIN="python3"
fi

echo "Using Python: ${PYTHON_BIN}"
if command -v ruff >/dev/null 2>&1; then
    ruff check .
fi

echo ""
echo "=== Step 2: Running Local Unit Tests ==="
PYTHONPATH=. ${PYTHON_BIN} -m unittest discover -s tests -p "*_test.py" -v

echo ""
echo "=== Step 2: Testing Milestone Scripts ==="
PYTHONPATH=. ${PYTHON_BIN} scripts/01_collect_data.py --num_steps 20 --seq_len 10
PYTHONPATH=. ${PYTHON_BIN} scripts/02_train_model.py --num_steps 5 --batch_size 8
PYTHONPATH=. ${PYTHON_BIN} scripts/03_evaluate_mppi.py --num_samples 10 --horizon 5 --eval_steps 2
PYTHONPATH=. ${PYTHON_BIN} scripts/visualize_anymal.py --num_steps 5 --headless

# 2. Docker Build & Container Verification
echo ""
echo "=== Step 3: Building Docker Development Container ==="
docker compose build

echo ""
echo "=== Step 4: Running Unit Tests Inside Docker Container ==="
docker run --rm \
  -v "${SCRIPT_DIR}:/workspace" \
  -w /workspace \
  -e DISPLAY=:1 \
  transformer-world-model-twm_dev \
  bash -c "PYTHONPATH=. python3 -m unittest discover -s tests -p '*_test.py' -v"

echo ""
echo "=========================================================="
echo "✅ All Local CI Checks Passed Successfully!"
echo "=========================================================="
