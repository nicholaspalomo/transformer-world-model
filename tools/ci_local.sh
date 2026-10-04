#!/usr/bin/env bash
# ==============================================================================
# Local CI emulator -- runs the same checks as .github/workflows/ci.yml before you
# push.
#
#   tools/ci_local.sh            everything, including the Docker build
#   tools/ci_local.sh --no-docker  skip the Docker stages (fast path)
#
# Nothing here is tied to one machine: the interpreter is discovered rather than
# hard-coded, and the container is addressed through `docker compose` so its image
# name follows COMPOSE_PROJECT_NAME instead of being spelled out.
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_DOCKER=1
[ "${1:-}" = "--no-docker" ] && RUN_DOCKER=0

# Same resolution order as the Makefile: an explicit PYTHON, an activated venv, a
# repo-local .venv, then PATH. An absolute path into one developer's home only ever
# worked on that developer's machine -- and on this one it now resolves to a bare
# system python with none of the dependencies installed.
if [ -n "${PYTHON:-}" ]; then
    PYTHON_BIN="${PYTHON}"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
    PYTHON_BIN="$(command -v python3)"
fi

step() { echo; echo "=========================================================="; echo "$1"; echo "=========================================================="; }

step "🚀 Local CI emulator   (python: ${PYTHON_BIN})"

if ! "${PYTHON_BIN}" -c "import jax, flax, brax" >/dev/null 2>&1; then
    echo "❌ ${PYTHON_BIN} cannot import jax/flax/brax."
    echo "   Create an environment and install the project first:"
    echo "     python3 -m venv .venv && . .venv/bin/activate && make install"
    echo "   ...or run the whole thing in the container:  make docker-up && make docker-shell"
    exit 1
fi

step "Step 1: IFTTT directives"
"${PYTHON_BIN}" tools/hooks/check_ifttt.py

step "Step 2: Linters and format check"
"${PYTHON_BIN}" -m ruff check .
"${PYTHON_BIN}" -m ruff format --check .
"${PYTHON_BIN}" -m black --check . --line-length=100

step "Step 3: Unit tests"
# Two discovery patterns: tests/test_diffusion_grpo.py is named the pytest way and
# would be silently skipped by "*_test.py" alone.
PYTHONPATH=. "${PYTHON_BIN}" -m unittest discover -s tests -p "*_test.py" -v
PYTHONPATH=. "${PYTHON_BIN}" -m unittest discover -s tests -p "test_*.py" -v

step "Step 4: Milestone scripts (smoke)"
PYTHONPATH=. "${PYTHON_BIN}" scripts/01_collect_data.py --num_steps 20 --seq_len 10
PYTHONPATH=. "${PYTHON_BIN}" scripts/02_train_model.py --num_steps 5 --batch_size 8
PYTHONPATH=. "${PYTHON_BIN}" scripts/03_evaluate_mppi.py --num_samples 10 --horizon 5 --eval_steps 2
PYTHONPATH=. "${PYTHON_BIN}" scripts/visualize_anymal.py --num_steps 5 --headless

if [ "${RUN_DOCKER}" -eq 0 ]; then
    step "✅ All non-Docker CI checks passed (--no-docker)"
    exit 0
fi

if ! docker info >/dev/null 2>&1; then
    echo "⚠️  Docker is not available; skipping the container stages."
    step "✅ All non-Docker CI checks passed"
    exit 0
fi

# Address the image through Compose rather than by a guessed name. The old
# hard-coded `transformer-world-model-twm_dev` was Compose's default naming for one
# particular directory name, so renaming the checkout -- or setting
# COMPOSE_PROJECT_NAME to run two of them side by side -- broke this step.
export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-twm}"
export USER_ID="${USER_ID:-$(id -u)}"
export GROUP_ID="${GROUP_ID:-$(id -g)}"
COMPOSE=(docker compose -f "${REPO_ROOT}/docker-compose.yml")

step "Step 5: Docker image build   (project: ${COMPOSE_PROJECT_NAME})"
"${COMPOSE[@]}" build

step "Step 6: Unit tests inside the container"
# --no-deps and no service ports: a one-shot `run` must not publish this project's
# host ports, or it would collide with an already-running `make docker-up` container
# of the same project. --rm so nothing is left behind.
"${COMPOSE[@]}" run --rm --no-deps --entrypoint="" "${SERVICE:-twm_dev}" \
    bash -lc 'PYTHONPATH=/workspace python -m unittest discover -s tests -p "*_test.py" -v && \
              PYTHONPATH=/workspace python -m unittest discover -s tests -p "test_*.py" -v'

step "✅ All local CI checks passed"
