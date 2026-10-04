SHELL := /bin/bash

# ==============================================================================
# Transformer World Model -- development & container orchestration
#
# The container this repo brings up is designed to share a machine with other
# Docker stacks without stepping on them:
#
#   * every published port is a variable with a project-specific default, and
#     `make docker-up` refuses to start if one of them is already taken;
#   * the Compose project name (default `twm`) namespaces the container, network,
#     image and volumes, so nothing is addressed by a global fixed name;
#   * caches live in a project-scoped named volume, not in the host's ~/.cache;
#   * the container runs as the host UID/GID, so it never leaves root-owned files
#     in the bind-mounted working tree.
#
# Machine-local overrides go in `.env` (`make env` to bootstrap one). See
# .env.example for what can be changed and why.
# ==============================================================================

# ---------------------------------------------------------------- local config
# Compose reads .env by itself; make does not, so include it here as well --
# otherwise the URLs printed below would still claim the default port while the
# container was actually published somewhere else. `-include` so a missing .env is
# not an error; the defaults below then apply.
-include .env
export

# Make keeps the quotes on `FOO="6099"` from a .env line; Compose strips them.
# Because `export` makes the Make variable win over the file, Compose would then
# receive a literal `"6099"` and fail with `invalid hostPort`, while check-ports
# below would grep for `:"6099" ` and cheerfully report the port free. Strip both
# quote styles from every value we read back out of .env.
unquote = $(subst ",,$(subst ','',$(1)))

USER_ID  ?= $(shell id -u)
GROUP_ID ?= $(shell id -g)

COMPOSE_PROJECT_NAME := $(call unquote,$(or $(COMPOSE_PROJECT_NAME),twm))

# LINT.IfChange(vnc_ports)
# Host-side ports. Defaults deliberately avoid the canonical 5900/5901/6080/6006/8888
# that other VNC and notebook containers claim. Only the host side is configurable;
# inside the container the services stay on 5900/6080/6006/8888.
HOST_VNC_PORT         := $(call unquote,$(or $(HOST_VNC_PORT),5915))
HOST_NOVNC_PORT       := $(call unquote,$(or $(HOST_NOVNC_PORT),6095))
HOST_TENSORBOARD_PORT := $(call unquote,$(or $(HOST_TENSORBOARD_PORT),6016))
HOST_JUPYTER_PORT     := $(call unquote,$(or $(HOST_JUPYTER_PORT),8898))
BIND_ADDR             := $(call unquote,$(or $(BIND_ADDR),127.0.0.1))
# LINT.ThenChange(//docker-compose.yml:vnc_ports, //.env.example, //scripts/start_vnc.sh:vnc_env)

VNC_URL := http://localhost:$(HOST_NOVNC_PORT)/vnc_lite.html?scale=true
TB_URL  := http://localhost:$(HOST_TENSORBOARD_PORT)

# Bind-mounting a path that does not exist on the host makes Docker create it there
# as an empty directory. Substitute harmless stand-ins instead, so bringing the
# container up never writes anything into the host's home.
HOST_GITCONFIG ?= $(if $(wildcard $(HOME)/.gitconfig),$(HOME)/.gitconfig,/dev/null)
HOST_SSH_DIR   ?= $(if $(wildcard $(HOME)/.ssh),$(HOME)/.ssh,./docker/empty)

COMPOSE_ENV := COMPOSE_PROJECT_NAME=$(COMPOSE_PROJECT_NAME) \
               USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) \
               HOST_GITCONFIG=$(HOST_GITCONFIG) HOST_SSH_DIR=$(HOST_SSH_DIR)

COMPOSE_BASE := $(COMPOSE_ENV) docker compose -f docker-compose.yml
COMPOSE_GPU  := $(COMPOSE_ENV) docker compose -f docker-compose.yml -f docker-compose.gpu.yml
SERVICE      := twm_dev

# ------------------------------------------------------------------- interpreter
# Resolution order: an explicit PYTHON=..., then a virtualenv activated in this
# shell, then a repo-local .venv, then whatever python3 is on PATH. Never an
# absolute path to one developer's home -- that only ever works on one machine.
ifeq ($(origin PYTHON), undefined)
  ifneq ($(VIRTUAL_ENV),)
    PYTHON := $(VIRTUAL_ENV)/bin/python
  else ifneq ($(wildcard $(CURDIR)/.venv/bin/python),)
    PYTHON := $(CURDIR)/.venv/bin/python
  else
    PYTHON := python3
  endif
endif

# Always invoke tools as `$(PYTHON) -m <tool>` rather than bare `ruff`/`black`, so
# they come from the same interpreter as the code being checked.
RUN_PY := PYTHONPATH=. $(PYTHON)

# -------------------------------------------------- host vs. container detection
# The repository is bind-mounted into the container, so this same Makefile is
# visible in both places and `make test` is a natural thing to type in either. On
# the host the command has to be wrapped in `docker compose exec`; inside the
# container it must not be, because there is no docker CLI in there and the wrap
# would fail with a bare "docker: command not found" that explains nothing.
#
# /.dockerenv is created by Docker in every container and never exists on the host.
ifneq ($(wildcard /.dockerenv),)
  INSIDE_CONTAINER := 1
endif

ifdef INSIDE_CONTAINER
  IN_CONTAINER = bash -lc
  HOST_ONLY = @echo "❌ '$@' manages the container, so it has to run on the HOST."; \
              echo "   Open a terminal on the host, cd to the repository root, and run:  make $@"; \
              exit 1
else
  IN_CONTAINER = $(COMPOSE_BASE) exec -T $(SERVICE) bash -lc
  HOST_ONLY = @true
endif

.PHONY: help env ports doctor install test test-all lint format check-ifttt install-hooks \
        ci-local pipeline pipeline-anymal pipeline-smoke collect-data collect-anymal \
        train train-anymal-wm train-wm evaluate evaluate-anymal visualize \
        visualize-anymal train-grpo visualize-grpo tensorboard \
        bazel-build bazel-test bazel-clean lock-deps \
        docker-build docker-up docker-up-gpu docker-down docker-restart docker-shell \
        docker-logs docker-status docker-clean \
        start-vnc stop-vnc vnc-status check-zombies clean clean-artifacts clean-pipeline

help:
	@echo "================================================================="
	@echo "  Transformer World Model  (project '$(COMPOSE_PROJECT_NAME)')"
	@echo "================================================================="
	@echo "  Setup"
	@echo "    make env               Create .env from .env.example (host-local overrides)"
	@echo "    make ports             Show this stack's host ports and whether they are free"
	@echo "    make doctor            Check the toolchain and report what is missing"
	@echo "    make install           Install Python dependencies into the current interpreter"
	@echo "    make install-hooks     Install the git pre-commit hooks"
	@echo ""
	@echo "  Container lifecycle (host only)"
	@echo "    make docker-build      Build the development image"
	@echo "    make docker-up         Start the container (CPU); checks ports are free first"
	@echo "    make docker-up-gpu     Start the container with one NVIDIA GPU attached"
	@echo "    make docker-down       Stop and remove the container and its network"
	@echo "    make docker-restart    docker-down followed by docker-up"
	@echo "    make docker-shell      Interactive bash shell inside the container"
	@echo "    make docker-logs       Follow the container logs"
	@echo "    make docker-status     Show this project's containers, ports and volumes"
	@echo "    make docker-clean      Remove this project's containers, network AND volumes"
	@echo ""
	@echo "  Visualization  ($(VNC_URL))"
	@echo "    make start-vnc         Start Xvfb/Openbox/x11vnc/noVNC in the container"
	@echo "    make stop-vnc          Stop them"
	@echo "    make vnc-status        Report which of them are running"
	@echo "    make tensorboard       Serve training logs at $(TB_URL)"
	@echo ""
	@echo "  Quality"
	@echo "    make test              Run the unit test suite"
	@echo "    make lint              Ruff + Flake8 + the IFTTT cross-file validator"
	@echo "    make format            Format with Ruff & Black"
	@echo "    make check-ifttt       Run the LINT.IfChange / LINT.ThenChange validator"
	@echo "    make ci-local          Run the complete CI workflow locally"
	@echo "    make bazel-build       Build all Bazel targets"
	@echo "    make bazel-test        Run all Bazel tests"
	@echo ""
	@echo "  Locomotion Pipeline (ANYmal B)"
	@echo "    make pipeline          Run full pipeline (collect -> train WM -> GRPO in imagination -> eval -> vis)"
	@echo "    make pipeline-smoke    Rapid smoke test of the end-to-end pipeline"
	@echo "    make collect-anymal    Collect ANYmal transitions into data/anymal_trajectories.npz"
	@echo "    make train-anymal-wm   Train Causal Transformer World Model on ANYmal data"
	@echo "    make train-grpo        Train Diffusion Policy in World Model imagination (GRPO)"
	@echo "    make evaluate-anymal   Closed-loop MPPI evaluation with World Model (ANYmal B)"
	@echo "    make visualize-grpo    Diffusion policy telemetry plot + 3D HTML"
	@echo "    make visualize-anymal  ANYmal B simulation on the VNC display + 3D HTML"
	@echo "    make clean-pipeline    Remove generated replay buffers, checkpoints, and visual artifacts"
	@echo ""
	@echo "  Experiments (Ant & Generic)"
	@echo "    make collect-data      Milestone 1: Brax data collection (Ant)"
	@echo "    make train             Milestones 2 & 3: train the Transformer World Model"
	@echo "    make evaluate          Milestone 4: closed-loop MPPI evaluation (Ant)"
	@echo "    make visualize         Rollout comparison plot (real vs imagined)"
	@echo ""
	@echo "  Cleanup"
	@echo "    make clean             Remove caches and byte-compiled files"
	@echo "    make clean-artifacts   Also remove generated plots and HTML rollouts"
	@echo "    make check-zombies     Report unreaped processes in the container"
	@echo "================================================================="
ifdef INSIDE_CONTAINER
	@echo "  Running INSIDE the container: build/run/test targets execute directly."
	@echo "  The docker-* targets manage the container and must run on the host."
else
	@echo "  Running on the HOST. Python targets use: $(PYTHON)"
endif
	@echo "================================================================="

# ------------------------------------------------------------------ host setup

env:
	@if [ -f .env ]; then \
		echo "ℹ️  .env already exists; leaving it alone. Defaults are in .env.example."; \
	else \
		cp .env.example .env; \
		echo "✅ Created .env from .env.example. Every value is commented out, so the"; \
		echo "   defaults apply until you uncomment something. Run 'make ports' to see"; \
		echo "   whether any of them collide with a container already on this machine."; \
	fi

# The whole point of the port defaults is that they do not collide -- but another
# stack can still be holding one, so say so plainly instead of letting `docker
# compose up` fail with "address already in use" and no hint about what to do.
ports:
	@echo "Host ports for project '$(COMPOSE_PROJECT_NAME)' (bound to $(BIND_ADDR)):"
	@mine=""; [ -n "$$($(COMPOSE_BASE) ps -q $(SERVICE) 2>/dev/null)" ] && mine=" (held by THIS stack)"; \
	 [ -n "$$mine" ] && echo "  note: project '$(COMPOSE_PROJECT_NAME)' is running, so ports shown as in use are its own." || true
	@printf '  %-14s %-6s -> container %-5s  %s\n' \
		noVNC       "$(HOST_NOVNC_PORT)"       6080 "$$(if ss -tulpn 2>/dev/null | grep -q ':$(HOST_NOVNC_PORT) '; then echo '❌ IN USE'; else echo '✅ free'; fi)"; \
	printf '  %-14s %-6s -> container %-5s  %s\n' \
		VNC         "$(HOST_VNC_PORT)"         5900 "$$(if ss -tulpn 2>/dev/null | grep -q ':$(HOST_VNC_PORT) '; then echo '❌ IN USE'; else echo '✅ free'; fi)"; \
	printf '  %-14s %-6s -> container %-5s  %s\n' \
		TensorBoard "$(HOST_TENSORBOARD_PORT)" 6006 "$$(if ss -tulpn 2>/dev/null | grep -q ':$(HOST_TENSORBOARD_PORT) '; then echo '❌ IN USE'; else echo '✅ free'; fi)"; \
	printf '  %-14s %-6s -> container %-5s  %s\n' \
		Jupyter     "$(HOST_JUPYTER_PORT)"     8888 "$$(if ss -tulpn 2>/dev/null | grep -q ':$(HOST_JUPYTER_PORT) '; then echo '❌ IN USE'; else echo '✅ free'; fi)"; \
	echo "  Override any of them in .env (see .env.example), then re-run 'make ports'."

# Run before `up`. A port held by another stack is the one failure mode that would
# otherwise surface as a raw Docker error, or -- worse, if that stack is stopped and
# restarts later -- as two containers fighting over the same port.
.PHONY: check-ports
check-ports:
	@if [ -n "$$($(COMPOSE_BASE) ps -q $(SERVICE) 2>/dev/null)" ]; then \
		echo "ℹ️  Project '$(COMPOSE_PROJECT_NAME)' is already running; its own ports are not a collision."; \
		exit 0; \
	fi; \
	command -v ss >/dev/null 2>&1 || { echo "⚠️  'ss' not found (iproute2); cannot check ports. Continuing."; exit 0; }; \
	busy=""; \
	for spec in "noVNC:$(HOST_NOVNC_PORT)" "VNC:$(HOST_VNC_PORT)" "TensorBoard:$(HOST_TENSORBOARD_PORT)" "Jupyter:$(HOST_JUPYTER_PORT)"; do \
		name=$${spec%%:*}; port=$${spec##*:}; \
		if ss -tulpn 2>/dev/null | grep -q ":$${port} "; then busy="$${busy}\n     $${name} port $${port}"; fi; \
	done; \
	if [ -n "$$busy" ]; then \
		echo "❌ Refusing to start: host port(s) already in use:"; \
		printf "$$busy\n"; \
		echo ""; \
		echo "   Something else on this machine holds them. Who:"; \
		echo "     docker ps --format '{{.Names}}\t{{.Ports}}'"; \
		echo "   Pick different host ports for THIS stack (the container side does not move):"; \
		echo "     make env      # creates .env"; \
		echo "     \$$EDITOR .env  # uncomment and change HOST_NOVNC_PORT / HOST_VNC_PORT / ..."; \
		echo "     make ports    # re-check"; \
		exit 1; \
	fi

doctor:
	@echo "=== Toolchain ==="
	@printf '  %-16s %s\n' "python"  "$(PYTHON) ($$($(PYTHON) -V 2>&1))"
	@printf '  %-16s %s\n' "jax"     "$$($(PYTHON) -c 'import jax; print(jax.__version__)' 2>/dev/null || echo 'NOT INSTALLED — run: make install')"
	@printf '  %-16s %s\n' "flax"    "$$($(PYTHON) -c 'import flax; print(flax.__version__)' 2>/dev/null || echo 'NOT INSTALLED — run: make install')"
	@printf '  %-16s %s\n' "brax"    "$$($(PYTHON) -c 'import brax; print(brax.__version__)' 2>/dev/null || echo 'NOT INSTALLED — run: make install')"
	@printf '  %-16s %s\n' "docker"  "$$(docker --version 2>/dev/null || echo 'not found')"
	@printf '  %-16s %s\n' "compose" "$$(docker compose version --short 2>/dev/null || echo 'not found')"
	@printf '  %-16s %s\n' "bazel"   "$$(bazel --version 2>/dev/null || echo 'not found')"
	@echo "=== Container ==="
	@printf '  %-16s %s\n' "project" "$(COMPOSE_PROJECT_NAME)"
	@r="$$($(COMPOSE_BASE) ps --status running --format '{{.Name}}' 2>/dev/null | tr '\n' ' ')"; \
		printf '  %-16s %s\n' "running" "$${r:-none (make docker-up)}"
	@$(MAKE) --no-print-directory ports

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-hooks:
	@chmod +x tools/hooks/pre-commit
	@mkdir -p .git/hooks
	@cp tools/hooks/pre-commit .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@if command -v pre-commit >/dev/null 2>&1; then pre-commit install; fi
	@echo "✅ Git pre-commit hooks installed."

# ------------------------------------------------------------------- quality

# LINT.IfChange(test_discovery)
# Two discovery patterns. tests/test_diffusion_grpo.py is named the pytest way and
# matches neither `*_test.py` nor (previously) any Bazel target, so the GRPO and
# diffusion-policy suite -- the largest test file in the repo -- ran nowhere.
test:
	$(RUN_PY) -m unittest discover -s tests -p "*_test.py" -v
	$(RUN_PY) -m unittest discover -s tests -p "test_*.py" -v
# LINT.ThenChange(//.github/workflows/ci.yml:test_discovery, //tools/ci_local.sh, //tests/BUILD.bazel)

test-all: test

lint: check-ifttt
	@echo "🔍 Ruff..."
	@$(PYTHON) -m ruff check .
	@echo "🔍 Flake8..."
	@$(PYTHON) -m flake8 twm/ scripts/ tests/ tools/ --count --max-line-length=100 \
		--extend-ignore=E501,E203,W503 --statistics

# No `|| true` here: a formatter that cannot run is a real problem, and swallowing
# its exit status meant `make format` reported success while changing nothing.
format:
	@echo "✨ Formatting with Ruff & Black..."
	@$(PYTHON) -m ruff format .
	@$(PYTHON) -m black . --line-length=100

check-ifttt:
	@echo "🔍 IFTTT cross-file directive validator..."
	@$(PYTHON) tools/hooks/check_ifttt.py

ci-local:
	@./tools/ci_local.sh

# --------------------------------------------------------------------- bazel

# Bazel runs inside the container, like every other build target. rules_python's
# py_binary/py_test pull in a C++ toolchain for the interpreter stub, so Bazel
# needs a working `cc` -- which this host does not have (`command -v gcc` is
# empty), while the image installs build-essential. Running it on the host fails
# with "Auto-Configuration Error: Cannot find gcc or CC". Keeping both sides on
# one interpreter also stops the host and the container overwriting each other's
# .bazel/ convenience symlinks in the shared bind mount.
bazel-build:
	$(IN_CONTAINER) "bazel build //..."

bazel-test:
	$(IN_CONTAINER) "bazel test //..."

# Only this repo's Bazel state, and only inside the container where it lives.
# Never `bazel clean --expunge` against a shared output base, which would throw
# away every other project's cache on this machine.
bazel-clean:
	$(IN_CONTAINER) "bazel clean"

# Regenerate requirements_lock.txt from what the container actually resolved, so
# the Bazel build and the image can never drift apart.
lock-deps:
	@$(COMPOSE_BASE) exec -T $(SERVICE) bash -lc 'pip freeze --exclude-editable' \
		| grep -viE '^(twm|pkg[-_]resources)' | sort > /tmp/twm-freeze.txt
	@$(PYTHON) tools/write_lock.py /tmp/twm-freeze.txt requirements_lock.txt
	@echo "✅ requirements_lock.txt regenerated."

# ---------------------------------------------------------------- pipeline & experiments
# Configurable parameters for ANYmal World Model & Diffusion Policy Pipeline.
# Override on the command line, e.g.: make pipeline-anymal COLLECT_STEPS=5000 WM_TRAIN_STEPS=500
DATA_PATH         ?= data/anymal_trajectories.npz
WM_CHECKPOINT     ?= checkpoints/world_model_anymal.npz
POLICY_CHECKPOINT ?= checkpoints/diffusion_policy_anymal.npz
COLLECT_STEPS     ?= 2000
WM_TRAIN_STEPS    ?= 200
GRPO_ITERATIONS   ?= 25
EVAL_SAMPLES      ?= 20
EVAL_HORIZON      ?= 5

pipeline: pipeline-anymal

pipeline-anymal:
	@echo "================================================================="
	@echo "  🚀 Starting Full ANYmal World Model & Policy Pipeline"
	@echo "================================================================="
	@echo "  [Stage 1/5] Collecting exploratory ANYmal locomotion data..."
	$(RUN_PY) scripts/01_collect_data.py --env_name anymal_b --num_steps $(COLLECT_STEPS)
	@echo ""
	@echo "  [Stage 2/5] Training Causal Transformer World Model on ANYmal transitions..."
	$(RUN_PY) scripts/02_train_model.py --data_path $(DATA_PATH) --save_checkpoint $(WM_CHECKPOINT) --num_steps $(WM_TRAIN_STEPS)
	@echo ""
	@echo "  [Stage 3/5] Training Diffusion Policy in World Model Imagination (GRPO)..."
	$(RUN_PY) scripts/train_diffusion_grpo.py --config configs/grpo_diffusion_anymal.yaml \
		--num_iterations $(GRPO_ITERATIONS) --use_world_model True \
		--world_model_checkpoint $(WM_CHECKPOINT) --save_policy_checkpoint $(POLICY_CHECKPOINT) \
		--buffer_path $(DATA_PATH)
	@echo ""
	@echo "  [Stage 4/5] Closed-Loop MPPI Evaluation with World Model..."
	$(RUN_PY) scripts/03_evaluate_mppi.py --env_name anymal_b --num_samples $(EVAL_SAMPLES) --horizon $(EVAL_HORIZON)
	@echo ""
	@echo "  [Stage 5/5] Visualizing Trained Diffusion Policy Rollout & Kinematics..."
	$(RUN_PY) scripts/visualize_diffusion_policy.py --num_steps 150 \
		--html_out anymal_diffusion_walk.html --plot_out anymal_diffusion_kinematics.png --headless \
		--policy_checkpoint $(POLICY_CHECKPOINT)
	@echo ""
	@echo "================================================================="
	@echo "  ✅ Full ANYmal Locomotion Pipeline Completed Successfully!"
	@echo "  Generated Artifacts:"
	@echo "    - Trajectory Data:            $(DATA_PATH)"
	@echo "    - World Model Checkpoint:     $(WM_CHECKPOINT)"
	@echo "    - Diffusion Policy Checkpoint:$(POLICY_CHECKPOINT)"
	@echo "    - Kinematics Plot:            anymal_diffusion_kinematics.png"
	@echo "    - 3D Interactive Walk:        anymal_diffusion_walk.html"
	@echo "================================================================="

pipeline-smoke:
	@echo "================================================================="
	@echo "  ⚡ Running ANYmal Pipeline Quick Smoke Test"
	@echo "================================================================="
	@echo "  [Smoke 1/5] Collecting sample transitions (100 steps)..."
	$(RUN_PY) scripts/01_collect_data.py --env_name anymal_b --num_steps 100
	@echo ""
	@echo "  [Smoke 2/5] Training World Model for 20 steps..."
	$(RUN_PY) scripts/02_train_model.py --data_path $(DATA_PATH) --save_checkpoint $(WM_CHECKPOINT) --num_steps 20
	@echo ""
	@echo "  [Smoke 3/5] Running 3 GRPO imagination iterations..."
	$(RUN_PY) scripts/train_diffusion_grpo.py --config configs/grpo_diffusion_anymal.yaml \
		--num_iterations 3 --use_world_model True \
		--world_model_checkpoint $(WM_CHECKPOINT) --save_policy_checkpoint $(POLICY_CHECKPOINT) \
		--buffer_path $(DATA_PATH)
	@echo ""
	@echo "  [Smoke 4/5] Evaluating MPPI (10 samples, horizon 3)..."
	$(RUN_PY) scripts/03_evaluate_mppi.py --env_name anymal_b --num_samples 10 --horizon 3 --eval_steps 3
	@echo ""
	@echo "  [Smoke 5/5] Generating policy rollout visualization (30 steps)..."
	$(RUN_PY) scripts/visualize_diffusion_policy.py --num_steps 30 \
		--html_out anymal_diffusion_walk.html --plot_out anymal_diffusion_kinematics.png --headless \
		--policy_checkpoint $(POLICY_CHECKPOINT)
	@echo ""
	@echo "  ✅ ANYmal Pipeline smoke test completed cleanly!"

train-anymal-wm:
	$(RUN_PY) scripts/02_train_model.py --data_path $(DATA_PATH) --save_checkpoint $(WM_CHECKPOINT) --num_steps $(WM_TRAIN_STEPS)

train-wm: train-anymal-wm

collect-data:
	$(RUN_PY) scripts/01_collect_data.py --num_steps 100 --seq_len 32

train:
	$(RUN_PY) scripts/02_train_model.py --num_steps 50 --batch_size 16

evaluate:
	$(RUN_PY) scripts/03_evaluate_mppi.py --num_samples 50 --horizon 10 --eval_steps 5

visualize:
	$(RUN_PY) notebooks/01_visualize_rollouts.py

# Only the two recipes that actually name an environment are guarded. The block
# used to wrap all seven targets while naming four partner files, so bumping
# --num_steps on collect-data demanded staging brax_wrapper.py and three scripts.
# The realistic response to that is `git commit --no-verify`, which also skips
# formatting and linting -- a guard nobody can satisfy is worse than no guard.
# LINT.IfChange(env_targets)
collect-anymal:
	$(RUN_PY) scripts/01_collect_data.py --env_name anymal_b --num_steps 50

evaluate-anymal:
	$(RUN_PY) scripts/03_evaluate_mppi.py --env_name anymal_b --num_samples 20 --horizon 5
# LINT.ThenChange(//twm/envs/brax_wrapper.py:env_registry, //scripts/01_collect_data.py:env_args, //scripts/03_evaluate_mppi.py:env_args)

visualize-anymal:
	$(RUN_PY) scripts/visualize_anymal.py --num_steps 200 --pause_sec 60.0

train-grpo:
	$(RUN_PY) scripts/train_diffusion_grpo.py --config configs/grpo_diffusion_anymal.yaml \
		--num_iterations $(GRPO_ITERATIONS) --use_world_model True \
		--world_model_checkpoint $(WM_CHECKPOINT) --save_policy_checkpoint $(POLICY_CHECKPOINT) \
		--buffer_path $(DATA_PATH)

visualize-grpo:
	$(RUN_PY) scripts/visualize_diffusion_policy.py --num_steps 150 \
		--html_out anymal_diffusion_walk.html --plot_out anymal_diffusion_kinematics.png --headless \
		--policy_checkpoint $(POLICY_CHECKPOINT)

clean-pipeline:
	rm -f $(DATA_PATH) $(WM_CHECKPOINT) $(POLICY_CHECKPOINT)
	rm -f anymal_diffusion_walk.html anymal_diffusion_kinematics.png
	@echo "🧹 Cleaned pipeline data, checkpoints, and visualization artifacts."

# --port is the port INSIDE the container; the host mapping is HOST_TENSORBOARD_PORT.
tensorboard:
	@echo "📈 TensorBoard: $(TB_URL)"
	$(IN_CONTAINER) "tensorboard --logdir runs --port 6006 --bind_all"

# ------------------------------------------------------------------ container

docker-build:
	$(HOST_ONLY)
	$(COMPOSE_BASE) build

docker-up: check-ports
	$(HOST_ONLY)
	$(COMPOSE_BASE) up -d
	@echo ""
	@echo "=========================================================="
	@echo "🚀 Project '$(COMPOSE_PROJECT_NAME)' is up."
	@echo "🌐 noVNC:       $(VNC_URL)"
	@echo "🖥️  VNC client:  $(BIND_ADDR):$(HOST_VNC_PORT)"
	@echo "🐚 Shell:       make docker-shell"
	@echo "=========================================================="

docker-up-gpu: check-ports
	$(HOST_ONLY)
	$(COMPOSE_GPU) up -d
	@echo "✅ Project '$(COMPOSE_PROJECT_NAME)' is up with one NVIDIA GPU attached."
	@echo "🌐 noVNC: $(VNC_URL)"

# --remove-orphans clears containers left behind by an earlier service name in THIS
# project only; it never touches another project's containers.
docker-down:
	$(HOST_ONLY)
	$(COMPOSE_BASE) down --remove-orphans

docker-restart: docker-down docker-up

docker-shell:
	$(HOST_ONLY)
	$(COMPOSE_BASE) exec $(SERVICE) bash

docker-logs:
	$(HOST_ONLY)
	$(COMPOSE_BASE) logs -f --tail=200

docker-status:
	$(HOST_ONLY)
	@echo "=== Containers (project '$(COMPOSE_PROJECT_NAME)') ==="
	@$(COMPOSE_BASE) ps || true
	@echo "=== Volumes ==="
	@docker volume ls --filter "label=com.docker.compose.project=$(COMPOSE_PROJECT_NAME)" || true
	@echo "=== Networks ==="
	@docker network ls --filter "label=com.docker.compose.project=$(COMPOSE_PROJECT_NAME)" || true

# Scoped to this Compose project: its containers, its network, its named volumes.
# Deliberately not `docker system prune`, which would delete other projects' images,
# build cache and volumes as well.
docker-clean:
	$(HOST_ONLY)
	@echo "⚠️  Removing containers, network and named volumes for project '$(COMPOSE_PROJECT_NAME)'."
	@echo "   The bind-mounted working tree is untouched; the container's home"
	@echo "   (pip cache, Bazel output base, shell history) is discarded."
	$(COMPOSE_BASE) down --volumes --remove-orphans

# ----------------------------------------------------------------------- vnc

# `start`, not the bare form: the bare form is the image CMD and ends in
# `exec tail -f /dev/null`, which would wedge this terminal instead of returning.
# HOST_*_PORT are passed through so the script prints the host URL, not its own
# internal port. They are also in the service environment, but not when someone
# runs this target from inside the container against a hand-started shell.
start-vnc:
	$(IN_CONTAINER) "HOST_NOVNC_PORT=$(HOST_NOVNC_PORT) HOST_VNC_PORT=$(HOST_VNC_PORT) /usr/local/bin/start_vnc.sh start"
	@echo "🖥️  noVNC: $(VNC_URL)"

stop-vnc:
	$(IN_CONTAINER) "/usr/local/bin/start_vnc.sh stop"

vnc-status:
	$(IN_CONTAINER) "/usr/local/bin/start_vnc.sh status"

# A zombie has already exited; only its parent can reap it, so kill/pkill cannot
# clear one. docker-compose.yml sets `init: true` so tini becomes PID 1 and reaps
# the Xvfb/x11vnc/websockify processes that start_vnc.sh orphans on restart.
check-zombies:
	-@$(IN_CONTAINER) 'z=$$(ps -eo stat --no-headers 2>/dev/null | awk "/^Z/ {n++} END {print n+0}"); \
		if [ "$$z" -eq 0 ]; then echo "✅ No zombie processes."; \
		else echo "⚠️  $$z zombie process(es); only their parent can reap them. If PID 1 holds them: make docker-restart"; fi' 2>/dev/null || true

# ------------------------------------------------------------------- cleanup

clean:
	find . -type d -name "__pycache__" -not -path "./third_party/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -not -path "./third_party/*" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache

clean-artifacts: clean clean-pipeline
	rm -f rollout_comparison.png anymal_kinematics.png anymal_diffusion_kinematics.png
	rm -f anymal_rollout.html anymal_diffusion_walk.html
