.PHONY: help install test collect-data train evaluate visualize docker-build docker-up docker-down docker-shell bazel-build bazel-test clean

PYTHON ?= /home/nico-palomo/workspace/venv/bin/python
ifeq ($(wildcard $(PYTHON)),)
	PYTHON := python3
endif

help:
	@echo "Available Makefile commands:"
	@echo "  make install             Install Python dependencies into current virtualenv"
	@echo "  make test                Run unit test suite"
	@echo "  make ci-local            Run complete CI workflow locally"
	@echo "  make collect-data        Run Milestone 1: Brax data collection (Ant)"
	@echo "  make collect-anymal      Collect data from ANYmal B quadruped into buffer"
	@echo "  make train               Run Milestones 2 & 3: Train Transformer World Model"
	@echo "  make evaluate            Run Milestone 4: Closed-loop MPPI evaluation (Ant)"
	@echo "  make evaluate-anymal     Run Milestone 4: Closed-loop MPPI evaluation on ANYmal B"
	@echo "  make visualize           Generate rollout comparison plot (real vs imagined)"
	@echo "  make visualize-anymal    Launch ANYmal B simulation on VNC display & generate 3D HTML"
	@echo "  make bazel-build         Build all targets using Bazel"
	@echo "  make bazel-test          Run tests using Bazel"
	@echo "  make docker-build        Build Docker development image"
	@echo "  make docker-up           Launch Docker container in background (VNC on 5900, web on 6080)"
	@echo "  make docker-down         Stop and remove Docker container"
	@echo "  make docker-shell        Open bash shell inside running Docker container"
	@echo "  make clean               Clean pycache, temporary files, and plot artifacts"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	PYTHONPATH=. $(PYTHON) -m unittest discover -s tests -p "*_test.py"

ci-local:
	./tools/ci_local.sh

collect-data:
	PYTHONPATH=. $(PYTHON) scripts/01_collect_data.py --num_steps 100 --seq_len 32

collect-anymal:
	PYTHONPATH=. $(PYTHON) scripts/01_collect_data.py --env_name anymal_b --num_steps 50

train:
	PYTHONPATH=. $(PYTHON) scripts/02_train_model.py --num_steps 50 --batch_size 16

evaluate:
	PYTHONPATH=. $(PYTHON) scripts/03_evaluate_mppi.py --num_samples 50 --horizon 10 --eval_steps 5

evaluate-anymal:
	PYTHONPATH=. $(PYTHON) scripts/03_evaluate_mppi.py --env_name anymal_b --num_samples 20 --horizon 5

visualize:
	PYTHONPATH=. $(PYTHON) notebooks/01_visualize_rollouts.py

visualize-anymal:
	PYTHONPATH=. $(PYTHON) scripts/visualize_anymal.py --num_steps 30

bazel-build:
	bazel build //...

bazel-test:
	bazel test //...

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-shell:
	docker exec -u devuser -it transformer_world_model_dev bash

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f rollout_comparison.png
