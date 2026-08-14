# Transformer World Model with MPPI Control in JAX

[Work in progress]

Implementation of an **Auto-Regressive Causal Transformer World Model** controlled via **Model Predictive Path Integral (MPPI)** planning, built entirely in **JAX**, **Flax (NNX API)**, **Optax**, and **Brax**, structured with a **Bazel build system**.

---

## 🌟 Architectural Overview

```text
               +-------------------------------------------------------+
               |                  Brax Physics Engine                  |
               |                (GPU-Accelerated Rigids)               |
               +-------------------------------------------------------+
                                   |                 ^
                     State (s_t)   |                 | Action (a_t*)
                                   v                 |
  +-------------------------------------------------------------------------+
  |                             MPPI Planner                                |
  |   (1000 Trajectory Samples, jax.lax.scan Auto-regressive Rollouts)      |
  +-------------------------------------------------------------------------+
                                   |
                         Imagined Rollouts (s, a)
                                   v
  +-------------------------------------------------------------------------+
  |                    Causal Transformer World Model                       |
  |   - Tokenizer: Interleaved continuous (s_t, a_t) embedding tokens       |
  |   - Causal Self-Attention: Strict lower-triangular causal masking       |
  |   - Dynamics Head: Next-state delta (s_{t+1}), reward (r_t), discount    |
  +-------------------------------------------------------------------------+
```

---

## 🛠️ Repository Structure & Bazel Targets

```text
transformer_world_model/
├── MODULE.bazel              # Bzlmod Python dependencies
├── .bazelrc                  # Google3-style Bazel configuration
├── BUILD.bazel               # Root BUILD target aliases
├── Dockerfile                # Headless Ubuntu container with Bazelisk & VNC
├── docker-compose.yml        # Compose service mapping VNC (5900) & noVNC (6080)
├── configs/                  # Environment & Model Yaml hyperparams
│   ├── env_brax_ant.yaml
│   └── model_twm_base.yaml
├── twm/                      # Main Package (BUILD.bazel: //twm:*)
│   ├── envs/                 # Brax wrappers and continuous tokenization
│   ├── models/               # Flax NNX Transformer & prediction heads
│   ├── planner/              # MPPI planner with jax.lax.scan
│   └── utils/                # Replay buffer & PRNG key management
├── scripts/                  # Milestone Executable Binaries (BUILD.bazel: //scripts:*)
│   ├── 01_collect_data.py    # Milestone 1: Brax exploration & buffer sampling
│   ├── 02_train_model.py     # Milestones 2 & 3: JIT-compiled training loop
│   ├── 03_evaluate_mppi.py   # Milestone 4: MPPI controller rollouts
│   └── start_vnc.sh          # TurboVNC / Openbox / noVNC daemon
├── tests/                    # Bazel Unit Test Suite (BUILD.bazel: //tests:*)
│   ├── env_test.py
│   ├── model_test.py
│   └── mppi_test.py
└── third_party/              # Reference Submodules
    ├── brax/                 # google/brax
    ├── dreamerv3/            # danijar/dreamerv3
    ├── flax/                 # google/flax
    └── walk_in_the_park/     # ikostrikov/walk_in_the_park
```

---

## 🚀 Quickstart & Commands

### Using Make

```bash
make help              # Show available make commands
make test              # Run unit test suite
make collect-data      # Milestone 1: Collect Brax trajectories (Ant)
make collect-anymal    # Milestone 1: Collect trajectories (ANYmal B quadruped)
make train             # Milestones 2 & 3: JIT-compiled model training loop
make evaluate          # Milestone 4: Closed-loop MPPI planner (Ant)
make evaluate-anymal   # Milestone 4: Closed-loop MPPI planner (ANYmal B)
make visualize         # Plot real vs imagined trajectory rollouts
make visualize-anymal  # Launch ANYmal B visualizer on VNC & generate 3D HTML
make docker-up         # Launch container with VNC server
```

### Using Bazel

```bash
# Build all targets
bazel build //...

# Run full test suite
bazel test //...

# Execute Milestone 1: Collect Data
bazel run //scripts:01_collect_data

# Execute Milestone 2 & 3: Train Transformer Model
bazel run //scripts:02_train_model

# Execute Milestone 4: MPPI Planning Evaluation
bazel run //scripts:03_evaluate_mppi
```

---

## 🐳 Running in Docker with VNC Forwarding

### 1. Launch Docker Container

```bash
docker compose up -d --build
docker exec -u devuser -it transformer_world_model_dev bash
# or using Makefile:
make docker-shell
```

### 2. Connect to VNC / noVNC Web Desktop

- **Web Browser (noVNC)**: Open **[http://localhost:6080/vnc.html](http://localhost:6080/vnc.html)** (or `http://localhost:6080/`) in your browser.
- **Desktop VNC Client**: Connect your VNC viewer (e.g., TigerVNC, RealVNC) to `localhost:5900` (Password: none).

If connecting from a remote machine over SSH:
```bash
ssh -L 5900:localhost:5900 -L 6080:localhost:6080 user@remote-host-ip
```

---

## 🔄 Continuous Integration (GitHub Actions)

The repository includes automated CI workflows in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) that run on every push and pull request:
1. **Build Repo & Unit Tests**: Validates Python & Bazel dependencies, runs all automated unit tests, and verifies dry runs of milestone scripts.
2. **Docker Build & Container Verification**: Automatically builds the Docker image and runs unit tests inside the container environment.

### Run Local CI Verification:
Before pushing commits, you can run the exact same checks locally:
```bash
./tools/ci_local.sh
# or using Makefile:
make ci-local
```

---

## 📚 Core Milestone Reference Papers

1. **Transformer World Models**: *IRIS: Transformers are Sample-Efficient World Models* (Alonso et al., 2023).
2. **MPPI Controller**: *Information Theoretic Model Predictive Control* (Williams et al., 2017).
3. **Physics Engine**: *Brax - A Differentiable Physics Engine for Large Scale Rigid Body Simulation* (Freeman et al., 2021).
4. **Robot Deployment**: *DayDreamer: World Models for Physical Robot Learning* (Wu et al., 2022).
