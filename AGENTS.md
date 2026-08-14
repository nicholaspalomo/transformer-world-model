# Coding Agent Style Guidelines & Rule Directives

## 📐 Code Style & Architecture (Google3 Standards)
- **Formatting**: Code is formatted using Google Python Style Guide conventions via Ruff, Black (line length 100), and YAPF.
- **Modularity**: Follow Google3 Bazel structure with explicit `BUILD.bazel` target definitions (`py_library`, `py_binary`, `py_test`).
- **JAX Purity**: All physics rollouts, attention blocks, and tokenizer mappings should remain functional and pure, avoiding un-jitted device array mutations inside Python loops.

---

## 🔗 Co-Dependent Changes: Google IFTTT Directives (`LINT.IfChange` / `LINT.ThenChange`)

When code or configuration in one place must stay in sync with code elsewhere (for example: adding a new robot environment, altering observation/action dimensions, updating token projection layers, or modifying replay buffer shapes), mark the dependency with `LINT.IfChange` / `LINT.ThenChange` directives so changes to one side prompt review and updates to the other.

Add these directives proactively when creating new co-dependent content, not just when maintaining existing pairs.

### Guidelines
1. Keep the guarded block as small as possible. Prefer several small labeled source->target pairs over one large catch-all block unless the whole region genuinely needs to change together.
2. Target paths should be formatted with repository-relative `//path/to/file:label` or `//path/to/file`.
3. Directives are enforced automatically via `tools/hooks/check_ifttt.py`, git pre-commit hooks, and CI.

### Example

```python
# LINT.IfChange(env_registry)
if env_name == "my_new_robot":
    from twm.envs.my_robot_env import MyRobotEnv

    self._env = MyRobotEnv()
# LINT.ThenChange(//twm/envs/my_robot_env.py:env_specs, //configs/env_my_robot.yaml:env_config, //Makefile:env_targets)
```

---

## 🛠️ Linters, Formatting & Pre-Commit Hooks

```bash
# 1. Run Google IFTTT cross-file directive validator
make check-ifttt

# 2. Run Google3 & PEP 8 Linters (Ruff & Flake8)
make lint

# 3. Format source files (Ruff & Black)
make format

# 4. Install Git Pre-Commit Hooks
make install-hooks

# 5. Run complete local CI emulator
make ci-local
```
