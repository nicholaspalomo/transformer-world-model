#!/usr/bin/env python3
"""
Python validator for Google LINT.IfChange / LINT.ThenChange directives.
Used in git pre-commit hooks and local CI validation.
Matches directives appearing alone on comment lines across Python, Shell, YAML, Markdown, C++, XML, etc.
"""

import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Directives must be alone on comment lines (e.g. '# LINT.IfChange(...)', '// LINT.IfChange(...)', '<!-- LINT.IfChange(...) -->')
IF_CHANGE_RE = re.compile(r"^\s*(?:#|//|<!--|--|;|\*)\s*LINT\.IfChange(?:\(([A-Za-z0-9_\-\.]+)\))?")
THEN_CHANGE_RE = re.compile(r"^\s*(?:#|//|<!--|--|;|\*)\s*LINT\.ThenChange\(([^)]+)\)")


def parse_targets(target_str):
    targets = []
    for item in target_str.split(","):
        item = item.strip()
        if not item:
            continue
        if item.startswith("//"):
            targets.append(item[2:])
        elif item.startswith(":"):
            targets.append(item)
        else:
            targets.append(item)
    return targets


def check_file(rel_path):
    full_path = os.path.join(REPO_ROOT, rel_path)
    if not os.path.exists(full_path) or os.path.isdir(full_path):
        return []

    errors = []
    try:
        with open(full_path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        return [f"{rel_path}: unable to read file: {e}"]

    if_stack = []
    seen_labels = set()
    in_markdown_code_block = False

    for idx, line in enumerate(lines, 1):
        if rel_path.endswith(".md") and line.strip().startswith("```"):
            in_markdown_code_block = not in_markdown_code_block
            continue

        if in_markdown_code_block:
            continue

        if_match = IF_CHANGE_RE.search(line)
        then_match = THEN_CHANGE_RE.search(line)

        if if_match:
            label = if_match.group(1)
            if label:
                if label in seen_labels:
                    errors.append(f"{rel_path}:{idx}: duplicate LINT.IfChange label '{label}'")
                seen_labels.add(label)
            if_stack.append((idx, label))

        if then_match:
            if not if_stack:
                errors.append(f"{rel_path}:{idx}: LINT.ThenChange without preceding LINT.IfChange")
            else:
                if_stack.pop()

            targets = parse_targets(then_match.group(1))
            for target in targets:
                if ":" in target:
                    t_file, t_label = target.split(":", 1)
                else:
                    t_file, t_label = target, None

                if t_file:
                    target_full_path = os.path.join(REPO_ROOT, t_file)
                    if not os.path.exists(target_full_path):
                        errors.append(f"{rel_path}:{idx}: target file not found '//{t_file}'")

    for unclosed_idx, unclosed_label in if_stack:
        lbl_str = f"({unclosed_label})" if unclosed_label else ""
        errors.append(
            f"{rel_path}:{unclosed_idx}: LINT.IfChange{lbl_str} without matching LINT.ThenChange"
        )

    return errors


def check_staged_ifttt():
    """Validates that if an IfChange block was edited, all ThenChange target files are staged."""
    import subprocess

    try:
        staged_files_proc = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        if staged_files_proc.returncode != 0 or not staged_files_proc.stdout.strip():
            return []
        staged_files = set(staged_files_proc.stdout.strip().splitlines())
    except Exception:
        return []

    errors = []
    # Check diff for each staged file
    for rel_path in staged_files:
        full_path = os.path.join(REPO_ROOT, rel_path)
        if not os.path.isfile(full_path):
            continue

        try:
            diff_proc = subprocess.run(
                ["git", "diff", "--cached", "-U0", rel_path],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )
            if diff_proc.returncode != 0 or not diff_proc.stdout:
                continue
            diff_text = diff_proc.stdout
        except Exception:
            continue

        # Extract modified line numbers in new file (from @@ -a,b +c,d @@)
        modified_lines = set()
        for line in diff_text.splitlines():
            if line.startswith("@@"):
                m = re.search(r"\+(\d+)(?:,(\d+))?", line)
                if m:
                    start = int(m.group(1))
                    count = int(m.group(2)) if m.group(2) is not None else 1
                    for l_num in range(start, start + count):
                        modified_lines.add(l_num)

        if not modified_lines:
            continue

        # Parse IfChange blocks in the file
        with open(full_path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        if_stack = []
        for idx, line in enumerate(lines, 1):
            if_match = IF_CHANGE_RE.search(line)
            then_match = THEN_CHANGE_RE.search(line)

            if if_match:
                if_stack.append((idx, if_match.group(1)))

            if then_match and if_stack:
                start_idx, label = if_stack.pop()
                end_idx = idx

                # Check if any modified line falls inside [start_idx, end_idx]
                block_modified = any(start_idx <= ml <= end_idx for ml in modified_lines)
                if block_modified:
                    targets = parse_targets(then_match.group(1))
                    for target in targets:
                        t_file = target.split(":", 1)[0] if ":" in target else target
                        if t_file and t_file not in staged_files:
                            lbl_msg = f" ({label})" if label else ""
                            errors.append(
                                f"❌ IFTTT Violation: '{rel_path}'{lbl_msg} was modified (lines {start_idx}-{end_idx}), "
                                f"but required partner file '//{t_file}' is not staged in this commit."
                            )

    return errors


def main():
    files = sys.argv[1:]
    if not files:
        # Scan all tracked files in repository
        files = []
        for root, _dirs, fnames in os.walk(REPO_ROOT):
            if (
                ".git" in root
                or "bazel-" in root
                or ".pytest_cache" in root
                or "third_party" in root
                or "node_modules" in root
            ):
                continue
            for fname in fnames:
                rel = os.path.relpath(os.path.join(root, fname), REPO_ROOT)
                files.append(rel)

    total_errors = []
    for rel_path in files:
        if (
            rel_path.startswith(".git")
            or rel_path.startswith("third_party")
            or rel_path.endswith(".png")
            or rel_path.endswith(".jpg")
            or rel_path.endswith(".stl")
            or rel_path.endswith(".obj")
        ):
            continue
        errs = check_file(rel_path)
        total_errors.extend(errs)

    # Check staged IFTTT cross-file synchronization
    staged_ifttt_errors = check_staged_ifttt()
    total_errors.extend(staged_ifttt_errors)

    if total_errors:
        print("❌ IFTTT Directives Lint Errors:")
        for err in total_errors:
            print(f"  {err}")
        sys.exit(1)
    else:
        print("✅ IFTTT Directives verified cleanly across all tracked files.")
        sys.exit(0)


if __name__ == "__main__":
    main()
