---
name: build-scripts
description: Use when creating or editing project scripts that should be reproducible, readable, and easy for a user to manually inspect after generation.
---

# Build Scripts

Use this skill when implementing scripts in this repository.

## Purpose

Create scripts that are easy to read, rerun, and explain line by line. Prefer explicit parameters, simple control flow, and visible assumptions so the user can inspect the implementation.

## Script style

- Put high-level runnable scripts in `scripts/`.
- Put reusable helper code in `utils/`.
- Keep paths relative to the project root unless the user asks otherwise.
- Make inputs, outputs, and defaults visible near the top of the script.
- Prefer deterministic, idempotent behavior: rerunning should not silently corrupt or duplicate outputs.
- Write clear status messages for major steps.
- Keep comments short and useful, explaining non-obvious choices.
- Avoid hidden external state; document required environment variables, local files, and runtime assumptions.
- Use the default runtime from `envs/README.md` unless the script needs a documented override.

## Data and outputs

- Read large or private raw files from `data-raw/`.
- Use `data/` for small versionable inputs, manifests, symlinks, or lightweight examples.
- Write analysis outputs under `results/`, preferably in a task-specific subdirectory.
- Do not add large generated files unless the user explicitly requests them.

## Manual inspectability

Make scripts easy to check by hand:

- Use small, named intermediate outputs when helpful.
- Include row counts, file counts, or basic summaries in logs or output notes.
- Fail with clear messages when required inputs are missing.
- Prefer simple control flow over clever abstractions.
- Add the smallest example input needed to demonstrate behavior only when requested or needed for checking.
