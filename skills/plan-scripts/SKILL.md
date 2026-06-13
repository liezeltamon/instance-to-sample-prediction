---
name: plan-scripts
description: Use when planning new or changed project scripts before writing code. Requires repo inspection, clear success criteria, scope, inputs, outputs, and manual-check strategy before implementation.
---

# Plan Scripts

Use this skill before creating or changing scripts in this repository.

## Purpose

Make the approach explainable before any code is written. The user should be able to describe the goal, inputs, outputs, assumptions, and validation strategy in plain language.

## Workflow

1. Inspect the repo first: read the relevant README files, existing scripts, data layout, and any task-specific docs.
2. State the user goal, expected audience, and what "done" means.
3. Identify the script location, inputs, outputs, dependencies, and assumptions.
4. Decide what should be versioned and what should stay local or ignored.
5. Define how the user can manually check the result after the script runs.
6. Produce a plan before editing files.

## Plan should include

- Script purpose and proposed path.
- Required inputs and where they should live.
- Outputs and where they should be written.
- Minimal manual-check files or examples to create.
- Commands to run the script and verify results.
- Risks, missing details, or choices that need user confirmation.

Do not build straight from a vague request. If intent is unclear after repo inspection, ask concise questions before implementation.
