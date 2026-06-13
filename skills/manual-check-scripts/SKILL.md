---
name: manual-check-scripts
description: Use when creating the minimal script-adjacent files, sample data, expected outputs, manifests, or notes needed for a user to manually verify generated scripts.
---

# Manual Check Scripts

Use this skill when a script needs small supporting artifacts so the user can inspect behavior by hand.

## Purpose

Create optional, self-contained materials that let the user go straight to interactive checking after a script is generated or modified. The goal is to make behavior inspectable line by line and easy to convert into automated tests later.

## Create only what is necessary

When useful or requested, create a per-script bundle under `tests/manual/<script-name>/`.

Add the smallest useful artifacts, such as:

- Tiny simulated or sample input data.
- A parameter file showing concrete, copy-pasteable argument values.
- A `run-command.txt` file with the exact environment-aware command.
- A manifest describing expected local files or symlinks.
- A small expected-output file or summary facts for comparison.
- A short note explaining what to inspect interactively.

## Manual-check guidance

For each generated script, make it clear:

- What command to run, using the default runtime from `envs/README.md` unless a script-specific override exists.
- Which input files are required.
- Which output files should appear.
- Which parameters or arguments are being tested.
- What small facts the user should compare by eye, such as row counts, column names, file names, or summary values.
- Which generated files are safe to delete and recreate.

Do not invent a new venv, conda/mamba environment, or interpreter path. If no default runtime or override is documented, note the missing runtime instead of guessing.

Keep each bundle self-contained and easy to delete. Avoid broad fixtures, large synthetic datasets, or extra documentation that is not needed to check the script.

Shape manual-check bundles so they can later become automated tests without redesign.
