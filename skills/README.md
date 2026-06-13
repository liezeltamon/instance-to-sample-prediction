## Repo-local skills

This directory contains repo-local instructions for Codex or other coding models. These skills help agents produce scripts that are planned, readable, reviewable, and easy to check.

Use `AGENTS.md` as the top-level session entrypoint. It decides how these generic skills should be used for the current project through its editable agent profile.

### Why these skills exist

The goal is not just working code. The goal is script work with visible assumptions, clear inputs and outputs, and a simple way to verify the result.

- Plans should make reasoning and assumptions visible.
- Scripts should be readable line by line.
- Reviews should surface correctness and reproducibility risks.
- Manual checks should make generated or modified scripts easy to inspect.

### Using these skills

- Start with `plan-scripts` before asking an agent to write or change scripts. Its purpose is to slow the work down enough that the approach is explainable.
- Use `build-scripts` when the agent is ready to create or edit script files. Its purpose is to keep code explicit, reproducible, and easy to inspect.
- Use `review-scripts` before accepting generated scripts. Its purpose is to check correctness, assumptions, paths, and risks.
- Use `manual-check-scripts` when the agent should add optional per-script checking materials. Its purpose is to help you run and inspect a script interactively.

If an agent reads `AGENTS.md`, you should not need to point to each skill manually every time. You can still mention a skill by path for targeted work, for example: "Use `skills/plan-scripts/SKILL.md` before making changes."

### Manual-check bundles

Optional manual-check bundles should live in `tests/manual/<script-name>/`.

Use them when a script would be easier to inspect with tiny simulated data, explicit parameter values, a runnable command, expected output summaries, or notes on what to check. Keep each bundle self-contained and easy to delete. Shape it so it can later become an automated test if useful.

### Expanding skills

The current skills are generic workflow modules. Add project-specific skills only when a repeated workflow needs more detail, such as database access, cluster jobs, sequencing data, metadata validation, or figure generation.

Expand an existing skill when the new guidance is part of the same workflow, such as adding script naming rules to `build-scripts` or more review checks to `review-scripts`.

Keep each `SKILL.md` short and action-oriented. If details become long, add `references/` inside that skill folder and link to the relevant file from `SKILL.md`.

Each skill should use this minimal structure:

```text
skills/<skill-name>/
└── SKILL.md
```

Use only plain Markdown unless a future task needs bundled examples, scripts, or reference files.
