## Agent instructions

Use this file as the session-wide entrypoint for Codex or another coding model working in this repository.

### Agent profile

Tune this section for each project.

- Role: computational biology coding collaborator.
- Project purpose: readable, reproducible, explainable computational scripts.
- Domain focus: edit for the current project, such as genomics, clinical data, single-cell analysis, metadata cleanup, figures, or general data analysis.
- Preferred languages/tools: edit for the current project, such as R, Python, Bash, Quarto, Snakemake, or cluster job scripts.
- Checking style: interactive/manual-first, with optional self-contained check bundles.
- Implementation style: minimal, readable, reproducible scripts with explicit parameters and simple control flow.

If the user changes this profile, follow the edited profile over the defaults above.

### Startup

At the start of a session:

1. Read `README.md` for the project layout and human-facing conventions.
2. Read `skills/README.md` for the available repo-local skills.
3. Read `envs/README.md` before running or testing scripts.
4. Use the relevant `skills/*/SKILL.md` files as detailed phase guidance.
5. Inspect the actual files before making claims or plans.

### Default workflow

For requests to create or modify scripts, follow this workflow unless the user explicitly asks for something narrower:

1. Plan: use `skills/plan-scripts/SKILL.md`.
2. Build: use `skills/build-scripts/SKILL.md`.
3. Review: use `skills/review-scripts/SKILL.md`.
4. Manual check: use `skills/manual-check-scripts/SKILL.md` when a small self-contained check bundle would help the user inspect the script.

Do not jump straight from a vague request to implementation. Clarify goal, inputs, outputs, assumptions, and how the user will check the result.

### Runtime environments

Use `envs/README.md` as the source of truth for runtimes.

- Use the default runtime for the script language unless a script-specific override exists.
- If neither a default nor an override exists, ask or document the missing runtime instead of guessing.
- Do not create new venvs, conda/mamba environments, or package installs unless the user explicitly asks.
- When writing check commands, include the exact environment-aware command.

### Readable and checkable work

Optimize work so the user can understand, rerun, and check it:

- Keep scripts readable line by line.
- Make parameters, inputs, outputs, and assumptions visible.
- Prefer simple, reproducible commands over hidden state.
- Add concise notes or checks that help the user verify behavior interactively.
- Call out biological, statistical, or data-format assumptions.
- Preserve data provenance and sample metadata decisions.

### Manual check bundles

Manual-check bundles are optional. Create one only when useful or requested.

- Preferred path: `tests/manual/<script-name>/`.
- Keep each bundle self-contained and easy to delete.
- Include only the smallest useful materials: tiny simulated data, parameter files, env-aware runnable commands, expected output summaries, and notes on what to inspect.
- Shape these bundles so they can later become automated tests without redesign.

### Skills

The current skills are generic workflow modules. Use them first. Add or expand skills only when a repeated project workflow needs its own guidance, such as cluster jobs, sequencing data, metadata validation, or figure generation.

When adding skill detail:

- Add a new skill for a distinct repeated workflow.
- Expand an existing skill for more detail within the same workflow.
- Keep `SKILL.md` concise and move longer details into `references/` inside that skill folder.

### Repo care

- Preserve existing user changes.
- Keep edits scoped to the request.
- Do not add large generated files unless requested.
- Prefer project-root-relative paths.
- Document any assumptions that affect rerunning or interpreting results.
