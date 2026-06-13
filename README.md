## Project template

A small directory structure template for computational projects. It is intended to stay simple enough for GitHub, cluster work, and different workflow tools.

### Starting with an agent

When starting a new project or model session, point Codex or another coding model at `AGENTS.md` once. That file is the top-level instruction entrypoint: it sets a tunable role, explains the default workflow, and routes the model to the repo-local skills.

The default agent role is a computational biology coding collaborator. Edit the profile in `AGENTS.md` for the current project if the domain, tools, or checking style should change.

### Layout

- `scripts/`: high-level runnable scripts for analyses, data preparation, and result generation.
- `utils/`: reusable helper code used by scripts.
- `data-raw/`: large, private, or external source data that should usually stay immutable and out of version control.
- `data/`: small versionable inputs, manifests, symlinks (e.g. to data-raw/), or lightweight examples arranged so scripts can run from the project root.
- `results/`: generated analysis outputs, preferably split into one subdirectory per task or experiment.
- `docs/`: project documentation.
- `envs/`: environment files and setup notes.
- `skills/`: repo-local instructions for Codex or other models that help plan, build, review, and manually check scripts.

### Skills

The skills are here to keep agent-assisted script work planned, readable, reviewable, and easy to check.

- `skills/plan-scripts/SKILL.md`: force a clear plan before code, including purpose, inputs, outputs, assumptions, and how the result will be checked.
- `skills/build-scripts/SKILL.md`: guide agents to write readable, reproducible scripts with explicit parameters and simple control flow.
- `skills/review-scripts/SKILL.md`: review scripts for correctness, reproducibility, assumptions, paths, edge cases, and remaining risks.
- `skills/manual-check-scripts/SKILL.md`: optionally create small per-script check bundles so you can inspect behavior interactively.

### Runtime environments

Document runtime environments in `envs/README.md`. Define defaults by language once, such as R, Python, Bash, or Quarto, then add script-specific overrides only when a script does not use the default.

Agents should use the documented runtime when running or checking scripts. They should not create new environments unless explicitly asked.

### Manual checks

For scripts that benefit from interactive checking, create an optional folder:

```text
tests/manual/<script-name>/
├── README.md
├── input.*
├── params.<ext>
├── run-command.txt
└── expected-summary.txt
```

- `README.md`: what to run, inspect, delete, or recreate.
- `input.*`: tiny fake or sample input.
- `params.<ext>`: copy-pasteable parameter values for interactive testing.
- `run-command.txt`: full env-aware command to rerun the script.
- `expected-summary.txt`: facts to compare by eye, such as row counts, column names, output files, or summary values.

Manual checks are manual first. They can become automated tests later because they already define fixed inputs, fixed parameters, and expected behavior. A future project can add a language-specific test file, such as `test_<script-name>.<ext>`, that reads the same bundle and asserts the expected facts.

Do not add GitHub Actions to this template by default. Once a project has real automated tests, add a GitHub Actions workflow to run them on push or pull request.

### Requirements

- Keep the template amenable to cluster work.
- Keep the template amenable to version control, including GitHub.
- Keep the layout portable across platforms and workflow management tools where possible.
