---
name: review-scripts
description: Use when reviewing generated or modified project scripts for correctness, reproducibility, assumptions, paths, edge cases, and manual verification steps.
---

# Review Scripts

Use this skill before calling a generated or changed script complete.

## Purpose

Review generated or modified scripts like real computational work: check whether the script is correct, reproducible, explainable, and honest about assumptions and risks.

## Review focus

- Confirm the script matches the planned goal and scope.
- Check that inputs, outputs, and paths match the repo layout.
- Look for hidden assumptions about local files, package versions, working directory, runtime environment, or environment variables.
- Confirm the script uses the default runtime from `envs/README.md` or has a documented script-specific override.
- Check whether rerunning the script is safe and predictable.
- Check whether errors are understandable when inputs are missing or malformed.
- Confirm large, private, or generated files are not accidentally added to version control.
- Verify the user has a clear way to inspect the result manually.

## Review output

Lead with issues that could affect correctness or reproducibility. Include file paths and line references when possible.

If no major issues are found, say so clearly and list any remaining manual checks or test gaps.

Do not rewrite the script during review unless the user explicitly asks for fixes.
