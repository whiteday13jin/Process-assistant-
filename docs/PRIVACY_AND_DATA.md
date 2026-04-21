# Data and Privacy Notes

## Goal

This project should be publishable to GitHub without leaking sensitive business information.

## Data Classification (Current Repo)

- `data/knowledge_base.json`: sample knowledge graph style data (safe demo content)
- `data/feedback_log.jsonl`: runtime feedback log (should not be committed)
- `reports/`: runtime generated analysis outputs (should not be committed)
- `reference/*.xlsx`: potentially real operational source files (should not be committed)

## What Is Ignored by Default

The repository `.gitignore` excludes:
- `reports/`
- `data/feedback_log.jsonl`
- `reference/*.xlsx`
- cache folders and local tool state

## Before Publishing

1. Verify there is no real personal or customer data in `examples/` and `data/`.
2. Ensure credentials are only provided through environment variables.
3. Keep `.env` local; only keep `.env.example` in git.
4. Re-run a secret scan before push.
