# Contributing Guide

Thanks for helping improve Process Assistant.

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Run the app locally:

```bash
python web_ui.py
```

## Development Conventions

- Keep functions focused and small.
- Add or update examples when behavior changes.
- Prefer explicit data contracts for JSON payloads.

## Commit and PR

- Use clear commit messages, e.g. `fix: sanitize login config`.
- Keep PRs small and focused.
- Include a short test/verification note in PR descriptions.

## Security and Data

- Never commit `.env` or private credentials.
- Never commit real production or customer data.
- Follow `SECURITY.md` for vulnerability reporting.
