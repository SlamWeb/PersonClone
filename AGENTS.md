# AGENTS.md

Use this file as the working contract for future agents in this open-source repo.

## Working Loop

```text
read SPEC.md -> state assumption -> make a small change -> verify -> update SPEC.md if the contract changed
```

Use `navigation.md` as the repository change map. When a change crosses a backend, frontend,
data, evaluation, or study boundary, update the affected module contract and focused tests together;
do not rely on chat history to remember the dependency.

## Boundaries

- Keep product, evaluation, and research code in explicit bounded contexts. Product evaluation stays under
  `src/personaforge/eval`; human-subject study tooling stays under
  `src/personaforge/studies`. The old `C:\PersonaForge` directory is deprecated and must not be used as a
  code or data source.
- Do not commit real crawled corpus, auth state, local indexes, model files, `.env`, or API keys.
- Prefer local-first behavior: user data and credentials stay on the user's machine.
- Keep MVP code paths explainable for interviews.

## Encoding

Chinese Markdown and sample text are allowed. Use UTF-8 for all files.

On Windows, prefer:

```powershell
chcp 65001 > $null
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

## Current Priority

This repository is the single product and research workspace. Keep the current work focused through
the navigation map and the nearest module SPEC; do not create a second research checkout.

1. preserve the working local Web and multi-author path
2. complete RAG and generation evaluation with frozen run IDs
3. keep Study 1 materials, protocol, analysis, and participant data isolated under their contracts
4. document and test every cross-module change
