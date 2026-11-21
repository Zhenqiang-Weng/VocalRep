# Contributing

Thank you for helping maintain this project. Because model training is expensive
and checkpoint files are large, keep each change focused, reviewable, and easy
to verify.

Before submitting any pull request, review [`NOTICE.md`](NOTICE.md) and
[`SECURITY.md`](SECURITY.md) for the repository's rights reservation,
third-party attribution rules, and security boundaries.

## Roles and Responsibilities

- **Maintainers** own final review, merge decisions, release/licensing status,
  and Git LFS governance for checkpoints under `ckpts/`.
- **Contributors** are responsible for keeping diffs scoped, running required
  static/regression checks, preserving existing checkpoint `state_dict`
  compatibility unless a breaking change is explicitly approved, and verifying
  that no private data, copyrighted media, or credentials are included.

## Intellectual Property and Contribution Terms

- This repository currently has **no open-source license** (all rights reserved;
  see [`NOTICE.md`](NOTICE.md)). Submitting a patch or pull request does not
  grant any license over existing VocalRep assets.
- Only submit code and documentation that you authored yourself or that you are
  legally permitted to contribute under compatible terms.
- Preserve all upstream copyright headers and license notices in adapted
  third-party files, and update the attribution table in [`NOTICE.md`](NOTICE.md)
  whenever a new third-party component is introduced.
- Never commit copyrighted audio, private speaker embeddings, API keys, or
  unverified third-party checkpoints.

## Development Environment

Activate the project environment and install the development dependencies:

```bash
conda activate mss
python -m pip install -r requirements.txt -r requirements-dev.txt
```

## Checks

Run at least the following checks before submitting a change:

```bash
ruff check .
ruff format --check .
python scripts/check_english_comments.py
python -m pytest -q
python -m compileall -q -x '/(\.venv|checkpoints|test_sample|results)/' .
bash -n train_accelerate.sh infer_with_spk.sh
```

Changes to a model, training runtime (`utils/train_utils.py`), or data pipeline
should also be verified with a small representative sample using the relevant
forward-pass, training, or inference path, and must keep `.gitignore` regression
checks in `tests/test_gitignore.py` passing.

## Code Conventions

- Target Python 3.10+, use four-space indentation, and save text files as UTF-8
  with LF line endings.
- Use English for comments, docstrings, and diagnostic messages (enforced by
  `scripts/check_english_comments.py`). Run `ruff format .` before submitting
  Python changes.
- Add clear type annotations and docstrings to new public functions.
- Keep model parameter/buffer names backward-compatible with published
  checkpoints under `ckpts/`.
- Pass paths, GPU IDs, credentials, and server addresses through command-line
  arguments or environment variables; do not hard-code them in source files.
- Do not commit datasets, logs, caches, exported artifacts, profiling dumps, or
  local editor/assistant settings.
- Never include API keys, access tokens, or other credentials in issues, logs,
  source files, or commits.
- Keep large formatting-only changes separate from functional changes so that
  reviews remain clear.

## Checkpoints and Git LFS

- Existing files in `ckpts/` are original project assets. Do not replace,
  regenerate, or modify them without explicit maintainer approval.
- Before adding a new checkpoint, verify its origin, license, and SHA-256
  checksum, then store it with Git LFS.
- Do not run `git lfs migrate` or any command that rewrites checkpoint history
  unless a maintainer has explicitly approved the history rewrite.

## Commits

Keep commits focused and write concise, imperative commit messages with a clear
scope. For example:

```text
docs: document CUDA installation
fix: correct discriminator dataset import
chore: remove tracked Python caches
```
