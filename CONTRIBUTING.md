# Contributing

Thank you for helping maintain this project. Because model training is expensive and checkpoint files are large, keep each change focused, reviewable, and easy to verify.

## Development environment

Activate the project environment and install the development dependencies:

```bash
conda activate mss
python -m pip install -r requirements-dev.txt
```

## Checks

Run at least the following checks before submitting a change:

```bash
ruff check .
python -m compileall -q .
bash -n train_accelerate.sh infer_with_spk.sh
```

Changes to a model or data pipeline should also be verified with a small representative sample, using the relevant forward-pass, training, or inference path.

## Code conventions

- Target Python 3.10, use four-space indentation, and save text files as UTF-8 with LF line endings.
- Add clear type annotations and docstrings to new public functions.
- Pass paths, GPU IDs, credentials, and server addresses through command-line arguments or environment variables; do not hard-code them in source files.
- Do not commit datasets, logs, caches, exported artifacts, or local editor settings.
- Never include API keys, access tokens, or other credentials in issues, logs, source files, or commits.
- Keep large formatting-only changes separate from functional changes so that reviews remain clear.

## Checkpoints and Git LFS

- Existing files in `ckpts/` are original project assets. Do not replace, regenerate, or modify them without explicit maintainer approval.
- Before adding a new checkpoint, verify its origin, license, and checksum, then store it with Git LFS.
- Do not run `git lfs migrate` or any command that rewrites checkpoint history unless a maintainer has explicitly approved the history rewrite.

## Commits

Keep commits focused and write concise, imperative commit messages with a clear scope. For example:

```text
docs: document CUDA installation
fix: correct discriminator dataset import
chore: remove tracked Python caches
```
