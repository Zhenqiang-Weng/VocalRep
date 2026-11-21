# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, credential leak, or unsafe artifact
handling issue in this repository, **please do not disclose it in a public
GitHub issue or pull request.**

Instead, report it privately through one of the following channels:

1. **GitHub Security Advisories:** Use the repository's **Security -> Report a
   vulnerability** workflow on GitHub (preferred).
2. **Direct Maintainer Contact:** Contact the repository owner via the email
   listed on the GitHub profile / commit history if private advisories are
   unavailable.

Please include:

- A concise description of the issue and affected file(s) or command(s).
- Minimal reproduction steps or proof of concept without including private audio
  or real credentials.
- Potential impact (for example, arbitrary code execution via unsafe
  deserialization, path traversal, or secret leakage).

Because VocalRep is an experimental research repository, fixes are handled on a
best-effort basis, and maintainers will acknowledge valid reports as soon as
practical.

## Security Scope and Safe Usage Practices

Users and contributors are responsible for observing these operational
safeguards when running VocalRep locally or in shared GPU environments:

- **Untrusted checkpoints (`torch.load`):** Full separator checkpoints loaded
  with `weights_only=False` rely on Python's `pickle` format and can execute
  arbitrary code if tampered with. Only load checkpoints from trusted, verified
  sources and verify SHA-256 checksums before use.
- **Pretrained speaker encoder verification:** The default CAM++ download helper
  in [`spk_extract/pretrained.py`](spk_extract/pretrained.py) pins revision
  `v1.0.0` and validates SHA-256 digests for both `configuration.json` and
  `campplus_cn_en_common.pt`. Do not bypass checksum verification for untrusted
  mirrors.
- **Secrets and tokens:** Never place `WANDB_API_KEY`, Hugging Face/ModelScope
  tokens, SSH keys, or `.env` files in source code, scripts, logs, issues, or
  commits. Pass credentials exclusively through environment variables.
- **Third-party dependencies:** Keep PyTorch, FFmpeg, and Python dependencies up
  to date in isolated virtual environments (`conda` / `venv`) and avoid running
  training or inference scripts as `root`.
- **Input audio and output directories:** Validate input directory paths before
  batch processing and avoid pointing `--store_dir` or `--results_path` at
  system directories or tracked source folders.
