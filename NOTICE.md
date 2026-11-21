# Rights, Responsibilities, and Third-Party Notices

This document defines the intellectual property status, maintainer and user
responsibilities, compliance boundaries, disclaimer of liability, and
third-party attributions for **VocalRep**.

## 1. Copyright and License Status

Copyright (c) Zhenqiang Weng and VocalRep contributors.

This repository is an experimental research codebase that is still being
organized and updated. **It has not been released under an open-source license.**
Unless and until an explicit license file is added to this repository by the
copyright holder:

- All rights are reserved for original VocalRep code, documentation,
  configurations, and model weights.
- No permission is granted to use, copy, modify, redistribute, sublicense, or
  deploy original repository assets for commercial or non-commercial purposes
  without prior written consent from the copyright holder.
- Individual files derived from third-party open-source projects remain governed
  by their respective upstream copyrights and licenses (see Section 5).

## 2. Division of Responsibilities

### Maintainers

- Maintain repository integrity, code review standards, and Git LFS tracking for
  original checkpoints under `ckpts/`.
- Preserve upstream copyright notices, license headers, and attributions when
  incorporating or adapting third-party code.
- Review security reports and coordinate remediation on a best-effort basis for
  an active research repository (see [`SECURITY.md`](SECURITY.md)).
- Maintainers do **not** guarantee backward compatibility across research
  iterations, production SLAs, or fitness for any specific downstream task.

### Users and Evaluators

- Verify that your access to and use of this repository complies with Section 1
  and all applicable upstream licenses.
- Obtain all required rights, consents, and licenses for any audio recordings,
  vocal stems, speaker embeddings, datasets, or external model weights you
  process.
- Inspect external checkpoints and scripts before execution; never load
  untrusted PyTorch pickle files (see [`SECURITY.md`](SECURITY.md)).
- Keep private datasets, extracted speaker embeddings, API keys (such as
  `WANDB_API_KEY`), and local caches out of version control and public issues.

### Contributors

- Follow [`CONTRIBUTING.md`](CONTRIBUTING.md) for development checks, coding
  standards, checkpoint governance, and intellectual property requirements.
- Ensure every submitted contribution is either your original work or properly
  licensed and attributed, with no confidential data, credentials, or
  copyrighted media included.

## 3. Audio Copyright, Voice Identity, and Ethical Use

VocalRep processes singing voices and global speaker identity embeddings. Any
authorized research use must adhere to the following boundaries:

- **Copyrighted audio:** This repository does not distribute commercial music or
  private datasets. Users are solely responsible for ensuring lawful acquisition
  and use of all training, validation, and inference audio.
- **Speaker and vocal identity:** Speaker embeddings derived from vocal
  recordings can constitute biometric or personally identifiable characteristics
  under applicable privacy regulations. Do not collect, store, or share vocal
  embeddings without appropriate authorization or legal basis.
- **Prohibited misuse:** Do not use this codebase or its outputs for
  unauthorized voice cloning, impersonation, fraud, copyright infringement, or
  any unlawful activity.

## 4. Disclaimer of Warranty and Limitation of Liability

TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS REPOSITORY, ITS SOURCE
CODE, CONFIGURATIONS, DOCUMENTATION, AND ANY ASSOCIATED MODEL WEIGHTS ARE
PROVIDED **"AS IS"** AND **"AS AVAILABLE"**, WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, ACCURACY, AND
NON-INFRINGEMENT.

IN NO EVENT SHALL THE AUTHORS, COPYRIGHT HOLDERS, OR MAINTAINERS BE LIABLE FOR
ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT,
OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THIS REPOSITORY OR THE
USE OR OTHER DEALINGS IN ITS CONTENTS, INCLUDING HARDWARE COSTS, DATA LOSS,
MODEL MISUSE, OR THIRD-PARTY CLAIMS.

## 5. Third-Party Components and Attributions

The table below lists major upstream projects referenced, adapted, or downloaded
on demand by this repository. Retained copyright and license notices inside
individual source files remain in full effect and take precedence for those
files.

| Repository path / component | Upstream project / attribution | Notes / upstream terms |
| --- | --- | --- |
| `train_accelerate_*.py`, `inference.py`, `utils/dataset.py`, `utils/model_utils.py`, `utils/losses.py`, `utils/metrics.py`, `scripts/prepare_weights_for_inference.py` | [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) (Roman Solovyev) | Training, dataset, metric, and demixing framework structure; upstream MIT license |
| `models/bs_roformer/` | BS-RoFormer & Mel-Band RoFormer implementations (lucidrains / ZFTurbo community adaptations) | Source-separation transformer backbones and attention wrappers |
| `models/scnet/` | SCNet architecture | Split-band music source separation backbone |
| `models/bandit_v2/` | BandIt v2 architecture | Band-split source separation backbone |
| `diffusion/models/dit/dit.py` | Meta Platforms, Inc. and affiliates (DiT) | Retains original Meta copyright header in file |
| `spk_extract/` and on-demand CAM++ weights | ModelScope / 3D-Speaker `iic/speech_campplus_sv_zh_en_16k-common_advanced` (revision `v1.0.0`) | Pretrained weights are downloaded on demand and SHA-256 verified; governed by upstream ModelScope / 3D-Speaker terms (Apache-2.0) |
| `discriminator/models/losses/pytorch_sdtw.py` | Mehran Maghoumi (`pytorch-softdtw-cuda`) & Sleepwalking (`pytorch-softdtw`) | Retains full MIT license header in file |
| `discriminator/models/losses/pytorch_ssim.py` | Po-Hsun Su (`pytorch-ssim`) | Retains upstream attribution in file |
| `discriminator/models/hifigan.py`, `discriminator/models/blocks.py`, `discriminator/helpers/` | HiFi-GAN / WaveNet-style discriminator components and Atomicoo helper utilities | Auxiliary adversarial training modules |
| `utils/muon.py` | Muon optimizer (`SingleDeviceMuonWithAuxAdam`) | Includes quintic orthogonalization credits in file |
| `utils/rmvpe.py` | RMVPE pitch estimation architecture | Pitch extraction utility |

If you believe any third-party copyright notice, attribution, or license link is
incomplete or inaccurate, please open an issue or contact the repository
maintainer so it can be corrected promptly.
