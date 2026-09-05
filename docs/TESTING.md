# Validation record

Validated on 2026-09-05 with Python 3.10, PyTorch/TorchAudio 2.11.0+cu130,
and an NVIDIA GeForce RTX 4080. Tests used an isolated environment; installation
is documented in the README.

## Real three-stem inference

Source: the project's published [Boy Friend demonstration](https://zhenqiang-weng.github.io/VocalRep_/).
The fixture downloads the mixture and the published vocal, backing-vocal, and
instrumental stems. The upstream `mix.WAV` actually contains MP3 data; after
decoding, the tested waveform is 884,736 stereo samples at 44,100 Hz (20.062 s).
The container's reported duration differs slightly from the decoded length.

```bash
python -m scripts.smoke_test_three_stem \
  --checkpoint checkpoints/vocalrep_original.ckpt
```

The local cache contains the original repository separation weight, downloaded
without changing `ckpts/`. A normal Git LFS checkout can instead pass the tracked
checkpoint path shown in the README.

| Item | Verified value |
| --- | --- |
| Separation weight SHA-256 | `423a4b5313b5938a678a91fa753412a08419e3b62593448b85d7d3f0c70a4db9` |
| CAM++ model | `iic/speech_campplus_sv_zh_en_16k-common_advanced`, revision `v1.0.0` |
| CAM++ weight SHA-256 | `92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199` |
| CAM++ embedding | Float32, finite, shape `(1, 192)` |
| Processing | Blind separation, dominant-speaker extraction, guided separation |
| Separation settings | 176,400 samples per chunk, batch size 1, overlap factor 3 |
| Output | Three finite, non-silent stereo stems per pass, each 884,736 samples |
| Processing time | 10.70 seconds for the folder pipeline, excluding separation weight loading |

Guided output RMS levels were 0.117227 (vocals), 0.029874 (backing vocals), and
0.267353 (instrumental). Floating-point WAV preserves peaks above 1.0, including
an instrumental peak of 1.15518; playback/export to integer PCM may require gain
adjustment. Audio and detailed JSON reports remain local under
`results/boy_friend/`, with source audio under `test_sample/boy_friend/`.

Published demo stems are model outputs, not ground-truth recordings. These
checks establish pipeline execution and output integrity, not SDR, listening
quality, or superiority over another separator. Downloaded music is excluded
from Git and is not redistributed with the code.

## Auxiliary model tests on real audio

For each guided stem, a 4,096-sample excerpt starting at 5 seconds was paired
with the corresponding published demo stem. Fresh test models performed one
music-discriminator update, generator/feature-matching backward, one DiT update,
and two-step Euler inference on CUDA. All losses and gradients were finite, and
enhanced outputs retained shape `(1, 2, 4096)`.

The music discriminator used period `[2]` and scale `[1]`. DiT used FFT/window
128, hop 32, input size `[65, 32]`, patch size `[5, 4]`, hidden size 48, depth 1,
and 4 heads. These are small execution checks, not trained production models.

| Stem | Discriminator loss | Feature-matching loss | Diffusion loss |
| --- | ---: | ---: | ---: |
| Vocals | 0.693150 | 0.000719 | 0.024693 |
| Backing vocals | 0.693150 | 0.000297 | 0.007548 |
| Instrumental | 0.693149 | 0.000382 | 0.014654 |

## Automated regression checks

`python -m pytest -q` passed 37 tests, covering:

- Mel, HiFi-GAN, and music discriminator updates, differentiable feature
  matching, reference/parameter gradient isolation, and checkpoint reloads.
- MLP, Transformer, template, and DiT execution; fixed parameter registration;
  stereo channel order; STFT reconstruction; Euler/RK4 overlap-add; scheduler
  state; gradient accumulation; pending-gradient resume; CPU Accelerate.
- CAM++ short/stereo/silent segment handling, bounded inference batches,
  separation boundaries across chunk batch sizes, FLAC subtypes, resampling,
  reversible normalization, and empty-input errors.

Additional checks: Ruff lint/format, Python compilation, English-comment scan,
shell syntax, and both auxiliary-training CLI help commands. CI runs static
checks and CPU regression tests without downloading checkpoints or demo audio.
Multi-GPU training, full training convergence, and TensorRT export were not tested.
