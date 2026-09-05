# Discriminator and diffusion training

The auxiliary wrappers own their optimizers. Discriminator updates consume
detached generated/reference audio. Generator adversarial and feature-matching
losses preserve gradients through generated audio while freezing discriminator
parameters and reference features. Diffusion training detaches both waveforms
and updates only its velocity model.

## Discriminator

`DiscriminatorConfig` supports `mel`, `wave` (HiFi-GAN), and `music` inputs.
`music` supports the existing pairwise ranking loss or standard hinge loss.
Set architecture options in `model_kwargs`:

```python
from discriminator.discriminator_wrapper import DiscriminatorConfig, DiscriminatorWrapper

config = DiscriminatorConfig(
    input_type="music",
    loss_type="rank",
    grad_acc_step=2,
    model_kwargs={"mpd_periods": [2, 3, 5], "msd_pool_scales": [1, 2]},
)
discriminator = DiscriminatorWrapper(config, device="cuda")
d_losses = discriminator.train_step(
    generated,
    reference,
    lengths=None,
    current_step=step,
    compute_generator_losses=False,
)
g_losses = discriminator.get_generator_losses(generated, reference, None, step)
generator_loss = reconstruction_loss + g_losses["gan_loss"] + g_losses["hdn_loss"]
generator_loss.backward()
```

Wave/music inputs are `[B, T]` or `[B, 1, T]`; reshape stereo channels into the
batch dimension. Mel inputs are `[B, T, mel_bins]` with a length per item. Mel
frequency windows must fit the configured channel count. Logits from different
periods/windows are concatenated because their temporal lengths can differ.

`train_step()` still returns generator losses by default for existing callers.
The main training scripts explicitly disable this extra calculation and call
`get_generator_losses()` once. Disabled losses are differentiable zero tensors.

## Diffusion

`diffusion.models.config.DiffusionConfig` is the shared configuration class,
also exported from `diffusion.diffusion_wrapper`. The registry lazily constructs
`dit`, `transformer`, `simple_mlp`, or `template` and forwards configuration.
The template and MLP are development baselines, not pretrained enhancers.

```python
from diffusion.diffusion_wrapper import DiffusionConfig, DiffusionWrapper

config = DiffusionConfig(model_type="dit", n_train_start=0)
diffusion = DiffusionWrapper(config, device="cuda")
losses = diffusion.train_step_dual(generated, reference, current_step=step)
diffusion.eval()
enhanced = diffusion.inference(generated, num_steps=10, method="euler")
```

Set `model_kwargs` for DiT architecture overrides. Its input frequency count
must equal `n_fft // 2 + 1`; both configured spatial dimensions must be divisible
by their patch dimensions. Training and inference pad/split time windows using
the model's configured frame count, rather than a hard-coded 690 frames. Euler
and RK4 preserve audio length and restore the previous train/eval mode.

The default flow uses deterministic interpolation (`sigma=0`). Positive sigma
uses a Gaussian bridge, including its time derivative in the velocity target.
Stochastic derivatives require strictly interior times `0 < t < 1`.

The combined trainer reads per-stem `training.diffusion_model` and shared
`training.diffusion_options`, for example:

```yaml
training:
  diffusion_model: [dit, dit, none]
  diffusion_options:
    n_train_start: 100000
    grad_acc_step: 1
    scheduler_type: noam
```

## Accumulation and checkpoints

`grad_acc_step` counts active wrapper calls; an update occurs only after a full
accumulation group, including when `current_step` starts at zero. With Accelerate,
leave its gradient accumulation at one because the wrappers manage their own
accumulation. Mixed precision and device placement may still use Accelerate.

Both wrappers support `state_dict()`, `save_checkpoint()`, `load_checkpoint()`,
and `load_checkpoint_from_dict()`. Checkpoints contain weights, optimizer,
scheduler, configuration, and pending accumulated gradients. Save/load uses the
unwrapped model, and only the main process writes a file. Instantiate the same
configuration when resuming training. Pending gradients are not automatically
flushed at epoch boundaries; they carry into the next active call or checkpoint.

Existing discriminator/DiT parameter names are retained. Weight-normalization
deprecation warnings remain to preserve checkpoint compatibility. The MLP no
longer creates trainable projections during forward; legacy MLP checkpoints
containing those ad hoc projections need explicit migration before loading.
Corrected gradient flow, hinge loss, stochastic targets, and diffusion scheduler
behavior mean newly trained runs are not numerically equivalent to legacy runs.

The separation checkpoint does not include trained auxiliary weights. Enabling
diffusion enhancement requires a checkpoint for each enabled stem; inference
loads its saved architecture configuration. The regression and GPU smoke tests
verify execution, gradients, and shapes, not model convergence or quality gains.
