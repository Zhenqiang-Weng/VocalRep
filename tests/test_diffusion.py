"""Exercise diffusion training, numerical paths, shape handling, and resume."""

import pytest
import torch
from torch import nn

from diffusion.diffusion_wrapper import DiffusionConfig, DiffusionWrapper
from diffusion.models.flow_matching import ConditionalFlowMatcher
from diffusion.models.registry import get_diffusion_model
from diffusion.models.scheduler import get_scheduler


def config(**kwargs):
    settings = dict(
        model_type="simple_mlp",
        hidden_size=8,
        num_heads=2,
        num_layers=1,
        ffn_hidden_size=16,
        dropout=0,
        n_train_start=0,
        scheduler_type="constant",
    )
    settings.update(kwargs)
    return DiffusionConfig(**settings)


@pytest.mark.parametrize("name", ["simple_mlp", "transformer", "template"])
def test_waveform_models_have_stable_parameters_and_shapes(name):
    model = get_diffusion_model(name, config(model_type=name))
    original = {id(p) for p in model.parameters()}
    for length in (17, 31):
        x = torch.randn(2, 2, length, requires_grad=True)
        y = model(x, torch.tensor([0.2, 0.7]))
        assert y.shape == x.shape
        y.square().mean().backward()
        assert torch.isfinite(x.grad).all()
        assert {id(p) for p in model.parameters()} == original


def test_transformer_preserves_channel_order():
    model = get_diffusion_model("transformer", config(model_type="transformer")).eval()
    wave = torch.randn(1, 2, 17)
    time = torch.tensor([0.2])
    expected = torch.cat([model(wave[:, channel : channel + 1], time) for channel in range(2)], 1)
    torch.testing.assert_close(model(wave, time), expected, atol=1e-6, rtol=1e-5)


def test_stochastic_velocity_matches_path_derivative():
    flow = ConditionalFlowMatcher(sigma=0.3)
    x0, x1, eps = [torch.randn(2, 2, 10, dtype=torch.float64) for _ in range(3)]
    time = torch.tensor([0.2, 0.75], dtype=torch.float64)
    xt = flow.sample_xt(x0, x1, time, eps)
    dt = 1e-6
    derivative = (
        flow.sample_xt(x0, x1, time + dt, eps) - flow.sample_xt(x0, x1, time - dt, eps)
    ) / (2 * dt)
    torch.testing.assert_close(flow.compute_conditional_flow(x0, x1, time, xt), derivative)
    torch.testing.assert_close(flow.sample_xt(x0, x1, torch.zeros(2), eps), x0)
    torch.testing.assert_close(flow.sample_xt(x0, x1, torch.ones(2), eps), x1)


def test_accumulation_and_checkpoint_resume(tmp_path, monkeypatch):
    wrapper = DiffusionWrapper(config(grad_acc_step=2))
    source, target = torch.randn(1, 2, 17), torch.randn(1, 2, 17)
    before = {k: v.clone() for k, v in wrapper.model.state_dict().items()}
    wrapper.train_step_dual(source, target, 0, t=torch.tensor(0.3))
    for key, value in wrapper.model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    monkeypatch.chdir(tmp_path)
    wrapper.save_checkpoint("resume.pt")
    restored = DiffusionWrapper(config(grad_acc_step=2))
    restored.load_checkpoint("resume.pt")
    assert restored._pending_steps == 1
    wrapper.train_step_dual(source, target, 1, t=torch.tensor(0.3))
    restored.train_step_dual(source, target, 1, t=torch.tensor(0.3))
    assert any(not torch.equal(v, before[k]) for k, v in wrapper.model.state_dict().items())
    for key, value in wrapper.model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[key])


class ZeroVelocity(nn.Module):
    def forward(self, x, time):
        return torch.zeros_like(x)


@pytest.mark.parametrize("method", ["euler", "rk4"])
@pytest.mark.parametrize("use_stft", [False, True])
def test_identity_inference_preserves_all_samples_and_mode(method, use_stft):
    wrapper = DiffusionWrapper(
        config(
            use_stft=use_stft,
            n_fft=32,
            win_length=32,
            hop_length=8,
            sample_rate=100,
            chunk_size_seconds=0.64,
        )
    )
    wrapper.model = ZeroVelocity().eval()
    for length in (7, 64, 151):
        wave = torch.randn(2, 2, length)
        output = wrapper.inference(wave, num_steps=2, method=method, batch_size=3)
        torch.testing.assert_close(output, wave, atol=1e-5, rtol=1e-5)
        assert not wrapper.model.training
    with pytest.raises(ValueError):
        wrapper.inference(wave, num_steps=0)
    assert not wrapper.model.training


def test_dit_training_and_inference_use_configured_frame_size():
    wrapper = DiffusionWrapper(
        config(
            model_type="dit",
            n_fft=16,
            win_length=16,
            hop_length=4,
            model_kwargs={
                "input_size": [9, 8],
                "patch_size": [3, 2],
                "hidden_size": 24,
                "depth": 1,
                "num_heads": 2,
            },
            sample_rate=64,
            chunk_size_seconds=0.5,
        )
    )
    wave = torch.randn(1, 2, 43)
    losses = wrapper.train_step_dual(wave, wave * 0.5, 0, t=torch.tensor([0.3]))
    assert torch.isfinite(losses["diffusion_loss"])
    output = wrapper.eval().inference(wave, num_steps=1, batch_size=2)
    assert output.shape == wave.shape and torch.isfinite(output).all()
    assert not wrapper.model.training


@pytest.mark.parametrize("name", ["noam", "warmup", "cosine"])
def test_scheduler_queries_do_not_advance_and_resume(name):
    parameter = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = get_scheduler(name, optimizer, n_warmup=2)
    first = scheduler.get_lr()
    assert scheduler.get_lr() == first
    optimizer.step()
    scheduler.step()
    state = scheduler.state_dict()
    restored = get_scheduler(name, optimizer, n_warmup=2)
    restored.load_state_dict(state)
    assert restored.get_lr() == scheduler.get_lr()


def test_accelerate_training_on_cpu():
    from accelerate import Accelerator

    accelerator = Accelerator(cpu=True)
    wrapper = DiffusionWrapper(config(), accelerator=accelerator)
    wave = torch.randn(1, 2, 17)
    losses = wrapper.train_step_single(wave, current_step=0)
    assert torch.isfinite(losses["diffusion_loss"])
