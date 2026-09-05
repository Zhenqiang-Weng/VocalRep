"""Check real discriminator gradients, disabled losses, and checkpoint recovery."""

import pytest
import torch

from discriminator.discriminator_wrapper import DiscriminatorConfig, DiscriminatorWrapper


@pytest.mark.parametrize("family", ["mel", "wave", "music"])
def test_training_and_feature_matching_gradients(family, tmp_path):
    kwargs = {"mpd_periods": [2], "msd_pool_scales": [1]}
    if family == "mel":
        kwargs.update(time_lengths=[16], freq_lengths=[8], n_mel_channels=8, hidden_size=8)
    config = DiscriminatorConfig(input_type=family, model_kwargs=kwargs, grad_acc_step=2)
    wrapper = DiscriminatorWrapper(config)
    shape = (1, 32, 8) if family == "mel" else (1, 512)
    fake = torch.randn(shape, requires_grad=True)
    real = torch.randn(shape, requires_grad=True)
    lengths = torch.tensor([32]) if family == "mel" else None
    wrapper.train_step(fake, real, lengths, 0, compute_generator_losses=False)
    assert fake.grad is None and real.grad is None
    before = {
        name: parameter.grad.clone() if parameter.grad is not None else None
        for name, parameter in wrapper.model.named_parameters()
    }
    losses = wrapper.get_generator_losses(fake, real, lengths, 0)
    assert losses["hdn_loss"].requires_grad
    losses["hdn_loss"].backward()
    assert fake.grad is not None and fake.grad.abs().sum() > 0
    assert real.grad is None
    for name, parameter in wrapper.model.named_parameters():
        if before[name] is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, before[name])
    wrapper.train_step(fake, real, lengths, 1, compute_generator_losses=False)
    assert wrapper._pending_steps == 0
    path = tmp_path / f"{family}.pt"
    wrapper.save_checkpoint(str(path))
    wrapper.load_checkpoint(str(path))


def test_disabled_discriminator_losses_are_differentiable_zeros():
    wrapper = DiscriminatorWrapper(
        DiscriminatorConfig(
            input_type="mel",
            n_train_start=10,
            enable_hdn_loss=False,
            model_kwargs={
                "time_lengths": [8],
                "freq_lengths": [4],
                "hidden_size": 4,
                "n_mel_channels": 4,
            },
        )
    )
    fake = torch.randn(1, 16, 4, requires_grad=True)
    losses = wrapper.get_generator_losses(fake, fake.detach(), torch.tensor([16]), 0)
    (losses["gan_loss"] + losses["hdn_loss"]).backward()
    assert torch.count_nonzero(fake.grad) == 0
