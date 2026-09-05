"""Adversarial and feature-matching training for mel and waveform discriminators."""

from contextlib import contextmanager
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .models.config import DiscriminatorConfig
from .models.duration import NoamScheduler
from utils.auxiliary_training import AuxiliaryTraining


def _flatten_features(features):
    """Yield tensors from nested feature-map lists without splitting their batches."""
    if isinstance(features, torch.Tensor):
        yield features
    else:
        for item in features:
            yield from _flatten_features(item)


class DiscriminatorWrapper(AuxiliaryTraining):
    """Own discriminator updates and expose differentiable generator losses."""

    def __init__(
        self,
        config: DiscriminatorConfig | dict,
        accelerator=None,
        device: Optional[torch.device] = None,
    ):
        self.config = DiscriminatorConfig.from_dict(config) if isinstance(config, dict) else config
        self.accelerator = accelerator
        self.device = torch.device(accelerator.device if accelerator else device or "cpu")
        self.model = self._create_model().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = NoamScheduler(
            self.optimizer, n_warmup=self.config.n_warmup, init_scale=self.config.init_scale
        )
        self._prepare_training()

    @property
    def discriminator(self):
        """Retain the public discriminator attribute used by training scripts."""
        return self.model

    def _create_model(self):
        if self.config.input_type == "mel":
            from .models.discriminator import Discriminator

            return Discriminator(self.config.get_mel_hparams())
        if self.config.input_type == "music":
            from .models.music_discriminator import MusicDiscriminator

            return MusicDiscriminator(**self.config.get_music_kwargs())
        from .models.hifigan import HiFiGANDiscriminator

        return HiFiGANDiscriminator(**self.config.get_wave_kwargs())

    def _inactive(self, lengths, current_step: int) -> bool:
        if current_step < self.config.n_train_start:
            return True
        if self.config.input_type == "mel":
            if lengths is None:
                raise ValueError("Mel discriminator inputs require sequence lengths.")
            return bool(lengths.min() < self.config.min_mel_frames())
        return False

    @contextmanager
    def _frozen_discriminator(self):
        # Keep input gradients while preventing generator loss from updating D.
        parameters = list(self.model.parameters())
        requires_grad = [parameter.requires_grad for parameter in parameters]
        training = self.model.training
        try:
            self.model.eval()
            for parameter in parameters:
                parameter.requires_grad_(False)
            yield
        finally:
            self.model.train(training)
            for parameter, required in zip(parameters, requires_grad):
                parameter.requires_grad_(required)

    def _music_loss(self, outputs, generator: bool) -> torch.Tensor:
        terms = []
        for item in outputs:
            real, fake = item["real_logit"], item["fake_logit"]
            if self.config.loss_type == "rank":
                difference = real.detach() - fake if generator else fake - real
                term = F.softplus(difference + self.config.rank_margin).mean()
            elif generator:
                term = -fake.mean()
            else:
                term = F.relu(1 - real).mean() + F.relu(1 + fake).mean()
            terms.append(term)
        if not terms:
            raise ValueError("The discriminator must have at least one active branch.")
        return torch.stack(terms).mean()

    @staticmethod
    def _matching_loss(fake_features, real_features, zero: torch.Tensor) -> torch.Tensor:
        fake = list(_flatten_features(fake_features))
        real = list(_flatten_features(real_features))
        if len(fake) != len(real) or any(f.shape != r.shape for f, r in zip(fake, real)):
            raise ValueError("Real and fake feature-map structures must match.")
        if not fake:
            return zero
        return torch.stack([F.l1_loss(f, r.detach()) for f, r in zip(fake, real)]).mean()

    def train_step(
        self,
        fake_data: torch.Tensor,
        real_data: torch.Tensor,
        lengths: Optional[torch.Tensor],
        current_step: int,
        return_features: bool = True,
        compute_generator_losses: bool = True,
    ) -> dict:
        """Update D using detached inputs; optionally return G losses for compatibility.

        current_step controls delayed start. Accumulation counts active calls,
        so step zero never prematurely flushes a partial batch.
        """
        zero = fake_data.sum() * 0
        if self._inactive(lengths, current_step):
            return {"disc_loss": zero.detach(), "gan_loss": zero, "hdn_loss": zero}
        if fake_data.shape != real_data.shape:
            raise ValueError("Real and generated data must have matching shapes.")
        self.model.train()
        real, fake = real_data.detach(), fake_data.detach()
        if self.config.input_type == "music":
            disc_loss = self._music_loss(self.model(real, fake), generator=False)
        else:
            d_real, starts, _ = self.model(real, lengths)
            if self.config.input_type == "mel":
                d_fake, _, _ = self.model(fake, lengths, start_frames_wins=starts)
            else:
                d_fake, _, _ = self.model(fake, lengths)
            disc_loss = F.mse_loss(d_real, torch.ones_like(d_real)) + F.mse_loss(
                d_fake, torch.zeros_like(d_fake)
            )
        self._backward_step(disc_loss)
        losses = {"disc_loss": disc_loss.detach(), "gan_loss": zero, "hdn_loss": zero}
        if compute_generator_losses:
            losses.update(
                self.get_generator_losses(
                    fake_data, real_data, lengths, current_step, return_features=return_features
                )
            )
        return losses

    def get_generator_losses(
        self,
        fake_data: torch.Tensor,
        real_data: torch.Tensor,
        lengths: Optional[torch.Tensor],
        current_step: int,
        return_features: bool = True,
    ) -> dict:
        """Return G losses while freezing discriminator parameters and reference features."""
        zero = fake_data.sum() * 0
        if self._inactive(lengths, current_step):
            return {"gan_loss": zero, "hdn_loss": zero}
        matching = self.config.enable_hdn_loss and return_features
        with self._frozen_discriminator():
            if self.config.input_type == "music":
                outputs = self.model(real_data.detach(), fake_data)
                gan_loss = self._music_loss(outputs, generator=True)
                terms = (
                    [
                        self._matching_loss(item["fake_fmaps"][:-1], item["real_fmaps"][:-1], zero)
                        for item in outputs
                    ]
                    if matching
                    else []
                )
                fm_loss = torch.stack(terms).mean() if terms else zero
            else:
                d_fake, starts, fake_features = self.model(fake_data, lengths)
                gan_loss = F.mse_loss(d_fake, torch.ones_like(d_fake))
                fm_loss = zero
                if matching:
                    with torch.no_grad():
                        if self.config.input_type == "mel":
                            _, _, real_features = self.model(
                                real_data.detach(), lengths, start_frames_wins=starts
                            )
                        else:
                            _, _, real_features = self.model(real_data.detach(), lengths)
                    fm_loss = self._matching_loss(fake_features, real_features, zero)
        return {
            "gan_loss": gan_loss * self.config.gan_weight,
            "hdn_loss": fm_loss * self.config.fml_weight,
        }
