"""Shared optimizer and checkpoint lifecycle for auxiliary audio models."""

from dataclasses import asdict
from pathlib import Path

import torch


class AuxiliaryTraining:
    """Manage an auxiliary model with explicit, local gradient accumulation.

    Each wrapper owns its optimizer. Accelerate can manage device placement and
    mixed precision, but its accumulation must remain one: grad_acc_step controls
    accumulation here independently of the separation model.
    """

    def _prepare_training(self) -> None:
        self._pending_steps = 0
        if self.accelerator is not None:
            if self.accelerator.gradient_accumulation_steps != 1:
                raise ValueError(
                    "Use wrapper grad_acc_step with Accelerator accumulation set to 1."
                )
            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler
            )
        self.optimizer.zero_grad(set_to_none=True)

    def _backward_step(self, loss: torch.Tensor) -> None:
        if not torch.isfinite(loss):
            self.optimizer.zero_grad(set_to_none=True)
            self._pending_steps = 0
            raise ValueError("Auxiliary training loss is non-finite.")
        scaled = loss / self.config.grad_acc_step
        if self.accelerator is None:
            scaled.backward()
        else:
            self.accelerator.backward(scaled)
        self._pending_steps += 1
        if self._pending_steps == self.config.grad_acc_step:
            clip = self.config.grad_clip_thresh
            if clip > 0:
                if self.accelerator is None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                else:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            if not getattr(self.optimizer, "step_was_skipped", False):
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._pending_steps = 0

    def _unwrapped_model(self):
        return self.model if self.accelerator is None else self.accelerator.unwrap_model(self.model)

    def state_dict(self) -> dict:
        """Capture weights, optimizer state, and any partially accumulated gradients."""
        model = self._unwrapped_model()
        return {
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": asdict(self.config),
            "pending_steps": self._pending_steps,
            "gradients": {
                name: p.grad.detach().clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            },
        }

    def save_checkpoint(self, filepath: str) -> None:
        """Save a checkpoint on the main process, including bare filenames."""
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = self.state_dict()
        if self.accelerator is None:
            torch.save(checkpoint, path)
        else:
            self.accelerator.save(checkpoint, path)

    def load_checkpoint(self, filepath: str) -> None:
        """Restore a trusted tensor checkpoint on the current device."""
        self.load_checkpoint_from_dict(
            torch.load(filepath, map_location=self.device, weights_only=True)
        )

    def load_checkpoint_from_dict(self, checkpoint_dict: dict) -> None:
        """Restore model and training state, including legacy checkpoints."""
        model = self._unwrapped_model()
        model.load_state_dict(checkpoint_dict["model"], strict=True)
        self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
        self.scheduler.load_state_dict(checkpoint_dict["scheduler"])
        self._pending_steps = checkpoint_dict.get("pending_steps", 0)
        gradients = checkpoint_dict.get("gradients", {})
        for name, parameter in model.named_parameters():
            gradient = gradients.get(name)
            parameter.grad = None if gradient is None else gradient.to(parameter)

    def get_learning_rate(self) -> float:
        """Return the first optimizer group's learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def train(self):
        """Switch the auxiliary model to training mode."""
        self.model.train()
        return self

    def eval(self):
        """Switch the auxiliary model to evaluation mode."""
        self.model.eval()
        return self
