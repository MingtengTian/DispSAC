from __future__ import annotations

from typing import Dict, Optional

import torch


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device: torch.device | str = "cpu",
        alpha: float = 0.6,
    ):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.alpha = float(alpha)
        self.storage: Dict[str, torch.Tensor] = {}
        self.priorities = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self.ptr = 0
        self.size = 0
        self.add_calls = 0
        self.sample_calls = 0

    def __len__(self) -> int:
        return self.size

    def _ensure_storage(self, batch: Dict[str, torch.Tensor]) -> None:
        if self.storage:
            return
        for key, value in batch.items():
            value = value.detach().to(self.device)
            self.storage[key] = torch.empty(
                (self.capacity, *value.shape[1:]),
                dtype=value.dtype,
                device=self.device,
            )
        shape_summary = ", ".join(f"{key}:{tuple(value.shape)}" for key, value in self.storage.items())
        print(
            f"[ReplayBuffer] initialized capacity={self.capacity} device={self.device} alpha={self.alpha:.3f} "
            f"fields={shape_summary}",
            flush=True,
        )

    def add_batch(self, batch: Dict[str, torch.Tensor], priorities: Optional[torch.Tensor] = None) -> None:
        self._ensure_storage(batch)
        self.add_calls += 1
        batch_size = batch["action"].shape[0]
        write_indices = (torch.arange(batch_size, device=self.device) + self.ptr) % self.capacity
        ptr_before = self.ptr
        size_before = self.size

        for key, value in batch.items():
            self.storage[key][write_indices] = value.detach().to(self.device)

        if priorities is None:
            default_priority = self.priorities[: self.size].max() if self.size > 0 else torch.tensor(1.0, device=self.device)
            priorities = torch.full((batch_size,), float(default_priority.item()), dtype=torch.float32, device=self.device)
        else:
            priorities = priorities.detach().to(self.device, dtype=torch.float32).view(-1)

        self.priorities[write_indices] = priorities
        self.ptr = int((self.ptr + batch_size) % self.capacity)
        self.size = min(self.size + batch_size, self.capacity)

        should_log = self.add_calls <= 3 or self.add_calls % 10 == 0 or size_before < self.capacity <= self.size
        if should_log:
            print(
                f"[ReplayBuffer.add_batch] call={self.add_calls} batch={batch_size} ptr={ptr_before}->{self.ptr} "
                f"size={size_before}->{self.size} priority_mean={priorities.mean().item():.4f} "
                f"priority_min={priorities.min().item():.4f} priority_max={priorities.max().item():.4f}",
                flush=True,
            )

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if batch_size > self.size:
            raise ValueError(f"Requested batch_size={batch_size}, only {self.size} samples available")
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {key: value[indices] for key, value in self.storage.items()}

    def sample_prioritized(self, batch_size: int, beta: float = 0.4) -> Dict[str, torch.Tensor]:
        if batch_size > self.size:
            raise ValueError(f"Requested batch_size={batch_size}, only {self.size} samples available")
        self.sample_calls += 1
        p = self.priorities[: self.size].clamp_min(1e-12).pow(self.alpha)
        probs = p / p.sum()
        indices = torch.multinomial(probs, batch_size, replacement=True)
        weights = (self.size * probs[indices]).pow(-beta)
        weights = weights / weights.max().clamp_min(1e-12)
        batch = {key: value[indices] for key, value in self.storage.items()}
        batch["indices"] = indices
        batch["weights"] = weights.unsqueeze(-1)

        if self.sample_calls <= 3 or self.sample_calls % 20 == 0:
            print(
                f"[ReplayBuffer.sample_prioritized] call={self.sample_calls} batch={batch_size} beta={beta:.3f} "
                f"size={self.size} prob_min={probs.min().item():.6f} prob_max={probs.max().item():.6f} "
                f"weight_mean={weights.mean().item():.4f} weight_std={weights.std().item():.4f}",
                flush=True,
            )
        return batch

    def update_priorities(self, indices: torch.Tensor, new_priorities: torch.Tensor) -> None:
        indices = indices.detach().to(self.device, dtype=torch.long).view(-1)
        new_priorities = new_priorities.detach().to(self.device, dtype=torch.float32).view(-1)
        self.priorities[indices] = new_priorities
