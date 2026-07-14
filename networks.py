from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from config import TrainConfig, get_default_config
from utils import infer_device


class StateEncoder(nn.Module):
    def __init__(self, cfg: TrainConfig | None = None, device: torch.device | str | None = None):
        super().__init__()
        self.cfg = cfg or get_default_config()
        self.device = torch.device(device or infer_device(self.cfg))
        # features = np.load(self.cfg.data.pretrain_features_path, allow_pickle=True)
        # data = features["data"] if isinstance(features, np.lib.npyio.NpzFile) else features
        period_count = self.cfg.env.period_count

        residual_dim = 2 * period_count
        parameter_dim = self.cfg.model.action_dim
        violation_dim = self.cfg.model.violation_dim
        fit_dim = self.cfg.model.fit_dim
        physics_dim = self.cfg.model.physics_dim
        enc_expand_ratio = self.cfg.model.encoder_expand_ratio

        self.fit_encoder = nn.Sequential(
            nn.LayerNorm(parameter_dim + residual_dim + 2),
            nn.Linear(parameter_dim + residual_dim + 2, fit_dim * enc_expand_ratio),
            nn.GELU(),
            nn.Linear(fit_dim * enc_expand_ratio, fit_dim),
            nn.GELU(),
        )
        self.physics_encoder = nn.Sequential(
            nn.LayerNorm(parameter_dim + violation_dim + 2),
            nn.Linear(parameter_dim + violation_dim + 2, physics_dim * enc_expand_ratio),
            nn.GELU(),
            nn.Linear(physics_dim * enc_expand_ratio, physics_dim),
            nn.GELU(),
        )
        self.to(self.device)

    def forward(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        parameter_state = state["parameter_state"].to(self.device)
        residual_state = state["residual_state"].to(self.device)
        violation_state = state["violation"].to(self.device)
        horizon_progress = state["horizon_progress"].to(self.device)
        done = state["done"].to(self.device)

        fit_input = torch.cat([parameter_state, residual_state, horizon_progress, done], dim=-1)
        physics_input = torch.cat([parameter_state, violation_state, horizon_progress, done], dim=-1)

        fit_feat = self.fit_encoder(fit_input)
        physics_feat = self.physics_encoder(physics_input)
        return fit_feat, physics_feat


class SquashedGaussianActor(nn.Module):
    def __init__(self, cfg: TrainConfig | None = None, feature_dim: int | None = None, device: torch.device | str | None = None):
        super().__init__()
        self.cfg = cfg or get_default_config()
        self.device = torch.device(device or infer_device(self.cfg))
        feature_dim = self.cfg.model.physics_dim + self.cfg.model.fit_dim
        actor_hidden_dim = feature_dim * self.cfg.model.actor_expand_ratio
        self.log_std_min = -5.0
        self.log_std_max = 2.0
        self.net = nn.Sequential(
            nn.Linear(feature_dim, actor_hidden_dim),
            nn.GELU(),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(actor_hidden_dim, self.cfg.model.action_dim)
        self.log_std_head = nn.Linear(actor_hidden_dim, self.cfg.model.action_dim)
        self.to(self.device)

    def _distribution(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Normal]:
        hidden = self.net(features)
        mean = self.mean_head(hidden)
        log_std = torch.clamp(self.log_std_head(hidden), self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return mean, log_std, Normal(mean, std)

    def sample(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, _, dist = self._distribution(features)
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action

    def mean_action(self, features: torch.Tensor) -> torch.Tensor:
        mean, _, _ = self._distribution(features)
        return torch.tanh(mean)


class DecomposedTwinCritic(nn.Module):
    def __init__(self, cfg: TrainConfig | None = None, fit_dim: int | None = None, physics_dim: int | None = None, action_dim: int | None = None, device: torch.device | str | None = None):
        super().__init__()
        self.cfg = cfg or get_default_config()
        self.device = torch.device(device or infer_device(self.cfg))
        fit_dim = fit_dim or self.cfg.model.fit_dim
        physics_dim = physics_dim or self.cfg.model.physics_dim
        action_dim = action_dim or self.cfg.model.action_dim

        def build_q(input_dim):
            h = input_dim
            r = self.cfg.model.critic_expand_ratio
            return nn.Sequential(
                nn.Linear(h, r*h), nn.GELU(),
                nn.Linear(r*h, r*h), nn.GELU(),
                nn.Linear(r*h, 1),
            )

        self.q_fit_1 = build_q(fit_dim + action_dim)
        self.q_fit_2 = build_q(fit_dim + action_dim)
        self.q_phys_1 = build_q(physics_dim + action_dim)
        self.q_phys_2 = build_q(physics_dim + action_dim)
        self.to(self.device)

    def forward(self, fit_feat: torch.Tensor, phys_feat: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action = action.to(fit_feat.device)
        fit_x = torch.cat([fit_feat, action], dim=-1)
        phys_x = torch.cat([phys_feat, action], dim=-1)
        return (
            self.q_fit_1(fit_x),
            self.q_fit_2(fit_x),
            self.q_phys_1(phys_x),
            self.q_phys_2(phys_x),
        )
