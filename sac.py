from __future__ import annotations

import copy, math
from typing import Dict

import torch

from config import TrainConfig, get_default_config
from networks import DecomposedTwinCritic, SquashedGaussianActor, StateEncoder
from utils import infer_device


class SACAgent:
    def __init__(
        self,
        cfg: TrainConfig | None = None,
        encoder: StateEncoder | None = None,
        actor: SquashedGaussianActor | None = None,
        critic: DecomposedTwinCritic | None = None,
        device: torch.device | str | None = None,
    ):
        self.cfg = cfg or get_default_config()
        self.device = torch.device(device or infer_device(self.cfg))
        self.encoder = encoder or StateEncoder(self.cfg, device=self.device)
        self.actor = actor or SquashedGaussianActor(self.cfg, device=self.device)
        self.critic = critic or DecomposedTwinCritic(self.cfg, device=self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.target_critic.eval()

        init_alpha = torch.tensor(self.cfg.sac.init_alpha, dtype=torch.float32, device=self.device)
        self.log_alpha = torch.nn.Parameter(init_alpha.log())
        self.target_entropy = float(self.cfg.sac.target_entropy)

        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.cfg.sac.critic_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.sac.critic_learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.sac.actor_learning_rate)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.cfg.sac.alpha_learning_rate)
        self.update_step = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _state_from_batch(self, batch: Dict[str, torch.Tensor], prefix: str = "") -> Dict[str, torch.Tensor]:
        return {
            "dispersion_input": batch[f"{prefix}dispersion_input"].to(self.device),
            "parameter_state": batch[f"{prefix}parameter_state"].to(self.device),
            "residual_state": batch[f"{prefix}residual_state"].to(self.device),
            "misfit": batch[f"{prefix}misfit"].to(self.device),
            "total_violation": batch[f"{prefix}total_violation"].to(self.device),
            "violation": batch[f"{prefix}violation"].to(self.device),
            "horizon_progress": batch[f"{prefix}horizon_progress"].to(self.device),
            "done": batch[f"{prefix}done"].to(self.device),
        }

    def select_action(self, state: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            fit_feat, phys_feat = self.encoder(state)
            actor_features = torch.cat([fit_feat, phys_feat], dim=-1)
            if deterministic:
                return self.actor.mean_action(actor_features)
            action, _, _ = self.actor.sample(actor_features)
            return action

    def soft_update_targets(self) -> None:
        q_pairs = (
            (self.target_critic.q_fit_1, self.critic.q_fit_1),
            (self.target_critic.q_fit_2, self.critic.q_fit_2),
            (self.target_critic.q_phys_1, self.critic.q_phys_1),
            (self.target_critic.q_phys_2, self.critic.q_phys_2),
        )
        for target_net, net in q_pairs:
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.mul_(1.0 - self.cfg.sac.tau)
                target_param.data.add_(self.cfg.sac.tau * param.data)

    def update(self, batch: Dict[str, torch.Tensor], weights: torch.Tensor | None = None) -> Dict[str, float | torch.Tensor]:
        self.update_step += 1
        state = self._state_from_batch(batch)
        next_state = self._state_from_batch(batch, prefix="next_")
        action = batch["action"].to(self.device)
        reward_fit = batch["reward_fit"].to(self.device)
        reward_phys = batch["reward_phys"].to(self.device)
        done = batch["terminal_done"].to(self.device)
        if weights is None:
            weights = torch.ones_like(reward_fit, device=self.device)
        else:
            weights = weights.to(self.device)
            if weights.ndim == 1:
                weights = weights.unsqueeze(-1)

        if self.update_step <= 3 or self.update_step % 25 == 0:
            print(
                f"\n[SACAgent.update] step={self.update_step} batch={tuple(action.shape)} reward_fit_mean={reward_fit.mean().item():.4f} "
                f"reward_phys_mean={reward_phys.mean().item():.4f} done_frac={done.float().mean().item():.4f} "
                f"weight_mean={weights.mean().item():.4f}"
            )

        fit_feat, phys_feat = self.encoder(state)
        with torch.no_grad():
            next_fit_feat, next_phys_feat = self.encoder(next_state)
            next_actor_features = torch.cat([next_fit_feat, next_phys_feat], dim=-1)
            next_action, next_log_prob, _ = self.actor.sample(next_actor_features)
            t_q_fit_1, t_q_fit_2, t_q_phys_1, t_q_phys_2 = self.target_critic(next_fit_feat, next_phys_feat, next_action)
            target_fit = reward_fit + self.cfg.sac.gamma * (1.0 - done) * (torch.min(t_q_fit_1, t_q_fit_2)  - self.alpha.detach() * next_log_prob)
            target_phys = reward_phys + self.cfg.sac.gamma * (1.0 - done) * (torch.min(t_q_phys_1, t_q_phys_2) - self.alpha.detach() * next_log_prob)

        q_fit_1, q_fit_2, q_phys_1, q_phys_2 = self.critic(fit_feat, phys_feat, action)
        td_error_fit_1 = (q_fit_1 - target_fit).abs()
        td_error_fit_2 = (q_fit_2 - target_fit).abs()
        td_error_phys_1 = (q_phys_1 - target_phys).abs()
        td_error_phys_2 = (q_phys_2 - target_phys).abs()
        td_error = td_error_fit_1 + td_error_fit_2 + td_error_phys_1 + td_error_phys_2
        critic_loss_fit = ((q_fit_1 - target_fit).pow(2) * weights.detach()).mean() + ((q_fit_2 - target_fit).pow(2) * weights.detach()).mean()
        critic_loss_phys = ((q_phys_1 - target_phys).pow(2) * weights.detach()).mean() + ((q_phys_2 - target_phys).pow(2) * weights.detach()).mean()
        critic_loss = critic_loss_fit + critic_loss_phys

        self.encoder_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.encoder_optimizer.step()
        self.critic_optimizer.step()
        self.encoder.zero_grad(set_to_none=True)

        actor_fit_feat, actor_phys_feat = self.encoder(state)
        actor_fit_feat = actor_fit_feat.detach()
        actor_phys_feat = actor_phys_feat.detach()
        actor_features = torch.cat([actor_fit_feat, actor_phys_feat], dim=-1)
        for p in self.critic.parameters():
            p.requires_grad_(False)
        new_action, log_prob, _ = self.actor.sample(actor_features)
        q_fit_1_pi, q_fit_2_pi, q_phys_1_pi, q_phys_2_pi = self.critic(actor_fit_feat, actor_phys_feat, new_action)
        total_q = torch.min(q_fit_1_pi, q_fit_2_pi) + self.cfg.sac.lambda_phys * torch.min(q_phys_1_pi, q_phys_2_pi)
        actor_loss = (self.alpha.detach() * log_prob - total_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)
        actor_loss_value = float(actor_loss.item())
        mean_log_prob_value = float(log_prob.mean().item())
        target_entropy_gap_value = float((-log_prob.mean() - self.target_entropy).item())

        alpha_loss = -(self.alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            self.log_alpha.clamp_(min=math.log(self.cfg.sac.min_alpha))

        alpha_loss_value = float(alpha_loss.item())

        self.soft_update_targets()

        td_error_detached = td_error.detach().squeeze(-1)

        return {
            "critic_loss": float(critic_loss.item()),
            "critic_loss_fit": float(critic_loss_fit.item()),
            "critic_loss_phys": float(critic_loss_phys.item()),
            "actor_loss": actor_loss_value,
            "alpha": float(self.alpha.item()),
            "alpha_loss": alpha_loss_value,
            "q_fit_1_mean": float(q_fit_1.mean().item()),
            "q_fit_2_mean": float(q_fit_2.mean().item()),
            "q_phys_1_mean": float(q_phys_1.mean().item()),
            "q_phys_2_mean": float(q_phys_2.mean().item()),
            "target_fit_mean": float(target_fit.mean().item()),
            "target_phys_mean": float(target_phys.mean().item()),
            "mean_log_prob": mean_log_prob_value,
            "target_entropy_gap": target_entropy_gap_value,
            "td_error": td_error_detached,
        }
