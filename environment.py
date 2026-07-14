"""Batched GPU environment for isotropic SAC training.

Physics-aware violation penalty (v2):
  - Continuous severity score from `_compute_violation_score`.
  - tanh-normalized so penalty magnitude is bounded regardless of violation scale.
  - Mild step ramp (0.5 -> 1.0) emphasizes final-state validity for inference,
    but not so aggressively as to kill small late-step rewards.
  - Adaptive cap: penalty never exceeds `physics_penalty_max_fraction` of
    |base_reward|. This guarantees the net reward stays positive when the
    policy is improving misfit, regardless of how invalid the model is.
  - Single penalty term only (no separate final-state penalty).

Required new fields in `SACConfig`:
  violation_normalize_scale: float = 5.0       # tanh denominator for raw violation
  physics_penalty_weight: float = 1.5          # max penalty magnitude when fully invalid
  physics_penalty_max_fraction: float = 0.7    # cap as fraction of |base_reward|
  physics_penalty_min_floor: float = 0.05      # minimum penalty when invalid (overrides cap downward)
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from config import TrainConfig, get_default_config
from utils import (
    SurfBatchForward,
    ObservationBatch,
    _valid_initial_model_mask,
    _vp_from_vs,
    clamp_parameters,
    infer_device,
    load_bspline_bases,
    masked_l2_misfit,
    parameter_bounds,
    parameters_to_profiles,
    prepare_layered_model_from_params,
)


SCALES = {
    "vmin": 0.2,
    "vmax": 0.1,
    "monotone": 0.01,
    "basement": 0.2,
    "vpvs": 0.2,
    "crust_no_drop": 0.01,
    "moho_vs_negative_jump": 0.05,
    "moho_vs_positive_jump": 0.5,
    "crust_floor": 0.2,
    "variability": 0.2,
}


def _smooth_violation(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return F.softplus(x / scale)


def _compute_violation_score(
    params_physical: torch.Tensor,
    cfg: TrainConfig,
    crust_bs: torch.Tensor,
    mantle_bs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Continuous geophysical-validity severity score."""
    device = params_physical.device
    batch = params_physical.shape[0]

    profiles = parameters_to_profiles(
        params_physical, crust_bs, mantle_bs, cfg.env.max_depth_km
    )
    depth = profiles["depth"]
    vs = profiles["vs"]
    moho = params_physical[:, 0]

    components = []

    velocity_low = _smooth_violation(0.5 - vs, scale=SCALES["vmin"]).max(dim=1).values
    components.append(velocity_low)

    negative_depth = _smooth_violation(-depth, scale=SCALES["basement"]).max(dim=1).values
    components.append(negative_depth)

    depth_max = depth[:, -1]
    depth_mismatch = _smooth_violation(torch.abs(depth_max - cfg.env.max_depth_km) - 1e-3, scale=SCALES["basement"])
    components.append(depth_mismatch)

    non_increasing_depth = _smooth_violation(-(depth[:, 1:] - depth[:, :-1]), scale=SCALES["basement"]).max(dim=1).values
    components.append(non_increasing_depth)

    basement_violation = _smooth_violation(4.0 - vs[:, -1], scale=SCALES["basement"])
    components.append(basement_violation)

    vp = _vp_from_vs(vs)
    vpvs = vp / torch.clamp(vs, min=1e-6)
    vpvs_low = _smooth_violation(cfg.env.vpvs_min - vpvs, scale=SCALES["vpvs"]).max(dim=1).values
    vpvs_high = _smooth_violation(vpvs - cfg.env.vpvs_max, scale=SCALES["vpvs"]).max(dim=1).values
    components.append(vpvs_low)
    components.append(vpvs_high)

    crust_mask = depth < moho.unsqueeze(-1)
    crust_inner = crust_mask[:, 1:] & crust_mask[:, :-1]
    crust_diffs = torch.diff(vs, dim=1)
    crust_drop = (-crust_diffs).masked_fill(~crust_inner, float("-inf"))
    crust_drop_violation = torch.where(
        torch.isfinite(crust_drop.max(dim=1).values),
        crust_drop.max(dim=1).values,
        torch.zeros(batch, device=device),
    )
    components.append(_smooth_violation(crust_drop_violation, scale=SCALES["crust_no_drop"]))

    moho_idx = torch.argmin(torch.abs(depth - moho.unsqueeze(-1)), dim=1)
    n = depth.shape[1]
    valid_moho = (moho_idx > 0) & (moho_idx < n - 1)
    above = torch.clamp(moho_idx - 1, min=0)
    below = torch.clamp(moho_idx + 1, max=n - 1)
    batch_idx = torch.arange(batch, device=device)
    vs_above = vs[batch_idx, above]
    vs_below = vs[batch_idx, below]
    jump_vs = vs_below - vs_above
    moho_negative_jump = _smooth_violation(-jump_vs, scale=SCALES["moho_vs_negative_jump"])
    jump_too_large = _smooth_violation(jump_vs - 2.0, scale=SCALES["moho_vs_positive_jump"])
    crust_floor_violation = _smooth_violation(2.5 - vs_above, scale=SCALES["crust_floor"])
    components.append(torch.where(valid_moho, moho_negative_jump, torch.zeros_like(moho_negative_jump)))
    components.append(torch.where(valid_moho, jump_too_large, torch.zeros_like(jump_too_large)))
    components.append(torch.where(valid_moho, crust_floor_violation, torch.zeros_like(crust_floor_violation)))

    mantle_mask = depth >= moho.unsqueeze(-1)
    mantle_inner = mantle_mask[:, 1:] & mantle_mask[:, :-1]
    mantle_diffs = torch.diff(vs, dim=1)
    mantle_drop = (-mantle_diffs).masked_fill(~mantle_inner, float("-inf"))
    max_mantle_drop = torch.where(
        torch.isfinite(mantle_drop.max(dim=1).values),
        mantle_drop.max(dim=1).values,
        torch.zeros(batch, device=device),
    )
    first_slope = torch.where(
        mantle_inner,
        mantle_diffs,
        torch.zeros_like(mantle_diffs),
    )
    max_slope = first_slope.abs().max(dim=1).values
    slope_change = torch.diff(first_slope, dim=1).abs()
    if slope_change.numel() > 0:
        max_slope_change = slope_change.max(dim=1).values
    else:
        max_slope_change = torch.zeros(batch, device=device)
    second_grad = torch.diff(first_slope, dim=1)
    if second_grad.numel() > 0:
        max_curvature = second_grad.abs().max(dim=1).values
    else:
        max_curvature = torch.zeros(batch, device=device)
    mantle_slope_violation = _smooth_violation(max_slope - cfg.env.slope_max, scale=SCALES["monotone"])
    mantle_slope_change_violation = _smooth_violation(max_slope_change - cfg.env.slope_change_max, scale=SCALES["monotone"])
    mantle_curvature_violation = _smooth_violation(max_curvature - cfg.env.curv_max, scale=SCALES["monotone"])
    components.append(mantle_slope_violation)
    components.append(mantle_slope_change_violation)
    components.append(mantle_curvature_violation)

    mantle_count = mantle_mask.float().sum(dim=1)
    mantle_std_vs = torch.zeros(batch, device=device)
    for i in range(batch):
        if mantle_count[i] > 3:
            vals = vs[i, mantle_mask[i]]
            mantle_std_vs[i] = vals.std(unbiased=False)
    mantle_variation_violation = torch.where(
        mantle_count > 20,
        _smooth_violation(mantle_std_vs - cfg.env.mvs_std_max, scale=SCALES["variability"]),
        torch.zeros(batch, device=device),
    )
    components.append(mantle_variation_violation)

    runaway_vs = _smooth_violation(vs - 5.5, scale=SCALES["vmax"]).max(dim=1).values
    components.append(runaway_vs)

    component_tensor = torch.stack(components, dim=1)
    component_scales = torch.tensor(
        [
            SCALES["vmin"],
            SCALES["basement"],
            SCALES["basement"],
            SCALES["basement"],
            SCALES["basement"],
            SCALES["vpvs"],
            SCALES["vpvs"],
            SCALES["crust_no_drop"],
            SCALES["moho_vs_negative_jump"],
            SCALES["moho_vs_positive_jump"],
            SCALES["crust_floor"],
            SCALES["monotone"],
            SCALES["monotone"],
            SCALES["monotone"],
            SCALES["variability"],
            SCALES["vmax"],
        ],
        device=device,
        dtype=component_tensor.dtype,
    )
    component_weights = (1.0 / component_scales) / (1.0 / component_scales).sum()
    total_score = (component_tensor * component_weights.unsqueeze(0)).sum(dim=1)

    valid_threshold = torch.log(torch.tensor(2.0, device=device, dtype=component_tensor.dtype))
    valid_margin = torch.clamp(valid_threshold - component_tensor, min=0.0)
    reward_boost = torch.where(
        component_tensor.max(dim=1).values < valid_threshold,
        torch.log1p(valid_margin / torch.clamp(component_tensor, min=1e-6) * component_scales).sum(dim=1) / 100,
        torch.zeros(batch, device=device, dtype=component_tensor.dtype),
    )
    return total_score, component_tensor, reward_boost


def calculate_residual(pred: Dict[str, torch.Tensor], obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    ray_phase_residual = pred["rayleigh_phase"] - obs["rayleigh_phase"]
    ray_phase_residual = ray_phase_residual.masked_fill(obs["rayleigh_phase"] <= 0, 0)
    ray_group_residual = pred["rayleigh_group"] - obs["rayleigh_group"]
    ray_group_residual = ray_group_residual.masked_fill(obs["rayleigh_group"] <= 0, 0)
    return {"rayleigh_phase": ray_phase_residual, "rayleigh_group": ray_group_residual}


class BatchedDispersionEnv:
    def __init__(self, cfg: Optional[TrainConfig] = None):
        self.cfg = cfg or get_default_config()
        self.device = infer_device(self.cfg)
        print(f"[env.__init__] device={self.device}", flush=True)
        self.crust_bspline, self.mantle_bspline = load_bspline_bases(
            self.cfg.data.crust_bspline_path,
            self.cfg.data.mantle_bspline_path,
            device=self.device,
        )
        self.forward_model = SurfBatchForward(self.cfg, device=self.device)

        self.batch_size = 0
        self.step_count: Optional[torch.Tensor] = None
        self.done: Optional[torch.Tensor] = None
        self.observed_periods: Optional[torch.Tensor] = None
        self.observed_rayleigh_phase: Optional[torch.Tensor] = None
        self.observed_rayleigh_group: Optional[torch.Tensor] = None
        self.current_params: Optional[torch.Tensor] = None
        self.current_profiles: Optional[Dict[str, torch.Tensor]] = None
        self.current_pred: Optional[Dict[str, torch.Tensor]] = None
        self.current_residual: Optional[Dict[str, torch.Tensor]] = None
        self.last_mile_threshold = self.cfg.sac.last_mile_threshold
        self.last_action_scale_gain: Optional[torch.Tensor] = None
        self.current_violation_vector: Optional[torch.Tensor] = None
        self.current_total_violation: Optional[torch.Tensor] = None

    def close(self) -> None:
        self.forward_model.close()

    def update_last_mile_threshold(self, fraction_below_threshold: float) -> None:
        if fraction_below_threshold >= self.cfg.sac.last_mile_trigger_fraction:
            self.last_mile_threshold = max(
                self.cfg.sac.last_mile_floor,
                self.last_mile_threshold * self.cfg.sac.last_mile_decay,
            )
            anchor = torch.quantile(self.current_misfit, 0.25).item()
            self.last_mile_threshold = float(min(
                self.last_mile_threshold,
                max(self.cfg.sac.last_mile_floor, anchor),
            ))

    def reset(self, batch: ObservationBatch) -> Dict[str, torch.Tensor]:
        t0 = time.perf_counter()
        print("[env.reset] start", flush=True)
        self.observed_periods = batch.observed_periods.to(self.device)
        self.observed_rayleigh_phase = batch.observed_rayleigh_phase.to(self.device)
        self.observed_rayleigh_group = batch.observed_rayleigh_group.to(self.device)
        self.current_params = clamp_parameters(batch.initial_params.to(self.device), self.cfg)
        self.batch_size = self.current_params.shape[0]
        self.step_count = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self.done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self._refresh_forward_state(self.current_params)
        _, high = parameter_bounds(self.cfg, device=self.current_params.device)
        params_phys = self.current_params * high.unsqueeze(0)
        total_violation, violation_vector, violation_reward_boost = _compute_violation_score(
            params_phys, self.cfg, self.crust_bspline, self.mantle_bspline
        )
        
        self.current_total_violation = total_violation
        self.current_violation_vector = violation_vector

        print(
            f"[env.reset] batch={self.batch_size} period_len={self.observed_periods.shape[-1]} "
            f"params_dim={self.current_params.shape[-1]} misfit_mean={self.current_misfit.mean().item():.4f} "
            f"misfit_std={self.current_misfit.std().item():.4f} param_mean={self.current_params.mean().item():.4f} "
            f"param_std={self.current_params.std().item():.4f} elapsed={time.perf_counter() - t0:.2f}s",
            flush=True,
        )
        return self.get_state()

    def reset_keep_params(self, batch: ObservationBatch) -> Dict[str, torch.Tensor]:
        t0 = time.perf_counter()
        print("[env.reset_keep_params] start", flush=True)
        self.observed_periods = batch.observed_periods.to(self.device)
        self.observed_rayleigh_phase = batch.observed_rayleigh_phase.to(self.device)
        self.observed_rayleigh_group = batch.observed_rayleigh_group.to(self.device)
        if self.current_params is not None and self.current_params.shape[0] == batch.initial_params.shape[0]:
            keep = self.current_params
        else:
            keep = clamp_parameters(batch.initial_params.to(self.device), self.cfg)
        self.current_params = clamp_parameters(keep, self.cfg)
        self.batch_size = self.current_params.shape[0]
        self.step_count = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self.done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self._refresh_forward_state(self.current_params)
        _, high = parameter_bounds(self.cfg, device=self.current_params.device)
        params_phys = self.current_params * high.unsqueeze(0)
        total_violation, violation_vector, violation_reward_boost = _compute_violation_score(
            params_phys, self.cfg, self.crust_bspline, self.mantle_bspline
        )
        
        self.current_total_violation = total_violation
        self.current_violation_vector = violation_vector

        print(
            f"[env.reset_keep_params] batch={self.batch_size} misfit_mean={self.current_misfit.mean().item():.4f} "
            f"misfit_std={self.current_misfit.std().item():.4f} elapsed={time.perf_counter() - t0:.2f}s",
            flush=True,
        )
        return self.get_state()

    def _refresh_forward_state(self, params: torch.Tensor) -> None:
        self.current_profiles = prepare_layered_model_from_params(
            params,
            self.crust_bspline,
            self.mantle_bspline,
            self.cfg,
        )
        self.current_pred = self.forward_model.forward(
            self.observed_periods,
            self.current_profiles["thickness"],
            self.current_profiles["vp_mid"],
            self.current_profiles["vs_mid"],
            self.current_profiles["rho_mid"],
        )
        self.current_residual = calculate_residual(
            self.current_pred,
            {"rayleigh_phase": self.observed_rayleigh_phase, "rayleigh_group": self.observed_rayleigh_group},
        )
        self.current_misfit = self._compute_total_misfit(self.current_pred)

    def _compute_total_misfit(self, pred: Dict[str, torch.Tensor]) -> torch.Tensor:
        rayleigh_phase = masked_l2_misfit(pred["rayleigh_phase"], self.observed_rayleigh_phase)
        rayleigh_group = masked_l2_misfit(pred["rayleigh_group"], self.observed_rayleigh_group)
        return (
            self.cfg.env.rayleigh_phase_weight * rayleigh_phase + self.cfg.env.rayleigh_group_weight * rayleigh_group
        ) / (self.cfg.env.rayleigh_phase_weight + self.cfg.env.rayleigh_group_weight)

    def _build_residual_state(self) -> torch.Tensor:
        return torch.cat(
            [self.current_residual["rayleigh_phase"], self.current_residual["rayleigh_group"]],
            dim=-1,
        )

    def get_state(self) -> Dict[str, torch.Tensor]:
        horizon = max(1, self.cfg.train.max_episode_steps)
        return {
            "dispersion_input": torch.stack(
                [self.observed_periods, self.observed_rayleigh_phase, self.observed_rayleigh_group],
                dim=1,
            ),
            "parameter_state": self.current_params,
            "residual_state": self._build_residual_state(),
            "misfit": self.current_misfit.unsqueeze(-1),
            "total_violation": self.current_total_violation.unsqueeze(-1),
            "violation": self.current_violation_vector,
            "horizon_progress": (self.step_count.float() / float(horizon)).unsqueeze(-1),
            "done": self.done.float().unsqueeze(-1),
        }

    def scale_action_by_group(self, action: torch.Tensor) -> torch.Tensor:
        scaled = action.clone()
        scaled[..., 0] *= self.cfg.sac.action_scale_moho
        scaled[..., 1:5] *= self.cfg.sac.action_scale_crust_vs
        scaled[..., 5:10] *= self.cfg.sac.action_scale_mantle_vs
        gain = torch.clamp(20 * self.current_misfit.unsqueeze(-1), min=0.5, max=3.0)
        return scaled * gain

    def _compute_done_mask(self) -> torch.Tensor:
        return self.step_count >= self.cfg.train.max_episode_steps

    def _compute_reward(
        self,
        prev_misfit: torch.Tensor,
        new_misfit: torch.Tensor,
        prev_violation: torch.Tensor,
        new_violation: torch.Tensor,
        violation_reward_boost: torch.Tensor,
        done: torch.Tensor,
        hard_invalid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        abs_improve = torch.tanh(self.cfg.sac.reward_abs_mscale * (prev_misfit - new_misfit))
        rel_improve = torch.tanh(
            self.cfg.sac.reward_rel_mscale
            * (prev_misfit - new_misfit)
            / (new_misfit + 0.001)
        )
        misfit_improve = abs_improve + rel_improve
        last_mile_threshold = max(self.last_mile_threshold, 1e-6)
        last_mile_focus = torch.clamp((last_mile_threshold - new_misfit) / last_mile_threshold, min=0.0)
        misfit_improve = misfit_improve * (1.0 + 1.5 * last_mile_focus)

        violation_abs_improve = torch.tanh(self.cfg.sac.reward_vscale * (prev_violation - new_violation))
        violation_improve = violation_abs_improve + violation_reward_boost

        final_term = torch.where(done, -new_misfit, torch.zeros_like(new_misfit))
        final_hard_invalid_term = torch.where(
            done, -hard_invalid, torch.zeros_like(new_violation)
        )
        reward_fit = (
            self.cfg.sac.reward_improve_weight * misfit_improve
            + self.cfg.sac.reward_final_misfit_weight * final_term
        )
        reward_phys = (
            self.cfg.sac.reward_violation_weight * violation_improve
            + self.cfg.sac.reward_final_hard_invalid_weight * final_hard_invalid_term
        )
        return reward_fit, reward_phys, violation_improve

    def step(self, action_delta_norm: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        action_delta_norm = action_delta_norm.to(self.device)
        action_delta_norm = torch.where(self.done.unsqueeze(-1), torch.zeros_like(action_delta_norm), action_delta_norm)
        scaled_action = self.scale_action_by_group(action_delta_norm)
        prev_misfit = self.current_misfit.clone()
        prev_violation = self.current_total_violation.clone()

        # ---- Apply action: per-dim clamp keeps params numerically safe.
        new_params = clamp_parameters(self.current_params + scaled_action, self.cfg)

        # ---- Physical violation score (continuous, >= 0)
        _, high = parameter_bounds(self.cfg, device=new_params.device)
        new_params_physical = new_params * high.unsqueeze(0)
        total_violation, violation_vector, violation_reward_boost = _compute_violation_score(
            new_params_physical,
            self.cfg,
            crust_bs=self.crust_bspline,
            mantle_bs=self.mantle_bspline,
        )
        valid_mask = _valid_initial_model_mask(
            new_params_physical,
            self.cfg,
            crust_bs=self.crust_bspline,
            mantle_bs=self.mantle_bspline,
        )
        # ---- Forward simulation
        new_profiles = prepare_layered_model_from_params(
            new_params,
            self.crust_bspline,
            self.mantle_bspline,
            self.cfg,
        )
        new_pred = self.forward_model.forward(
            self.observed_periods,
            new_profiles["thickness"],
            new_profiles["vp_mid"],
            new_profiles["vs_mid"],
            new_profiles["rho_mid"],
        )
        new_residual = calculate_residual(
            new_pred,
            {"rayleigh_phase": self.observed_rayleigh_phase, "rayleigh_group": self.observed_rayleigh_group},
        )
        new_misfit = self._compute_total_misfit(new_pred)

        # ---- Step bookkeeping
        self.step_count = self.step_count + (~self.done).long()
        new_done = self._compute_done_mask()

        reward_fit, reward_phys, violation_improve = self._compute_reward(
            prev_misfit,
            new_misfit,
            prev_violation,
            total_violation,
            violation_reward_boost,
            new_done,
            hard_invalid=(~valid_mask).float(),
        )

        self.current_total_violation = total_violation
        self.current_violation_vector = violation_vector

        reward = reward_fit + self.cfg.sac.lambda_phys * reward_phys
        reward_fit = torch.where(self.done, torch.zeros_like(reward_fit), reward_fit)
        reward_phys = torch.where(self.done, torch.zeros_like(reward_phys), reward_phys)
        reward = torch.where(self.done, torch.zeros_like(reward), reward)
        positive_boost = violation_reward_boost[violation_reward_boost > 0]

        self.current_params = new_params
        self.current_profiles = new_profiles
        self.current_pred = new_pred
        self.current_residual = new_residual
        self.current_misfit = new_misfit
        self.done = new_done
        
        info = {
            "misfit": self.current_misfit,
            "reward_fit": reward_fit,
            "reward_phys": reward_phys,
            "reward": reward,
            "done": self.done,
            "step_count": self.step_count,
            "frac_below_003": float((self.current_misfit <= 0.03).float().mean().item()),
            "last_mile_threshold": float(self.last_mile_threshold),
            "violation_score_mean": float(total_violation.mean().item()),
            "violation_score_max": float(total_violation.max().item()),
            "violation_improve_mean": float(violation_improve.mean().item()),
            "violation_reward_boost_mean": float(positive_boost.mean().item()) if positive_boost.numel() else 0.0,
            "frac_invalid": float((~valid_mask).float().mean().item())
        }
        print(
            f"[env.step] step={int(self.step_count.max().item())}/{self.cfg.train.max_episode_steps} "
            f"reward_mean={reward.mean().item():.4f} reward_std={reward.std().item():.4f} ",
            flush=True,
        )
        print(
            f"[env.step] misfit_mean={self.current_misfit.mean().item():.4f} prev_misfit_mean={prev_misfit.mean().item():.4f} "
            f"misfit_delta={(self.current_misfit.mean() - prev_misfit.mean()).item():.4f} ",
            flush=True,
        )
        print(
            f"[env.step] violation_score_mean={info['violation_score_mean']:.4f} violation_improve_mean={info['violation_improve_mean']:.4f} "
            f"violation_boost_mean={info['violation_reward_boost_mean']:.4f} frac_invalid={info['frac_invalid']:.2%}", 
            flush=True,
        )
        print(
            f"[env.step] done={int(self.done.sum().item())}/{self.batch_size} "
            f"params_action_mean={action_delta_norm.norm(dim=-1).mean().item():.4f} "
            f"params_scaled_action_mean={scaled_action.norm(dim=-1).mean().item():.4f}",
            flush=True,
        )
        return self.get_state(), reward, self.done, info
