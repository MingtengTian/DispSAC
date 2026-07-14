"""Isotropic SAC training entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from tqdm.auto import tqdm

from config import TrainConfig, get_default_config
from environment import BatchedDispersionEnv
from augmentation import augmentation
from networks import DecomposedTwinCritic, SquashedGaussianActor, StateEncoder
from replay_buffer import ReplayBuffer
from sac import SACAgent
from utils import ObservationBatch, infer_device, load_feature_array, sample_random_initial_parameters


def _fmt_metric(value):
    return "None" if value is None else f"{float(value):.4f}"


def split_train_validation_features(cfg: TrainConfig) -> tuple[torch.Tensor, torch.Tensor]:
    features = load_feature_array(cfg.data.pretrain_features_path)
    tensor = torch.tensor(features, dtype=torch.float32)
    n_total = tensor.shape[0]
    n_val = max(1, int(n_total * cfg.data.validation_fraction))
    return tensor[n_val:], tensor[:n_val]


def make_observation_batch_from_features(
    cfg: TrainConfig,
    features: torch.Tensor,
    device: torch.device,
) -> ObservationBatch:
    n_pick = min(cfg.env.batch_stations, features.shape[0])
    perm = torch.randperm(features.shape[0])[:n_pick]
    picked = features[perm].to(device)
    picked = augmentation(picked, noise_level=cfg.env.noise_level, max_cut_ratio=cfg.env.max_cut_ratio)
    init_params = sample_random_initial_parameters(cfg, n_pick, device=device)
    return ObservationBatch(
        observed_periods=picked[:, :, 0],
        observed_rayleigh_phase=picked[:, :, 1],
        observed_rayleigh_group=picked[:, :, 2],
        initial_params=init_params,
    )


def build_sac_components(cfg: TrainConfig) -> Dict[str, object]:
    device = infer_device(cfg)
    env = BatchedDispersionEnv(cfg)
    encoder = StateEncoder(cfg, device=device)
    actor = SquashedGaussianActor(cfg, device=device)
    critic = DecomposedTwinCritic(cfg, device=device)
    agent = SACAgent(cfg, encoder=encoder, actor=actor, critic=critic, device=device)
    buffer = ReplayBuffer(cfg.sac.replay_buffer_size, device=device, alpha=cfg.sac.replay_alpha)
    return {"env": env, "agent": agent, "buffer": buffer, "device": device}


def collect_transition_batch(state: Dict[str, torch.Tensor], action: torch.Tensor, reward_fit: torch.Tensor, reward_phys: torch.Tensor, next_state: Dict[str, torch.Tensor], done: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "dispersion_input": state["dispersion_input"],
        "parameter_state": state["parameter_state"],
        "residual_state": state["residual_state"],
        "misfit": state["misfit"],
        "total_violation": state["total_violation"],
        "violation": state["violation"],
        "horizon_progress": state["horizon_progress"],
        "done": state["done"],
        "action": action,
        "reward_fit": reward_fit.unsqueeze(-1),
        "reward_phys": reward_phys.unsqueeze(-1),
        "next_dispersion_input": next_state["dispersion_input"],
        "next_parameter_state": next_state["parameter_state"],
        "next_residual_state": next_state["residual_state"],
        "next_misfit": next_state["misfit"],
        "next_total_violation": next_state["total_violation"],
        "next_violation": next_state["violation"],
        "next_horizon_progress": next_state["horizon_progress"],
        "next_done": next_state["done"],
        "terminal_done": done.unsqueeze(-1).float(),
    }


def transition_priorities(next_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    next_misfit = next_state["misfit"].squeeze(-1)
    next_total_violation = next_state["total_violation"].squeeze(-1)
    return 1.0 / (next_misfit + 0.01) + 10 * next_total_violation


def evaluate_agent(
    cfg: TrainConfig,
    agent: SACAgent,
    features: torch.Tensor,
    device: torch.device,
    deterministic: bool = True,
) -> Dict[str, float]:
    env = BatchedDispersionEnv(cfg)
    final_misfits = []
    try:
        batch = make_observation_batch_from_features(cfg, features, device)
        state = env.reset(batch)
        for _ in range(cfg.train.max_episode_steps * cfg.sac.station_persistence):
            action = agent.select_action(state, deterministic=deterministic)
            state, *_ = env.step(action)
        final_misfits.append(env.current_misfit.detach())
    finally:
        env.close()
    merged = torch.cat(final_misfits)
    return {
        "mean_final_misfit": float(merged.mean().item()),
        "median_final_misfit": float(merged.median().item()),
        "fraction_leq_003": float((merged <= 0.03).float().mean().item()),
    }


def save_checkpoint(cfg: TrainConfig, agent: SACAgent, update: int, global_step: int) -> Path:
    cfg.data.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = cfg.data.output_dir / f"sac_update_{update}.pt"
    torch.save(
        {
            "encoder": agent.encoder.state_dict(),
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict(),
            "target_critic": agent.target_critic.state_dict(),
            "log_alpha": agent.log_alpha.detach().cpu(),
            "config": cfg.to_dict(),
            "update": update,
            "global_step": global_step,
        },
        ckpt,
    )
    return ckpt


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(42)
    device = infer_device(cfg)
    print("[train] Starting SAC training")
    print(f"[train] seed=42 device={device}")
    print(
        f"[train] total_updates={cfg.train.total_updates} batch_stations={cfg.env.batch_stations} "
        f"max_episode_steps={cfg.train.max_episode_steps} station_persistence={cfg.sac.station_persistence} "
        f"warmup_steps={cfg.sac.warmup_steps} replay_ratio={cfg.sac.updates_per_step}"
    )
    print(f"[train] feature_path={cfg.data.pretrain_features_path}")
    print(f"[train] output_dir={cfg.data.output_dir}")
    
    components = build_sac_components(cfg)
    env: BatchedDispersionEnv = components["env"]
    agent: SACAgent = components["agent"]
    buffer: ReplayBuffer = components["buffer"]
    device: torch.device = components["device"]

    try:
        train_features, validation_features = split_train_validation_features(cfg)
        print(
            f"[train] train_features={tuple(train_features.shape)} validation_features={tuple(validation_features.shape)}"
        )

        updates = range(1, cfg.train.total_updates + 1)
        pbar = tqdm(updates, total=cfg.train.total_updates, desc="SAC", dynamic_ncols=True)
        global_step = 0
        persistent_batch = None
        persistent_age = 0

        for update in pbar:
            sampled_new_cohort = persistent_batch is None or persistent_age >= cfg.sac.station_persistence
            if sampled_new_cohort:
                persistent_batch = make_observation_batch_from_features(cfg, train_features, device)
                persistent_age = 0
                print(
                    f"[train] update={update:05d} sampled new cohort batch={persistent_batch.initial_params.shape[0]} "
                    f"period_len={persistent_batch.observed_periods.shape[-1]}"
                )
            persistent_age += 1
            if persistent_age == 1:
                print(f"[train] update={update:05d} reset from sampled initial parameters")
                state = env.reset(persistent_batch)
            else:
                print(
                    f"[train] update={update:05d} continuing cohort from previous final parameters "
                    f"cohort_age={persistent_age}/{cfg.sac.station_persistence}"
                )
                state = env.reset_keep_params(persistent_batch)

            action_mode = "warmup-random" if global_step < cfg.sac.warmup_steps else "policy"
            print(
                f"[train] update={update:05d} start global_step={global_step} buffer_size={len(buffer)} action_mode={action_mode}"
            )

            episode_rewards = []
            latest_metrics: Dict[str, float | torch.Tensor] = {}

            for step_idx in range(cfg.train.max_episode_steps):
                if global_step < cfg.sac.warmup_steps:
                    action = torch.empty(env.batch_size, cfg.model.action_dim, device=device).uniform_(-1.0, 1.0)
                else:
                    is_last_step = step_idx == cfg.train.max_episode_steps - 1
                    action = agent.select_action(state, deterministic=is_last_step)
                print(
                    f"[train.rollout] update={update:05d} step={step_idx + 1}/{cfg.train.max_episode_steps} "
                    f"action_mean={action.mean().item():.4f} action_std={action.std().item():.4f} "
                    f"misfit_mean={state['misfit'].mean().item():.4f}"
                )

                next_state, reward, done, info = env.step(action)
                priorities = transition_priorities(next_state)
                buffer.add_batch(
                    collect_transition_batch(
                        state,
                        action,
                        info["reward_fit"],
                        info["reward_phys"],
                        next_state,
                        done,
                    ),
                    priorities=priorities,
                )
                state = next_state
                episode_rewards.append(reward.mean().item())
                global_step += env.batch_size

                print(
                    f"[train.rollout] update={update:05d} step={step_idx + 1}/{cfg.train.max_episode_steps} done"
                )
                print(
                    f"[train.rollout] reward_mean={reward.mean().item():.4f} reward_std={reward.std().item():.4f} misfit_mean={env.current_misfit.mean().item():.4f} "
                )
                print(
                    f"[train.rollout] priority_mean={priorities.mean().item():.4f} priority_max={priorities.max().item():.4f} "
                )
                print(
                    f"[train.rollout] done_count={int(done.sum().item())}/{env.batch_size} "
                )

                if len(buffer) >= cfg.sac.batch_size and global_step >= cfg.sac.warmup_steps:
                    for replay_idx in range(cfg.sac.updates_per_step):
                        sample = buffer.sample_prioritized(cfg.sac.batch_size, beta=cfg.sac.replay_beta)
                        indices = sample.pop("indices")
                        weights = sample.pop("weights")
                        latest_metrics = agent.update(sample, weights=weights)
                        td_error = latest_metrics["td_error"]
                        new_priorities = td_error.abs() + 1e-3
                        buffer.update_priorities(indices, new_priorities)
                        print(
                            f"[train.update] update={update:05d} rollout_step={step_idx + 1} replay={replay_idx + 1}/{cfg.sac.updates_per_step} "
                        )
                        print(
                            f"[train.update] critic_loss={_fmt_metric(latest_metrics.get('critic_loss'))} "
                            f"critic_loss_fit={_fmt_metric(latest_metrics.get('critic_loss_fit'))} "
                            f"critic_loss_phys={_fmt_metric(latest_metrics.get('critic_loss_phys'))} "
                        )
                        print(
                            f"[train.update] actor_loss={_fmt_metric(latest_metrics.get('actor_loss'))} "
                            f"alpha={_fmt_metric(latest_metrics.get('alpha'))} "
                            f"alpha_loss={_fmt_metric(latest_metrics.get('alpha_loss'))} "
                        )
                        print(
                            f"[train.update] q_fit_1_mean={_fmt_metric(latest_metrics.get('q_fit_1_mean'))} "
                            f"q_fit_2_mean={_fmt_metric(latest_metrics.get('q_fit_2_mean'))} "
                            f"q_phys_1_mean={_fmt_metric(latest_metrics.get('q_phys_1_mean'))} "
                            f"q_phys_2_mean={_fmt_metric(latest_metrics.get('q_phys_2_mean'))} "
                        )
                        print(
                            f"[train.update] target_fit_mean={_fmt_metric(latest_metrics.get('target_fit_mean'))} "
                            f"target_phys_mean={_fmt_metric(latest_metrics.get('target_phys_mean'))} "
                            f"mean_log_prob={_fmt_metric(latest_metrics.get('mean_log_prob'))} "
                            f"target_entropy_gap={_fmt_metric(latest_metrics.get('target_entropy_gap'))} "
                            f"td_mean={td_error.mean().item():.4f} td_std={td_error.std().item():.4f}"
                        )

            train_mean_misfit = float(env.current_misfit.mean().item())
            mean_reward = float(sum(episode_rewards) / max(len(episode_rewards), 1))
            frac_below_threshold = float((env.current_misfit <= env.last_mile_threshold).float().mean().item())
            env.update_last_mile_threshold(frac_below_threshold)
            mean_priority = float(buffer.priorities[: len(buffer)].mean().item()) if len(buffer) > 0 else 0.0
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(misfit=f"{train_mean_misfit:.4f}", invalid_fraction=f"{info['frac_invalid']:.2%}")

            print(
                f"\n[update {update:05d}] reward={mean_reward:.4f} train_misfit={train_mean_misfit:.4f}"
            )
            print(
                f"[update {update:05d}] critic_loss={_fmt_metric(latest_metrics.get('critic_loss'))} "
                f"critic_loss_fit={_fmt_metric(latest_metrics.get('critic_loss_fit'))} "
                f"critic_loss_phys={_fmt_metric(latest_metrics.get('critic_loss_phys'))}"
            )
            print(
                f"[update {update:05d}] actor_loss={_fmt_metric(latest_metrics.get('actor_loss'))} "
                f"alpha={_fmt_metric(latest_metrics.get('alpha'))} alpha_loss={_fmt_metric(latest_metrics.get('alpha_loss'))}"
            )
            print(
                f"[update {update:05d}] q_fit_1_mean={_fmt_metric(latest_metrics.get('q_fit_1_mean'))} "
                f"q_fit_2_mean={_fmt_metric(latest_metrics.get('q_fit_2_mean'))} "
                f"q_phys_1_mean={_fmt_metric(latest_metrics.get('q_phys_1_mean'))} "
                f"q_phys_2_mean={_fmt_metric(latest_metrics.get('q_phys_2_mean'))}"
            )
            print(
                f"[update {update:05d}] target_fit_mean={_fmt_metric(latest_metrics.get('target_fit_mean'))} "
                f"target_phys_mean={_fmt_metric(latest_metrics.get('target_phys_mean'))} "
                f"mean_log_prob={_fmt_metric(latest_metrics.get('mean_log_prob'))} "
                f"target_entropy_gap={_fmt_metric(latest_metrics.get('target_entropy_gap'))}"
            )
            print(
                f"[update {update:05d}] frac_below_003={float(info['frac_below_003']):.2%} "
                f"last_mile_threshold={env.last_mile_threshold:.4f} mean_priority={mean_priority:.4f}"
            )
            print(
                f"[update {update:05d}] cohort_age={persistent_age}/{cfg.sac.station_persistence} "
                f"buffer_size={len(buffer)} global_step={global_step} threshold={env.last_mile_threshold:.4f}"
            )

            if update % cfg.train.save_every == 0:
                ckpt = save_checkpoint(cfg, agent, update, global_step)
                print(f"Saved checkpoint -> {ckpt}")

            if update % cfg.train.validation_every == 0:
                try:
                    val_metrics = evaluate_agent(cfg, agent, validation_features, device, deterministic=True)
                    print(
                        f"[validation {update:05d}] mean_final_misfit={val_metrics['mean_final_misfit']:.4f} "
                        f"median_final_misfit={val_metrics['median_final_misfit']:.4f} <=0.03={val_metrics['fraction_leq_003']:.2%} "
                    )
                except Exception as e:
                    print(f"[validation {update:05d}] FAILED with {type(e).__name__}: {e} -- continuing training")
    finally:
        env.close()


if __name__ == "__main__":
    train(get_default_config())
