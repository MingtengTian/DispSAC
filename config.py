"""Configuration for isotropic SAC training."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

pwd = os.path.dirname(os.path.abspath(__file__))

SURF_ROOT = Path(pwd) / "CPS_running"
BSPLINE_DATA_ROOT = Path("/share/home/tianmingteng/Downloads/RL/SAC/iso/datafolder/1_100_36_new_criteria")
OUTPUT_DIR = Path(pwd) / "outputs"


@dataclass
class ModelConfig:
    fit_dim: int = 84
    physics_dim: int = 28
    action_dim: int = 10
    violation_dim: int = 16
    encoder_expand_ratio: int = 4
    critic_expand_ratio: int = 4
    actor_expand_ratio: int = 4
    

@dataclass
class TrainLoopConfig:
    total_updates: int = 1000
    max_episode_steps: int = 20
    save_every: int = 30
    validation_every: int = 15
    

@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 5e-5
    batch_size: int = 256
    replay_buffer_size: int = 100000
    warmup_steps: int = 5000
    updates_per_step: int = 4*2

    target_entropy: Optional[float] = -10
    init_alpha: float = 0.4
    min_alpha: float = 0.01

    replay_alpha: float = 0.6
    replay_beta: float = 0.4

    action_scale_moho: float = 0.015
    action_scale_crust_vs: float = 0.010
    action_scale_mantle_vs: float = 0.008

    reward_improve_weight: float = 2.0
    reward_final_misfit_weight: float = 0.5
    
    reward_violation_weight: float = 1.0
    reward_final_hard_invalid_weight: float = 1.0

    lambda_phys: float = 3.0
    
    last_mile_threshold: float = 0.03
    last_mile_floor: float = 0.005
    last_mile_decay: float = 0.95

    reward_abs_mscale: float = 20.0
    reward_rel_mscale: float = 5.0
    reward_vscale: float = 1.5
    
    last_mile_trigger_fraction: float = 0.10
    station_persistence: int = 3
    

@dataclass
class EnvConfig:
    batch_stations: int = 96
    forward_workers: int = 96
    noise_level: float = 0.05/5
    max_cut_ratio: float = 0.25

    period_count: int = 36

    rayleigh_phase_weight: float = 1.0
    rayleigh_group_weight: float = 1.0

    device: str = "cuda"
    max_depth_km: float = 200
    moho_min_km: float = 7.0
    moho_max_km: float = 70.0

    v_upper_crust_min: float = 2.2
    v_upper_crust_max: float = 3.5
    v_middle_crust_min: float = 3.0
    v_middle_crust_max: float = 3.7
    v_lower_crust_min: float = 3.5
    v_lower_crust_max: float = 4.2
    v_deep_crust_min: float = 3.8
    v_deep_crust_max: float = 4.4
    v_litho_min: float = 4.0
    v_litho_max: float = 4.7
    v_lab_min: float = 4.1
    v_lab_max: float = 4.6
    v_astheno_min: float = 4.2
    v_astheno_max: float = 4.7

    vpvs_min: float = 1.50
    vpvs_max: float = 2.60
    slope_max: float = 0.18
    slope_change_max: float = 0.055
    curv_max: float = 0.12
    mvs_std_max: float = 0.18


@dataclass
class DataConfig:
    pretrain_features_path: Path = BSPLINE_DATA_ROOT / "train_valid" / "pretrain_features.npz"
    crust_bspline_path: Path = BSPLINE_DATA_ROOT / "Crust_Bs_vs.txt"
    mantle_bspline_path: Path = BSPLINE_DATA_ROOT / "Mantle_Bs_vs.txt"
    surf_root: Path = SURF_ROOT
    random_sample_data_root: Path = BSPLINE_DATA_ROOT
    output_dir: Path = OUTPUT_DIR
    synthetic_eval_feature_path: Optional[Path] = None
    synthetic_eval_label_path: Optional[Path] = None
    validation_fraction: float = 0.1


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainLoopConfig = field(default_factory=TrainLoopConfig)

    def to_dict(self):
        return asdict(self)


RLFinetuneConfig = TrainConfig


def get_default_config() -> TrainConfig:
    cfg = TrainConfig()
    if cfg.sac.target_entropy is None:
        if cfg.model.action_dim > 10:
            cfg.sac.target_entropy = -10.0
        else:
            cfg.sac.target_entropy = -float(cfg.model.action_dim)
    return cfg
