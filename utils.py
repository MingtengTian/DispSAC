from __future__ import annotations

import inspect
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
from config import TrainConfig, get_default_config


@dataclass
class ObservationBatch:
    observed_periods: torch.Tensor
    observed_rayleigh_phase: torch.Tensor
    observed_rayleigh_group: torch.Tensor
    initial_params: torch.Tensor


def ensure_external_paths(cfg: TrainConfig) -> None:
    """Make local project dependencies importable."""
    for path in [cfg.data.surf_root]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path | str,
    device: Optional[torch.device | str] = None,
    strict: bool = False,
) -> nn.Module:
    """Load checkpoint using the repository's existing pattern."""
    device = torch.device(device or "cpu")
    print(f"[utils.load_checkpoint] loading {checkpoint_path} on {device}", flush=True)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    model_state = state_dict["model"] if isinstance(state_dict, dict) and "model" in state_dict else state_dict
    print(
        f"[utils.load_checkpoint] checkpoint keys={len(model_state) if hasattr(model_state, 'keys') else 'n/a'} strict={strict}",
        flush=True,
    )
    model.load_state_dict(model_state, strict=strict)
    model.to(device)
    print("[utils.load_checkpoint] model weights loaded", flush=True)
    return model


def load_bspline_bases(
    crust_path: Path | str,
    mantle_path: Path | str,
    device: Optional[torch.device | str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    c_bs = torch.tensor(np.loadtxt(crust_path), dtype=torch.float32, device=device)
    m_bs = torch.tensor(np.loadtxt(mantle_path), dtype=torch.float32, device=device)
    return c_bs, m_bs


def empirical_vp_rho_from_vs(vs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    vp = 0.9409 + 2.0947 * vs - 0.8206 * vs**2 + 0.2683 * vs**3 - 0.0251 * vs**4
    rho = 1.6612 * vp - 0.4721 * vp**2 + 0.0671 * vp**3 - 0.0043 * vp**4 + 0.000106 * vp**5
    return vp, rho


def unpack_parameter_vector(params: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "moho": params[..., 0],
        "crust_vs": params[..., 1:5],
        "mantle_vs": params[..., 5:10],
    }


def parameter_bounds(cfg: TrainConfig, device: Optional[torch.device | str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    env_cfg = cfg.env
    low = torch.tensor(
        [
            env_cfg.moho_min_km,
            env_cfg.v_upper_crust_min,
            env_cfg.v_middle_crust_min,
            env_cfg.v_lower_crust_min,
            env_cfg.v_deep_crust_min,
            env_cfg.v_litho_min,
            env_cfg.v_lab_min,
            env_cfg.v_lab_min,
            env_cfg.v_astheno_min,
            env_cfg.v_astheno_min,
        ],
        dtype=torch.float32,
        device=device,
    )
    high = torch.tensor(
        [
            env_cfg.moho_max_km,
            env_cfg.v_upper_crust_max,
            env_cfg.v_middle_crust_max,
            env_cfg.v_lower_crust_max,
            env_cfg.v_deep_crust_max,
            env_cfg.v_litho_max,
            env_cfg.v_lab_max,
            env_cfg.v_lab_max,
            env_cfg.v_astheno_max,
            env_cfg.v_astheno_max,
        ],
        dtype=torch.float32,
        device=device,
    )
    return low, high


def clamp_parameters(params: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    low, high = parameter_bounds(cfg, device=params.device)
    params_clamped = torch.clamp(params*high, min=low, max=high)
    return params_clamped/high


def _vp_from_vs(vs: torch.Tensor) -> torch.Tensor:
    return 0.9409 + 2.0947 * vs - 0.8206 * vs**2 + 0.2683 * vs**3 - 0.0251 * vs**4


def _valid_initial_model_mask(
    params: torch.Tensor,
    cfg: TrainConfig,
    crust_bs: Optional[torch.Tensor] = None,
    mantle_bs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Validation adapted for isotropic Vs models.

    Args:
        params: (batch, 10) denormalized physical parameters
    Returns:
        Boolean mask of valid models, shape (batch,)
    """
    if crust_bs is None or mantle_bs is None:
        crust_bs, mantle_bs = load_bspline_bases(
            cfg.data.crust_bspline_path,
            cfg.data.mantle_bspline_path,
            device=params.device,
        )
    profiles = parameters_to_profiles(params, crust_bs, mantle_bs, cfg.env.max_depth_km)
    depth = profiles["depth"]
    vs = profiles["vs"]
    batch = params.shape[0]

    valid = torch.ones(batch, dtype=torch.bool, device=params.device)

    # 1) Basic sanity
    valid &= torch.all(vs >= 0.5, dim=1)
    valid &= torch.all(depth >= 0.0, dim=1)
    valid &= (torch.abs(depth[:, -1] - cfg.env.max_depth_km) <= 1e-3)
    valid &= torch.all(torch.diff(depth, dim=1) > 0.0, dim=1)

    # 2) Basement floor
    valid &= vs[:, -1] >= 4.0

    # 3) VP/VS sanity
    vp = _vp_from_vs(vs)
    vpvs = vp / torch.clamp(vs, min=1e-6)
    valid &= torch.all((vpvs >= cfg.env.vpvs_min) & (vpvs <= cfg.env.vpvs_max), dim=1)

    moho = params[:, 0]
    for i in range(batch):
        if not valid[i]:
            continue
        depth_i = depth[i]
        vs_i = vs[i]
        moho_i = moho[i]
        moho_idx = int(torch.argmin(torch.abs(depth_i - moho_i)).item())

        # 4) Crust monotonicity
        crust_mask = depth_i < moho_i
        if int(crust_mask.sum().item()) > 1:
            vs_c = vs_i[crust_mask]
            if torch.any(torch.diff(vs_c) < 0.0):
                valid[i] = False
                continue

        # 5) Moho jump
        if 0 < moho_idx < depth_i.numel() - 1:
            jump_vs = float((vs_i[moho_idx + 1] - vs_i[moho_idx - 1]).item())
            if jump_vs < 0.0:
                valid[i] = False
                continue
            if jump_vs > 2.0:
                valid[i] = False
                continue
            if vs_i[moho_idx - 1] < 2.5:
                valid[i] = False
                continue

        # 6) Mantle smoothness
        mantle_mask = depth_i >= moho_i
        if int(mantle_mask.sum().item()) > 3:
            vs_m = vs_i[mantle_mask]
            d_m = depth_i[mantle_mask]
            first_grad = torch.gradient(vs_m, spacing=(d_m,))[0]
            second_grad = torch.gradient(first_grad, spacing=(d_m,))[0]

            max_slope = float(torch.max(torch.abs(first_grad)).item())
            max_slope_change = float(torch.max(torch.abs(torch.diff(first_grad))).item()) if first_grad.numel() > 1 else 0.0
            max_curvature = float(torch.max(torch.abs(second_grad)).item())

            if max_slope > cfg.env.slope_max:
                valid[i] = False
                continue
            if max_slope_change > cfg.env.slope_change_max:
                valid[i] = False
                continue
            if max_curvature > cfg.env.curv_max:
                valid[i] = False
                continue

            if int(mantle_mask.sum().item()) > 20 and torch.std(vs_m, unbiased=False) > cfg.env.mvs_std_max:
                valid[i] = False
                continue

    # 7) Global velocity ceiling
    valid &= torch.all(vs <= 5.5, dim=1)
    return valid


def sample_random_initial_parameters(
    cfg: TrainConfig,
    batch_size: int,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """
    Sample unconstrained initial parameters uniformly from the full bounds.

    Returns:
        (batch_size, 10) normalized parameters in [0, 1].
    """
    device = torch.device(device or infer_device(cfg))
    low, high = parameter_bounds(cfg, device=device)
    out = low.unsqueeze(0) + torch.rand(
        (batch_size, low.numel()), dtype=torch.float32, device=device
    ) * (high - low).unsqueeze(0)
    return out / high


def parameters_to_profiles(
    params: torch.Tensor,
    crust_bspline: torch.Tensor,
    mantle_bspline: torch.Tensor,
    max_depth: float,
) -> Dict[str, torch.Tensor]:
    parts = unpack_parameter_vector(params)
    batch = params.shape[0]
    device = params.device
    dtype = params.dtype
    cnum = crust_bspline.shape[0]
    mnum = mantle_bspline.shape[0]
    depth_1d = torch.linspace(0.0, float(max_depth), cnum + mnum, device=device, dtype=dtype)
    depth = depth_1d.unsqueeze(0).expand(batch, -1)
    moho = parts["moho"].unsqueeze(-1)

    crust_mask = depth < moho
    mantle_mask = ~crust_mask
    vs = torch.zeros_like(depth)

    crust_t = torch.clamp(depth / (moho + 1e-9), 0.0, 1.0) * float(cnum - 1)
    crust_lo = torch.floor(crust_t).long().clamp(0, cnum - 1)
    crust_hi = torch.clamp(crust_lo + 1, max=cnum - 1)
    crust_wh = crust_t - crust_lo.to(dtype)
    crust_basis = (1.0 - crust_wh).unsqueeze(-1) * crust_bspline[crust_lo] + crust_wh.unsqueeze(-1) * crust_bspline[crust_hi]
    crust_vs = (crust_basis * parts["crust_vs"].unsqueeze(1)).sum(dim=-1)

    mantle_span = float(max_depth) - moho + 1e-9
    mantle_t = torch.clamp((depth - moho) / mantle_span, 0.0, 1.0) * float(mnum - 1)
    mantle_lo = torch.floor(mantle_t).long().clamp(0, mnum - 1)
    mantle_hi = torch.clamp(mantle_lo + 1, max=mnum - 1)
    mantle_wh = mantle_t - mantle_lo.to(dtype)
    mantle_basis = (1.0 - mantle_wh).unsqueeze(-1) * mantle_bspline[mantle_lo] + mantle_wh.unsqueeze(-1) * mantle_bspline[mantle_hi]
    mantle_vs = (mantle_basis * parts["mantle_vs"].unsqueeze(1)).sum(dim=-1)

    vs = torch.where(crust_mask, crust_vs, mantle_vs)
    layer_type = mantle_mask.to(torch.int64)

    return {
        "depth": depth,
        "vs": vs,
        "layer_type": layer_type,
    }


def depths_to_thickness(depths: torch.Tensor) -> torch.Tensor:
    return depths[:, 1:] - depths[:, :-1]


def prepare_layered_model_from_params(
    params: torch.Tensor,
    crust_bspline: torch.Tensor,
    mantle_bspline: torch.Tensor,
    cfg: TrainConfig,
) -> Dict[str, torch.Tensor]:
    _, high = parameter_bounds(cfg, device=params.device)
    profiles = parameters_to_profiles(params*high, crust_bspline, mantle_bspline, max_depth=cfg.env.max_depth_km)
    thickness = depths_to_thickness(profiles["depth"])
    vs_mid = 0.5 * (profiles["vs"][:, 1:] + profiles["vs"][:, :-1])
    vp_mid, rho_mid = empirical_vp_rho_from_vs(vs_mid)
    profiles.update(
        {
            "thickness": thickness,
            "vs_mid": vs_mid,
            "vp_mid": vp_mid,
            "rho_mid": rho_mid,
        }
    )
    return profiles


def load_feature_array(path: Path | str, key: str = "data") -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        return data[key]
    return data


def infer_device(cfg: TrainConfig) -> torch.device:
    if cfg.env.device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(cfg.env.device)
    return torch.device("cpu")


class SurfBatchForward:
    """
    Thin wrapper around Surf/local surf96-style forward modelling.

    This implementation currently uses the local CPS API
    per station and keeps the interface needed by the RL package. It can be
    swapped later for a more vectorized ADsurf GPU path once the exact batch
    calling convention is finalized.
    """

    def __init__(self, cfg: Optional[TrainConfig] = None, device: Optional[torch.device | str] = None):
        self.cfg = cfg or get_default_config()
        ensure_external_paths(self.cfg)
        self.device = torch.device(device or infer_device(self.cfg))
        # self.ifunc_rayleigh = 2
        self.call_count = 0
        self.n_workers = max(1, int(self.cfg.env.forward_workers))
        self._pool: Optional[ProcessPoolExecutor] = None
        self._pool_workers = 0
        print(f"[SurfBatchForward] initialized with n_workers={self.n_workers}", flush=True)

    def _get_pool(self, worker_count: int) -> ProcessPoolExecutor:
        worker_count = max(1, int(worker_count))
        if self._pool is None or self._pool_workers != worker_count:
            self.close()
            self._pool = ProcessPoolExecutor(max_workers=worker_count)
            self._pool_workers = worker_count
            print(f"[SurfBatchForward] created worker pool workers={worker_count}", flush=True)
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            try:
                self._pool.shutdown(wait=True, cancel_futures=False)
            except TypeError:
                self._pool.shutdown(wait=True)
            self._pool = None
            self._pool_workers = 0
            print("[SurfBatchForward] worker pool closed", flush=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def forward(
        self,
        periods: torch.Tensor,
        thickness: torch.Tensor,
        vp: torch.Tensor,
        vs: torch.Tensor,
        rho: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        t0 = time.perf_counter()
        self.call_count += 1

        periods_np = periods.detach().cpu().numpy()
        thickness_np = thickness.detach().cpu().numpy()
        vp_np = vp.detach().cpu().numpy()
        vs_np = vs.detach().cpu().numpy()
        rho_np = rho.detach().cpu().numpy()
        t1 = time.perf_counter()
        batch_size = periods_np.shape[0]
        worker_count = min(self.n_workers, batch_size)
        print(
            f"[SurfParallelForward] call={self.call_count} cpu-convert={t1 - t0:.2f}s "
            f"batch={batch_size} workers={worker_count}",
            flush=True,
        )

        surf_root = str(self.cfg.data.surf_root)
        args_list = [
            (i, periods_np[i], thickness_np[i], vp_np[i], vs_np[i], rho_np[i], surf_root)
            for i in range(batch_size)
        ]
        rayleigh_phase_out = [None] * batch_size
        rayleigh_group_out = [None] * batch_size
        # work_one_station(args_list[0])
        pool = self._get_pool(worker_count)
        futures = {pool.submit(work_one_station, a): a[0] for a in args_list}
        completed = 0
        for fut in as_completed(futures):
            idx, phase, group = fut.result()
            rayleigh_phase_out[idx] = phase
            rayleigh_group_out[idx] = group
            completed += 1
            if completed == 1 or completed == batch_size or completed % max(1, batch_size // 4) == 0:
                ti = time.perf_counter()
                print(
                    f"[SurfParallelForward] call={self.call_count} completed {completed}/{batch_size} "
                    f"in {ti - t1:.2f}s",
                    flush=True,
                )

        t2 = time.perf_counter()
        out = {
            "rayleigh_phase": torch.tensor(np.stack(rayleigh_phase_out, axis=0), dtype=torch.float32, device=self.device),
            "rayleigh_group": torch.tensor(np.stack(rayleigh_group_out, axis=0), dtype=torch.float32, device=self.device),
        }
        t3 = time.perf_counter()
        print(f"[SurfParallelForward] call={self.call_count} pack={t3 - t2:.2f}s total={t3 - t0:.2f}s", flush=True)
        return out


def work_one_station_step(
    mode: str,
    tmpdir: str,
    mod_dir: str,
    mod_name: str,
    env: Dict[str, str],
    log: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run one CPS mode with direct binaries (no csh/tcsh dependency).

    This follows the same command sequence as vs_dispersion.csh:
      sprep96 -> sdisp96 -> sregn96|slegn96 -> sdpsrf96 -> sdpegn96
    """
    mode = mode.upper()
    if mode not in {"R", "L"}:
        raise ValueError(f"Unsupported mode: {mode}")

    mod_path = os.path.join(mod_dir, mod_name)
    dfile = os.path.join(mod_dir, "dfile")
    pfile = os.path.join(mod_dir, "pfile")

    mod_rel = os.path.relpath(mod_path, tmpdir)
    dfile_rel = os.path.relpath(dfile, tmpdir)
    pfile_rel = os.path.relpath(pfile, tmpdir)

    def _sub(cmd: list, name: str) -> None:
        result = subprocess.run(
            cmd, cwd=tmpdir, env=env,
            capture_output=True, text=True,
        )
        log.append(
            f"=== {name} rc={result.returncode} ===\n"
            f"STDOUT: {result.stdout[:300]}\n"
            f"STDERR: {result.stderr[:300]}"
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{name} failed (rc={result.returncode}):\n" + "\n".join(log)
            )
        # Some CPS binaries print errors to stderr but still return rc=0.
        err_lower = result.stderr.lower()
        if ("usage:" in err_lower) or ("does not exist" in err_lower):
            raise RuntimeError(
                f"{name} reported an error despite rc=0:\n" + "\n".join(log)
            )

    _sub(
        [
            "sprep96", "-M", mod_rel, "-d", dfile_rel, f"-{mode}",
            "-NMOD", "1", "-HS", "0", "-HR", "0", "-PARR", pfile_rel,
        ],
        "sprep96",
    )
    _sub(["sdisp96"], "sdisp96")
    _sub(["sregn96" if mode == "R" else "slegn96"], "sregn96" if mode == "R" else "slegn96")
    _sub(["sdpsrf96", f"-{mode}", "-XMIN", "0", "-XMAX", "4", "-YMIN", "2.0", "-YMAX", "5.0"], "sdpsrf96")
    _sub(["sdpegn96", f"-{mode}", "-U", "-C", "-ASC"], "sdpegn96")

    asc_name = f"S{mode}EGN.ASC"
    asc_path = os.path.join(tmpdir, asc_name)
    if not os.path.exists(asc_path):
        available = sorted(os.listdir(tmpdir))
        raise FileNotFoundError(
            f"{asc_name} not found. tmpdir contents: {available}\n"
            "CPS log:\n" + "\n".join(log)
        )

    arr = np.loadtxt(asc_path, skiprows=1, ndmin=2)
    mode0 = arr[arr[:, 0] == 0]
    if mode0.size == 0:
        raise RuntimeError(f"No mode=0 rows in {asc_name}")

    order = np.argsort(mode0[:, 2])
    mode0 = mode0[order]
    p_out = mode0[:, 2].astype(np.float64)
    c_out = mode0[:, 4].astype(np.float64)

    # Keep side-effect files equivalent to vs_dispersion.csh output.
    disp_dir = os.path.join(tmpdir, "dispfile")
    os.makedirs(disp_dir, exist_ok=True)
    np.savetxt(
        os.path.join(disp_dir, f"disp_{mode}_mod0.dat"),
        np.column_stack([mode0[:, 2], mode0[:, 4], mode0[:, 5]]),
        fmt="%.6f",
    )
    if mode == "R" and mode0.shape[1] >= 9:
        np.savetxt(
            os.path.join(disp_dir, "ZHratio_R_mod0.dat"),
            np.column_stack([mode0[:, 2], mode0[:, 3], mode0[:, 4], mode0[:, 5], mode0[:, 8]]),
            fmt="%.6f",
        )

    return p_out, c_out


def work_one_station(args: tuple) -> Tuple[int, np.ndarray, np.ndarray]:
    if len(args) >= 7:
        idx, periods, thickness, vp, vs, rho, cps_root = args[:7]
    else:
        idx, periods, thickness, vp, vs, rho = args
        cps_root = os.environ.get(
            "CPS_ISO_DISPERSION_ROOT",
            "/data/tianmingteng/Script/TMT_ModelDesign/pretraining/datamaker_cps/isotropy/cps_work",
        )

    periods = np.asarray(periods, dtype=np.float64).reshape(-1)
    thickness = np.asarray(thickness, dtype=np.float64).reshape(-1)
    vp = np.asarray(vp, dtype=np.float64).reshape(-1)
    vs = np.asarray(vs, dtype=np.float64).reshape(-1)
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)

    if periods.size == 0 or thickness.size == 0 or not (thickness.size == vp.size == vs.size == rho.size):
        z = np.zeros_like(periods, dtype=np.float32)
        return idx, z, z

    cps_bin = os.environ.get("CPS_PROGRAM_BIN", "/share/home/tianmingteng/CPS_PROGRAM/bin")
    env = os.environ.copy()
    env["PATH"] = f"{cps_bin}:/usr/bin:/bin:/usr/local/bin:{env.get('PATH', '')}"
    Path(cps_root).mkdir(parents=True, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="cps_", dir=str(cps_root))
    mod_dir = os.path.join(tmpdir, "modelfile")
    os.makedirs(mod_dir, exist_ok=True)

    try:
        mod_path = os.path.join(mod_dir, "Litho_start.mod")
        h = thickness.copy()
        h[h <= 0.0] = 2.0
        lines = [
            "MODEL.01",
            "Created from isotropic SAC work_one_station",
            "ISOTROPIC",
            "KGS",
            "SPHERICAL EARTH",
            "1-D",
            "CONSTANT VELOCITY",
            "LINE08", "LINE09", "LINE10", "LINE11",
            " H(KM) VP(KM/S) VS(KM/S) RHO(GM/CC)   QP   QS  ETAP  ETAS  FREFP  FREFS",
        ]
        for i in range(h.size):
            lines.append(f"{h[i]:.6f} {vp[i]:.6f} {vs[i]:.6f} {rho[i]:.6f} 900 400 0 0 1 1")
        lines.append(f"9999 {vp[-1]:.6f} {vs[-1]:.6f} {rho[-1]:.6f} 900 400 0 0 1 1")
        with open(mod_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        dfile = os.path.join(mod_dir, "dfile")
        pfile = os.path.join(mod_dir, "pfile")
        with open(dfile, "w", encoding="utf-8") as fh:
            fh.write("100 0.1 4096 0 0\n")
        with open(pfile, "w", encoding="utf-8") as fh:
            fh.write(f"{len(periods)}\n")
            fh.write("\n".join(f"{p:.4f}" for p in periods) + "\n")

        mod_rel = os.path.relpath(mod_path, tmpdir)
        dfile_rel = os.path.relpath(dfile, tmpdir)
        pfile_rel = os.path.relpath(pfile, tmpdir)

        def _run(cmd: list[str], name: str) -> None:
            result = subprocess.run(cmd, cwd=tmpdir, env=env, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"{name} failed (rc={result.returncode}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
            err_lower = result.stderr.lower()
            if ("usage:" in err_lower) or ("does not exist" in err_lower):
                raise RuntimeError(f"{name} reported an error despite rc=0:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

        _run(["sprep96", "-M", mod_rel, "-d", dfile_rel, "-R", "-NMOD", "1", "-HS", "0", "-HR", "0", "-PARR", pfile_rel], "sprep96")
        _run(["sdisp96"], "sdisp96")
        _run(["sregn96"], "sregn96")
        _run(["sdpsrf96", "-R", "-XMIN", "0", "-XMAX", "4", "-YMIN", "2.0", "-YMAX", "5.0"], "sdpsrf96")
        _run(["sdpegn96", "-R", "-U", "-C", "-ASC"], "sdpegn96")

        asc_path = os.path.join(tmpdir, "SRFGN.ASC")
        if not os.path.exists(asc_path):
            asc_path = os.path.join(tmpdir, "SREGN.ASC")
        rows = []
        with open(asc_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 6:
                    continue
                try:
                    rows.append([float(x) for x in parts])
                except ValueError:
                    continue
        if not rows:
            raise RuntimeError(f"No numeric rows found in {os.path.basename(asc_path)}")
        arr = np.asarray(rows, dtype=np.float64)
        mode0 = arr[arr[:, 0] == 0]
        if mode0.size == 0:
            raise RuntimeError(f"No mode=0 rows in {os.path.basename(asc_path)}")
        order = np.argsort(mode0[:, 2])
        mode0 = mode0[order]
        p_out = mode0[:, 2].astype(np.float64)
        ray_phase_out = mode0[:, 4].astype(np.float64)
        ray_group_out = mode0[:, 5].astype(np.float64) if mode0.shape[1] > 5 else ray_phase_out.copy()
        phase = np.interp(periods, p_out, ray_phase_out).astype(np.float32)
        group = np.interp(periods, p_out, ray_group_out).astype(np.float32)
        return idx, phase, group
    except Exception:
        z = np.zeros_like(periods, dtype=np.float32)
        return idx, z, z
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def masked_l2_misfit(pred: torch.Tensor, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        mask = obs > 0
    valid = mask.to(pred.dtype)
    diff2 = ((pred - obs) ** 2) * valid
    denom = torch.clamp(valid.sum(dim=-1), min=1.0)
    return torch.sqrt(diff2.sum(dim=-1) / denom)
