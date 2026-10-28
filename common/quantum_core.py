from __future__ import annotations

"""
Public-safe quantum utilities for the Flux quantum-evidence experiments.

The public repository operates only on already-exported local subproblems.
No routing, candidate generation, or production Flux solver code is required.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_SUBPROBLEM_ROOT = REPO_ROOT / "instances" / "local_subproblems"


@dataclass(frozen=True)
class QSchedule:
    smin: float
    ramp_time: float
    hold_time: float
    driver_scale: float = 1.0

    @property
    def duration(self) -> float:
        return 2.0 * float(self.ramp_time) + float(self.hold_time)


def _scalar(z: np.lib.npyio.NpzFile, *names: str, default: Any = None) -> Any:
    for name in names:
        if name in z.files:
            value = np.asarray(z[name])
            if value.shape == ():
                return value.item()
            if value.size == 1:
                return value.reshape(()).item()
            return value
    return default


def _array(
    z: np.lib.npyio.NpzFile,
    *names: str,
    dtype: Any | None = None,
    required: bool = True,
) -> np.ndarray | None:
    for name in names:
        if name in z.files:
            out = np.asarray(z[name])
            if dtype is not None:
                out = out.astype(dtype, copy=False)
            return out
    if required:
        raise KeyError(f"Missing required array; tried {names}")
    return None


def list_local_case_files(set_name: str) -> list[Path]:
    root = LOCAL_SUBPROBLEM_ROOT / set_name
    if not root.exists():
        raise FileNotFoundError(
            f"Local-subproblem set not found: {root}"
        )
    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No .npz local subproblems found in {root}"
        )
    return files


def load_local_case(path: str | Path) -> dict[str, Any]:
    """Load an exported quantum local subproblem.

    Supported scalar names are intentionally permissive so the loader works with
    the exporter variants used during repository preparation.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        energies = _array(
            z, "energies", "cube_energies", dtype=np.float64
        )
        feasible = _array(
            z, "feasible", "cube_feasible", dtype=bool
        )
        A = _array(
            z, "A", "conflict_A", "conflict_matrix", dtype=np.int8
        )

        k = int(
            _scalar(
                z,
                "k",
                default=int(round(math.log2(len(energies)))),
            )
        )
        if len(energies) != (1 << k):
            raise ValueError(
                f"{path}: energies has length {len(energies)}, expected 2^k={1<<k}"
            )
        if len(feasible) != len(energies):
            raise ValueError(
                f"{path}: feasible/energies length mismatch"
            )
        if A.shape != (k, k):
            raise ValueError(
                f"{path}: A has shape {A.shape}, expected {(k, k)}"
            )

        start_idx = int(_scalar(z, "start_idx"))
        target_idx = int(_scalar(z, "target_idx"))
        if not (0 <= start_idx < len(energies)):
            raise ValueError(f"{path}: invalid start_idx={start_idx}")
        if not (0 <= target_idx < len(energies)):
            raise ValueError(f"{path}: invalid target_idx={target_idx}")

        # If the exporter saved the exact normalized problem used by the
        # original experiment, prefer it. Otherwise the original energy-shift
        # convention is reconstructed from the raw energies.
        normalized = _array(
            z,
            "normalized_problem",
            "problem_normalized",
            "problem",
            dtype=np.float64,
            required=False,
        )
        if normalized is not None and len(normalized) != len(energies):
            raise ValueError(
                f"{path}: normalized problem length mismatch"
            )

        cube = {
            "k": k,
            "energies": energies,
            "feasible": feasible,
            "start_idx": start_idx,
            "target_idx": target_idx,
        }
        if normalized is not None:
            cube["normalized_problem"] = normalized

        gain_default = float(
            energies[start_idx] - energies[target_idx]
        )
        gain = float(_scalar(z, "gain", default=gain_default))
        barrier = float(_scalar(z, "barrier", default=math.nan))
        ratio = float(
            _scalar(
                z,
                "barrier_over_gain",
                default=(
                    barrier / gain
                    if math.isfinite(barrier) and abs(gain) > 1e-15
                    else math.nan
                ),
            )
        )

        schedule = QSchedule(
            smin=float(
                _scalar(
                    z,
                    "smin",
                    "schedule_smin",
                    default=0.5,
                )
            ),
            ramp_time=float(
                _scalar(
                    z,
                    "ramp_time",
                    "schedule_ramp_time",
                    default=1.0,
                )
            ),
            hold_time=float(
                _scalar(
                    z,
                    "hold_time",
                    "schedule_hold_time",
                    default=0.0,
                )
            ),
            driver_scale=float(
                _scalar(
                    z,
                    "driver_scale",
                    "schedule_driver_scale",
                    default=1.0,
                )
            ),
        )

        return {
            "path": path,
            "case_id": int(
                _scalar(
                    z,
                    "case_id",
                    default=int(
                        "".join(ch for ch in path.stem if ch.isdigit())
                        or 0
                    ),
                )
            ),
            "set_name": str(
                _scalar(
                    z,
                    "set_name",
                    "source_set",
                    default=path.parent.name,
                )
            ),
            "instance_index": int(
                _scalar(z, "instance_index", "source_index", default=-1)
            ),
            "quality_fraction": float(
                _scalar(z, "quality_fraction", default=math.nan)
            ),
            "terms": {"A": A},
            "cube": cube,
            "geom": {
                "gain": gain,
                "barrier": barrier,
                "barrier_over_gain": ratio,
            },
            "schedule": schedule,
        }


def load_local_cases(set_name: str) -> list[dict[str, Any]]:
    cases = [load_local_case(p) for p in list_local_case_files(set_name)]
    cases.sort(key=lambda c: int(c["case_id"]))
    return cases


def normalized_problem(cube: dict[str, Any]) -> tuple[np.ndarray, float]:
    """Return the problem energies used for quantum evolution.

    An additive energy shift is dynamically irrelevant. The original mechanism
    experiments used the incumbent/start energy as zero. If an exporter stored
    an explicit normalized_problem array, it is used verbatim.
    """
    if "normalized_problem" in cube:
        p = np.asarray(cube["normalized_problem"], dtype=np.float64)
        return p, 1.0
    E = np.asarray(cube["energies"], dtype=np.float64)
    sidx = int(cube["start_idx"])
    return E - float(E[sidx]), 1.0


def transverse_driver_sparse(k: int) -> sparse.csr_matrix:
    """Return H_D = -sum_q X_q in computational-basis ordering."""
    dim = 1 << int(k)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for state in range(dim):
        for q in range(k):
            rows.append(state)
            cols.append(state ^ (1 << q))
            data.append(-1.0)
    return sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(dim, dim),
        dtype=np.float64,
    ).tocsr()


def apply_driver_rotation(
    psi: np.ndarray,
    theta: float,
    k: int,
) -> np.ndarray:
    """Apply exp(+i theta sum X_q), corresponding to H_D=-sum X_q."""
    out = np.asarray(psi, dtype=np.complex128).copy()
    c = math.cos(float(theta))
    s = 1j * math.sin(float(theta))
    for q in range(k):
        half = 1 << q
        block = half << 1
        view = out.reshape(-1, block)
        a = view[:, :half].copy()
        b = view[:, half:].copy()
        view[:, :half] = c * a + s * b
        view[:, half:] = s * a + c * b
    return out


def _strang_step(
    psi: np.ndarray,
    problem: np.ndarray,
    *,
    k: int,
    s: float,
    driver_scale: float,
    dt: float,
) -> np.ndarray:
    theta = (
        0.5
        * float(dt)
        * float(driver_scale)
        * (1.0 - float(s))
    )
    psi = apply_driver_rotation(psi, theta, k)
    psi *= np.exp(
        -1j * float(dt) * float(s) * problem
    )
    psi = apply_driver_rotation(psi, theta, k)
    return psi


def _schedule_steps(
    schedule: QSchedule,
    *,
    ramp_steps: int,
    hold_steps_per_unit: int,
) -> list[tuple[float, float]]:
    """Return (s, dt) midpoint steps for down-ramp, hold, and up-ramp."""
    out: list[tuple[float, float]] = []

    nr = max(1, int(ramp_steps))
    if schedule.ramp_time > 0:
        dt = float(schedule.ramp_time) / nr
        for step in range(nr):
            u = (step + 0.5) / nr
            s = 1.0 - u * (1.0 - schedule.smin)
            out.append((float(s), dt))

    if schedule.hold_time > 0:
        nh = max(
            1,
            int(math.ceil(
                float(hold_steps_per_unit)
                * float(schedule.hold_time)
            )),
        )
        dt = float(schedule.hold_time) / nh
        out.extend(
            (float(schedule.smin), dt)
            for _ in range(nh)
        )

    if schedule.ramp_time > 0:
        dt = float(schedule.ramp_time) / nr
        for step in range(nr):
            u = (step + 0.5) / nr
            s = schedule.smin + u * (1.0 - schedule.smin)
            out.append((float(s), dt))

    return out


def _probability_summary(
    psi: np.ndarray,
    cube: dict[str, Any],
    target_energy_raw: float,
) -> dict[str, float]:
    p = np.abs(psi) ** 2
    total = float(np.sum(p))
    if total <= 0:
        raise FloatingPointError("Quantum state has zero norm")
    p = p / total

    E = np.asarray(cube["energies"], dtype=np.float64)
    feasible = np.asarray(cube["feasible"], dtype=bool)
    sidx = int(cube["start_idx"])
    tidx = int(cube["target_idx"])
    quality = feasible & (
        E <= float(target_energy_raw) + 1e-12
    )

    return {
        "p_target": float(p[tidx]),
        "p_target_quality_or_better": float(np.sum(p[quality])),
        "p_feasible": float(np.sum(p[feasible])),
        "p_start": float(p[sidx]),
    }


def coherent_quantum(
    cube: dict[str, Any],
    *,
    schedule: QSchedule,
    target_energy_raw: float,
    ramp_steps: int = 80,
    hold_steps_per_unit: int = 16,
) -> dict[str, Any]:
    problem, _ = normalized_problem(cube)
    k = int(cube["k"])
    sidx = int(cube["start_idx"])
    psi = np.zeros(len(problem), dtype=np.complex128)
    psi[sidx] = 1.0

    steps = _schedule_steps(
        schedule,
        ramp_steps=ramp_steps,
        hold_steps_per_unit=hold_steps_per_unit,
    )
    for s, dt in steps:
        psi = _strang_step(
            psi,
            problem,
            k=k,
            s=s,
            driver_scale=schedule.driver_scale,
            dt=dt,
        )

    out = _probability_summary(
        psi, cube, target_energy_raw
    )
    out.update(
        {
            "smin": float(schedule.smin),
            "ramp_time": float(schedule.ramp_time),
            "hold_time": float(schedule.hold_time),
            "driver_scale": float(schedule.driver_scale),
            "schedule_duration": float(schedule.duration),
            "steps": len(steps),
        }
    )
    return out


def _apply_random_z_jumps(
    psi: np.ndarray,
    *,
    k: int,
    gamma: float,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Trajectory unraveling of computational-basis dephasing.

    Independent Pauli-Z jumps reproduce exponential damping of computational-
    basis coherences. The probability convention is chosen so that a one-qubit
    off-diagonal decays approximately as exp(-gamma * dt).
    """
    if gamma <= 0 or dt <= 0:
        return psi

    p_jump = 0.5 * (
        1.0 - math.exp(-float(gamma) * float(dt))
    )
    if p_jump <= 0:
        return psi

    indices = np.arange(len(psi), dtype=np.int64)
    out = psi
    for q in range(k):
        if rng.random() < p_jump:
            signs = 1.0 - 2.0 * (
                (indices >> q) & 1
            )
            out = out * signs
    return out


def dephased_quantum_trajectories(
    cube: dict[str, Any],
    *,
    schedule: QSchedule,
    target_energy_raw: float,
    gamma: float,
    trajectories: int,
    ramp_steps: int = 80,
    hold_steps_per_unit: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    if trajectories <= 0:
        raise ValueError("trajectories must be positive")

    problem, _ = normalized_problem(cube)
    k = int(cube["k"])
    sidx = int(cube["start_idx"])
    E = np.asarray(cube["energies"], dtype=np.float64)
    feasible = np.asarray(cube["feasible"], dtype=bool)
    tidx = int(cube["target_idx"])
    quality = feasible & (
        E <= float(target_energy_raw) + 1e-12
    )

    steps = _schedule_steps(
        schedule,
        ramp_steps=ramp_steps,
        hold_steps_per_unit=hold_steps_per_unit,
    )
    rng = np.random.default_rng(seed)

    p_target = np.empty(trajectories, dtype=np.float64)
    p_quality = np.empty(trajectories, dtype=np.float64)
    p_feas = np.empty(trajectories, dtype=np.float64)
    p_start = np.empty(trajectories, dtype=np.float64)

    for tr in range(trajectories):
        psi = np.zeros(len(problem), dtype=np.complex128)
        psi[sidx] = 1.0
        for s, dt in steps:
            psi = _strang_step(
                psi,
                problem,
                k=k,
                s=s,
                driver_scale=schedule.driver_scale,
                dt=dt,
            )
            psi = _apply_random_z_jumps(
                psi,
                k=k,
                gamma=float(gamma),
                dt=float(dt),
                rng=rng,
            )

        p = np.abs(psi) ** 2
        p /= np.sum(p)
        p_target[tr] = p[tidx]
        p_quality[tr] = np.sum(p[quality])
        p_feas[tr] = np.sum(p[feasible])
        p_start[tr] = p[sidx]

    se = (
        float(np.std(p_quality, ddof=1) / math.sqrt(trajectories))
        if trajectories > 1
        else 0.0
    )
    return {
        "p_target": float(np.mean(p_target)),
        "p_target_quality_or_better": float(np.mean(p_quality)),
        "p_target_quality_se": se,
        "p_feasible": float(np.mean(p_feas)),
        "p_start": float(np.mean(p_start)),
        "smin": float(schedule.smin),
        "ramp_time": float(schedule.ramp_time),
        "hold_time": float(schedule.hold_time),
        "driver_scale": float(schedule.driver_scale),
        "schedule_duration": float(schedule.duration),
        "steps": len(steps),
        "trajectories": int(trajectories),
        "gamma": float(gamma),
    }
