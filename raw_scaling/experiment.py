from __future__ import annotations

"""
EulerQ — Raw Flux Scaling with Warm-Start SimCIM and PT Diagnostics
==================================================================

Purpose
-------
This script evaluates the scaling behavior of the Flux candidate-selection
problem on complete raw Flux instances and compares several optimization and
diagnostic components used in the study.

The experiment focuses on:

* validating the implicit Ising representation used for larger Flux instances;
* evaluating warm-start SimCIM on complete raw Flux subproblems;
* measuring the timing components of warm-start SCIP;
* studying parallel-tempering behavior on representative Flux instances.

Experiments
-----------
A. Dense-versus-implicit representation validation on small Flux instances
   - verify conflict degrees;
   - compare implicit and dense Ising fields;
   - verify QUBO/Ising energy consistency;
   - compare implicit spectral normalization with the dense reference.

B. Warm-start scalable SimCIM on complete raw Flux subproblems
   - initialize from a greedy incumbent;
   - tune solver parameters on designated training instances;
   - freeze the selected configuration before evaluation;
   - retain the best feasible state encountered during the trajectory;
   - preserve the incumbent whenever no better feasible solution is found.

C. Warm-start SCIP timing analysis
   - model construction time;
   - warm-start insertion time;
   - optimization time.

D. Parallel-tempering diagnostics on representative Flux subproblems
   - spin acceptance across temperatures;
   - replica-swap acceptance;
   - cold-to-hot-to-cold replica round trips;
   - time to the first same-cardinality improvement;
   - behavior under different replica counts.

All experiments operate directly on complete raw Flux candidate-selection
instances. No oracle-selected hard neighborhoods are used in the raw scaling
benchmark.
"""

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pyscipopt import Model, quicksum

HERE = Path(__file__).resolve()
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common.problem_data import ProblemData, load_problem_set  # noqa: E402


# -----------------------------------------------------------------------------
# I/O / small helpers
# -----------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")



def lex_better(a1: int, d1: float, a2: int, d2: float) -> bool:
    return a1 > a2 or (a1 == a2 and d1 < d2 - 1e-10)


def refinement_fraction(
    start_a: int,
    start_d: float,
    result_a: int,
    result_d: float,
    ref_a: int,
    ref_d: float,
) -> float:
    """Fraction of greedy->reference refinement recovered.

    Same-cardinality distance refinement is the main regime.  If the reference
    has higher cardinality, return a cardinality-aware quantity instead of
    pretending distance gaps are comparable.
    """
    if ref_a == start_a:
        if result_a < start_a:
            return -float(start_a - result_a)
        if result_a > start_a:
            return 1.0 + float(result_a - start_a)
        denom = start_d - ref_d
        if denom <= 1e-12:
            return 1.0 if result_d <= ref_d + 1e-9 else 0.0
        return float((start_d - result_d) / denom)
    # Higher-cardinality reference: cardinality dominates.
    return float(result_a - start_a) / max(1, ref_a - start_a)


def tuning_score(
    start_a: int,
    start_d: float,
    result_a: int,
    result_d: float,
    ref_a: int,
    ref_d: float,
) -> float:
    """Higher is better; cardinality dominates, then normalized refinement."""
    if result_a != start_a:
        return 1000.0 * (result_a - start_a)
    return refinement_fraction(start_a, start_d, result_a, result_d, ref_a, ref_d)



# -----------------------------------------------------------------------------
# Public candidate-selection helpers
# -----------------------------------------------------------------------------


def eval_bits(bits: np.ndarray, d: ProblemData) -> tuple[int, float, float]:
    b = np.asarray(bits, dtype=np.int8)
    assigned = int(b @ d.count.astype(np.int32))
    distance = float(b @ d.distance)
    business = float(d.M * assigned - distance)
    return assigned, distance, business


def greedy_pair_first(d: ProblemData) -> np.ndarray:
    """Deterministic pair-first feasible incumbent used by the original study."""
    out = np.zeros(d.num_candidates, dtype=np.int8)
    used_riders: set[int] = set()
    used_orders: set[int] = set()
    order = sorted(
        range(d.num_candidates),
        key=lambda i: (-int(d.count[i]), float(d.distance[i]), int(i)),
    )
    for i in order:
        r = int(d.rider[i])
        a = int(d.o1[i])
        b = int(d.o2[i])
        if r in used_riders or a in used_orders or (b >= 0 and b in used_orders):
            continue
        out[i] = 1
        used_riders.add(r)
        used_orders.add(a)
        if b >= 0:
            used_orders.add(b)
    return out


@dataclass
class Occ:
    rider: np.ndarray
    order: np.ndarray
    rider_order: np.ndarray
    order_pair: np.ndarray


def start_occupancy(
    bits: np.ndarray, d: ProblemData, nr: int, no: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ro = np.zeros(nr, dtype=np.int16)
    oo = np.zeros(no, dtype=np.int16)
    r_o = np.zeros((nr, no), dtype=np.int16)
    op = np.zeros((no, no), dtype=np.int16)
    for i in np.flatnonzero(bits):
        r, a, b = int(d.rider[i]), int(d.o1[i]), int(d.o2[i])
        ro[r] += 1
        oo[a] += 1
        r_o[r, a] += 1
        if b >= 0:
            oo[b] += 1
            r_o[r, b] += 1
            op[a, b] += 1
            op[b, a] += 1
    return ro, oo, r_o, op


def replicate_occ(
    base: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    shape: tuple[int, ...],
) -> Occ:
    ro, oo, r_o, op = base

    def rep(arr: np.ndarray) -> np.ndarray:
        return np.broadcast_to(arr, shape + arr.shape).copy()

    return Occ(rep(ro), rep(oo), rep(r_o), rep(op))


def neighbor_count(
    states_i: np.ndarray, occ: Occ, d: ProblemData, i: int
) -> np.ndarray:
    """Exact number of selected unique conflict neighbours of candidate i."""
    r, a, b = int(d.rider[i]), int(d.o1[i]), int(d.o2[i])
    if b < 0:
        union = (
            occ.rider[..., r]
            + occ.order[..., a]
            - occ.rider_order[..., r, a]
        )
    else:
        union = (
            occ.rider[..., r]
            + occ.order[..., a]
            + occ.order[..., b]
            - occ.rider_order[..., r, a]
            - occ.rider_order[..., r, b]
            - occ.order_pair[..., a, b]
            + states_i
        )
    return union - states_i


def apply_flip_update(
    occ: Occ, d: ProblemData, i: int, dx: np.ndarray
) -> None:
    r, a, b = int(d.rider[i]), int(d.o1[i]), int(d.o2[i])
    occ.rider[..., r] += dx
    occ.order[..., a] += dx
    occ.rider_order[..., r, a] += dx
    if b >= 0:
        occ.order[..., b] += dx
        occ.rider_order[..., r, b] += dx
        occ.order_pair[..., a, b] += dx
        occ.order_pair[..., b, a] += dx


def feasible_mask(occ: Occ) -> np.ndarray:
    return (
        (np.max(occ.rider, axis=-1) <= 1)
        & (np.max(occ.order, axis=-1) <= 1)
    )


# -----------------------------------------------------------------------------
# Warm-start SCIP with timing decomposition
# -----------------------------------------------------------------------------


def warm_scip(
    d: ProblemData,
    greedy: np.ndarray,
    limit_s: float,
    seed: int,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    model = Model("raw_flux_public")
    model.hideOutput()
    try:
        model.setIntParam(
            "randomization/randomseedshift",
            int(seed) % 2147483647,
        )
    except Exception:
        pass

    x = [
        model.addVar(vtype="B", name=f"x_{i}")
        for i in range(d.num_candidates)
    ]
    for ids in d.by_rider:
        if len(ids) > 0:
            model.addCons(quicksum(x[int(i)] for i in ids) <= 1)
    for ids in d.by_order:
        if len(ids) > 0:
            model.addCons(quicksum(x[int(i)] for i in ids) <= 1)
    model.setObjective(
        quicksum(
            float(d.raw_reward[i]) * x[i]
            for i in range(d.num_candidates)
        ),
        "maximize",
    )
    build_ms = 1000.0 * (time.perf_counter() - t0)

    tw = time.perf_counter()
    warm_ok = False
    try:
        sol = model.createSol()
        for i, var in enumerate(x):
            model.setSolVal(sol, var, float(greedy[i]))
        warm_ok = bool(model.addSol(sol, free=True))
    except Exception:
        warm_ok = False
    warm_ms = 1000.0 * (time.perf_counter() - tw)

    model.setRealParam("limits/time", float(limit_s))
    ts = time.perf_counter()
    model.optimize()
    optimize_ms = 1000.0 * (time.perf_counter() - ts)

    bits = greedy.copy()
    if model.getNSols() > 0:
        sol = model.getBestSol()
        cand = np.asarray(
            [int(model.getSolVal(sol, var) > 0.5) for var in x],
            dtype=np.int8,
        )
        aa, dd, _ = eval_bits(cand, d)
        ga, gd, _ = eval_bits(bits, d)
        if lex_better(aa, dd, ga, gd):
            bits = cand

    a, dist, score = eval_bits(bits, d)
    try:
        gap = float(model.getGap())
    except Exception:
        gap = math.nan
    try:
        nodes = int(model.getNNodes())
    except Exception:
        nodes = -1
    try:
        solving_time_s = float(model.getSolvingTime())
    except Exception:
        solving_time_s = math.nan

    status = str(model.getStatus())
    return {
        "bits": bits,
        "assigned": a,
        "distance": dist,
        "business": score,
        "status": status,
        "optimal": status.lower() == "optimal",
        "gap": gap,
        "nodes": nodes,
        "warm_start_accepted": warm_ok,
        "build_ms": build_ms,
        "warm_start_ms": warm_ms,
        "optimize_ms": optimize_ms,
        "scip_reported_solving_ms": (
            1000.0 * solving_time_s
            if math.isfinite(solving_time_s)
            else math.nan
        ),
        "wall_ms": build_ms + warm_ms + optimize_ms,
    }


# -----------------------------------------------------------------------------
# Scalable warm-start soft-spin SimCIM
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SoftConfig:
    coupling: float
    noise: float
    pump_start: float
    pump_end: float
    warm_alpha: float
    warm_std: float
    momentum: float = 0.9
    saturation: float = 1.0


# Curated reverse/warm-start grid.  It deliberately spans weak/strong memory of
# the incumbent and weak/strong collective forcing.  Selection is per SIZE on
# training instances, never per held-out test instance.
SOFT_GRID = [
    SoftConfig(0.60, 0.02, -0.12, 0.00, 0.15, 0.03),
    SoftConfig(0.80, 0.03, -0.10, 0.02, 0.20, 0.04),
    SoftConfig(1.00, 0.05, -0.10, 0.05, 0.20, 0.05),
    SoftConfig(1.20, 0.05, -0.08, 0.06, 0.20, 0.05),
    SoftConfig(0.80, 0.04, -0.06, 0.05, 0.35, 0.04),
    SoftConfig(1.00, 0.05, -0.05, 0.08, 0.35, 0.05),
    SoftConfig(1.20, 0.05, -0.05, 0.08, 0.35, 0.05),
    SoftConfig(1.40, 0.08, -0.04, 0.10, 0.35, 0.08),
    SoftConfig(0.80, 0.03,  0.00, 0.08, 0.50, 0.04),
    SoftConfig(1.00, 0.05,  0.00, 0.10, 0.50, 0.05),
    SoftConfig(1.20, 0.05,  0.00, 0.10, 0.50, 0.05),
    SoftConfig(1.40, 0.08,  0.00, 0.12, 0.50, 0.08),
    SoftConfig(0.80, 0.05, -0.15, 0.00, 0.12, 0.08),
    SoftConfig(1.00, 0.08, -0.15, 0.05, 0.15, 0.08),
    SoftConfig(1.20, 0.08, -0.15, 0.05, 0.15, 0.08),
    SoftConfig(0.60, 0.02, -0.05, 0.02, 0.70, 0.03),
]


class WarmGroupSoftSpin:
    """Implicit exact Flux Ising field with warm-start refinement dynamics.

    Normalized QUBO:
        E(z) = -sum_i reward_i z_i + sum_(unique conflict edges) z_i z_j

    Ising convention used by the original EulerQ SimCIM implementation:
        E(s) = -0.5 s^T J s - h^T s + const
        J_ij = -1/4  on a unique conflict
        h_i  = reward_i/2 - degree_i/4

    Jx is computed exactly by resource-group inclusion/exclusion.  No dense Q,
    J or conflict matrix is constructed for large instances.
    """

    def __init__(
        self,
        d: ProblemData,
        nr: int,
        no: int,
        device: str = "cpu",
        *,
        normalization: str = "spectral_power",
        power_iters: int = 24,
        power_seed: int = 12345,
    ):
        self.d = d
        self.nr = int(nr)
        self.no = int(no)
        self.device = torch.device(device)
        self.m = len(d.rider)
        self.normalization = normalization

        self.r = torch.as_tensor(d.rider.astype(np.int64), device=self.device)
        self.a = torch.as_tensor(d.o1.astype(np.int64), device=self.device)
        self.b = torch.as_tensor(np.maximum(d.o2, 0).astype(np.int64), device=self.device)
        self.has_b = torch.as_tensor((d.o2 >= 0), device=self.device)
        self.has_any_b = bool(np.any(d.o2 >= 0))
        self.ro1 = self.r * self.no + self.a
        self.ro2 = self.r * self.no + self.b
        self.pair = self.a * self.no + self.b

        # Exact unique-conflict degree from inclusion/exclusion.
        rc = np.bincount(d.rider, minlength=nr)
        oc = np.bincount(d.o1, minlength=no).astype(np.int64)
        mask = d.o2 >= 0
        oc += np.bincount(d.o2[mask], minlength=no)
        ro = np.zeros((nr, no), dtype=np.int64)
        np.add.at(ro, (d.rider, d.o1), 1)
        np.add.at(ro, (d.rider[mask], d.o2[mask]), 1)
        op = np.zeros((no, no), dtype=np.int64)
        np.add.at(op, (d.o1[mask], d.o2[mask]), 1)
        np.add.at(op, (d.o2[mask], d.o1[mask]), 1)
        deg = rc[d.rider] + oc[d.o1] - ro[d.rider, d.o1] - 1
        if np.any(mask):
            ii = np.flatnonzero(mask)
            deg[ii] = (
                rc[d.rider[ii]]
                + oc[d.o1[ii]]
                + oc[d.o2[ii]]
                - ro[d.rider[ii], d.o1[ii]]
                - ro[d.rider[ii], d.o2[ii]]
                - op[d.o1[ii], d.o2[ii]]
            )
        self.degree_np = deg.astype(np.float64)
        self.h_raw_np = d.reward.astype(np.float64) / 2.0 - self.degree_np / 4.0
        self.h_raw = torch.as_tensor(self.h_raw_np.astype(np.float32), device=self.device)
        self.count_t = torch.as_tensor(d.count.astype(np.float32), device=self.device)
        self.distance_t = torch.as_tensor(d.distance.astype(np.float32), device=self.device)

        self.spectral_j_estimate = math.nan
        self.row_sum_j_bound = float(np.max(self.degree_np) / 4.0) if self.m else 0.0
        self.h_scale = float(np.max(np.abs(self.h_raw_np))) if self.m else 0.0
        if normalization == "none":
            self.scale = 1.0
        elif normalization == "row_sum":
            self.scale = max(self.row_sum_j_bound, self.h_scale, 1e-12)
        elif normalization == "spectral_power":
            self.spectral_j_estimate = self._estimate_spectral_radius(power_iters, power_seed)
            self.scale = max(self.spectral_j_estimate, self.h_scale, 1e-12)
        else:
            raise ValueError(f"unknown normalization {normalization!r}")
        self.h = self.h_raw / float(self.scale)

    @staticmethod
    def _sum_groups(x: torch.Tensor, ids: torch.Tensor, ng: int) -> torch.Tensor:
        out = torch.zeros((x.shape[0], ng), dtype=x.dtype, device=x.device)
        out.scatter_add_(1, ids.unsqueeze(0).expand(x.shape[0], -1), x)
        return out

    def _conflict_sum(self, x: torch.Tensor) -> torch.Tensor:
        """Sum x_j over unique conflict neighbors j of every candidate i."""
        replicas = x.shape[0]
        sr = self._sum_groups(x, self.r, self.nr)
        so = self._sum_groups(x, self.a, self.no)
        if self.has_any_b:
            so.scatter_add_(1, self.b.unsqueeze(0).expand(replicas, -1), x * self.has_b)
        sro = self._sum_groups(x, self.ro1, self.nr * self.no)
        if self.has_any_b:
            sro.scatter_add_(1, self.ro2.unsqueeze(0).expand(replicas, -1), x * self.has_b)
        sp = self._sum_groups(x * self.has_b, self.pair, self.no * self.no)

        rider_sum = sr[:, self.r]
        oa = so[:, self.a]
        roa = sro[:, self.ro1]
        confsum = rider_sum + oa - roa - x
        if self.has_any_b:
            ob = so[:, self.b]
            rob = sro[:, self.ro2]
            pp = sp[:, self.pair]
            pair_union_neighbors = rider_sum + oa + ob - roa - rob - pp
            confsum = torch.where(self.has_b.unsqueeze(0), pair_union_neighbors, confsum)
        return confsum

    def raw_jx(self, x: torch.Tensor) -> torch.Tensor:
        return -0.25 * self._conflict_sum(x)

    def raw_field(self, x: torch.Tensor) -> torch.Tensor:
        return self.raw_jx(x) + self.h_raw

    def _estimate_spectral_radius(self, iters: int, seed: int) -> float:
        if self.m == 0:
            return 0.0
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        v = torch.randn((1, self.m), generator=gen, device=self.device, dtype=torch.float32)
        v = v / torch.linalg.vector_norm(v).clamp_min(1e-20)
        last_norm = 0.0
        for _ in range(max(3, int(iters))):
            w = self.raw_jx(v)
            n = torch.linalg.vector_norm(w).clamp_min(1e-20)
            last_norm = float(n.item())
            v = w / n
        w = self.raw_jx(v)
        rayleigh = float(torch.sum(v * w).item())
        return max(abs(rayleigh), last_norm, 1e-12)

    def _checkpoint_best(
        self,
        x: torch.Tensor,
        best_bits: np.ndarray,
        best_a: int,
        best_d: float,
    ) -> tuple[np.ndarray, int, float, float, int]:
        bits_t = (x >= 0).to(torch.float32)
        br = self._sum_groups(bits_t, self.r, self.nr)
        bo = self._sum_groups(bits_t, self.a, self.no)
        if self.has_any_b:
            bo.scatter_add_(1, self.b.unsqueeze(0).expand(x.shape[0], -1), bits_t * self.has_b)
        feas_t = (br.max(dim=1).values <= 1.0) & (bo.max(dim=1).values <= 1.0)
        assigned_t = bits_t @ self.count_t
        distance_t = bits_t @ self.distance_t
        feas = feas_t.detach().cpu().numpy()
        assigned = np.rint(assigned_t.detach().cpu().numpy()).astype(np.int32)
        distance = distance_t.detach().cpu().numpy().astype(np.float64)
        best_rep = -1
        for rr in np.flatnonzero(feas):
            aa, dd = int(assigned[rr]), float(distance[rr])
            if lex_better(aa, dd, best_a, best_d):
                best_a, best_d, best_rep = aa, dd, int(rr)
        if best_rep >= 0:
            best_bits = bits_t[best_rep].to(torch.int8).detach().cpu().numpy().copy()
        return best_bits, best_a, best_d, float(np.mean(feas)), best_rep

    def solve(
        self,
        cfg: SoftConfig,
        *,
        warm_start: np.ndarray,
        steps: int,
        replicas: int,
        seed: int,
        checkpoint_count: int = 20,
    ) -> dict[str, Any]:
        torch.manual_seed(int(seed))
        dtype = torch.float32
        t0 = time.perf_counter()

        warm_start = np.asarray(warm_start, dtype=np.int8)
        if len(warm_start) != self.m:
            raise ValueError("warm_start length mismatch")
        spin_np = (2 * warm_start.astype(np.float32) - 1.0).astype(np.float32)
        spin0 = torch.as_tensor(spin_np, dtype=dtype, device=self.device)
        x = cfg.warm_alpha * spin0.unsqueeze(0).expand(replicas, -1).clone()
        if cfg.warm_std > 0:
            x += cfg.warm_std * torch.randn_like(x)
        # Replica 0 is an exact incumbent-sign copy, guaranteeing one replica
        # begins exactly in the intended basin.
        x[0] = cfg.warm_alpha * spin0
        x.clamp_(-cfg.saturation, cfg.saturation)
        vel = torch.zeros_like(x)

        u = torch.linspace(0.0, 1.0, steps, device=self.device)
        uu = 0.5 * (1.0 + torch.tanh(3.0 * (2.0 * u - 1.0)) / math.tanh(3.0))
        pumps = cfg.pump_start + (cfg.pump_end - cfg.pump_start) * uu

        start_a, start_d, _ = eval_bits(warm_start, self.d)
        best_bits = warm_start.copy()
        best_a, best_d = start_a, start_d
        best_step = 0
        first_improve_step = -1
        last_feasible_rate = 1.0
        checks = 0
        # Roughly checkpoint_count observations, plus final step.
        checkpoint_every = max(1, int(math.ceil(steps / max(1, checkpoint_count))))

        for step_idx, pump in enumerate(pumps, start=1):
            local_field = self.raw_jx(x) / float(self.scale)
            force = float(pump) * x + cfg.coupling * (local_field + self.h)
            if cfg.noise > 0:
                force = force + cfg.noise * torch.randn_like(x)
            vel = cfg.momentum * vel + (1.0 - cfg.momentum) * force
            x = torch.clamp(x + vel, -cfg.saturation, cfg.saturation)

            if step_idx % checkpoint_every == 0 or step_idx == steps:
                old_a, old_d = best_a, best_d
                best_bits, best_a, best_d, last_feasible_rate, _ = self._checkpoint_best(
                    x, best_bits, best_a, best_d
                )
                checks += 1
                if lex_better(best_a, best_d, old_a, old_d):
                    best_step = step_idx
                    if first_improve_step < 0 and best_a == start_a and best_d < start_d - 1e-10:
                        first_improve_step = step_idx

        wall_ms = 1000.0 * (time.perf_counter() - t0)
        return {
            "bits": best_bits,
            "assigned": best_a,
            "distance": best_d,
            "wall_ms": wall_ms,
            "best_step": best_step,
            "first_same_card_improvement_step": first_improve_step,
            "same_card_improved": bool(best_a == start_a and best_d < start_d - 1e-10),
            "final_checkpoint_feasible_replica_rate": last_feasible_rate,
            "trajectory_checks": checks,
            "normalization": self.normalization,
            "normalization_scale": self.scale,
            "spectral_j_estimate": self.spectral_j_estimate,
            "row_sum_j_bound": self.row_sum_j_bound,
            "h_scale": self.h_scale,
        }


# -----------------------------------------------------------------------------
# Dense-vs-implicit validation
# -----------------------------------------------------------------------------


def dense_conflict_matrix(d: ProblemData, nr: int, no: int) -> np.ndarray:
    m = len(d.rider)
    A = np.zeros((m, m), dtype=np.float64)
    rider_groups: list[list[int]] = [[] for _ in range(nr)]
    order_groups: list[list[int]] = [[] for _ in range(no)]
    for i in range(m):
        rider_groups[int(d.rider[i])].append(i)
        order_groups[int(d.o1[i])].append(i)
        if int(d.o2[i]) >= 0:
            order_groups[int(d.o2[i])].append(i)
    for ids in rider_groups + order_groups:
        if len(ids) <= 1:
            continue
        ix = np.asarray(ids, dtype=np.int32)
        A[np.ix_(ix, ix)] = 1.0
    np.fill_diagonal(A, 0.0)
    return A


def validate_dense_implicit(
    d: ProblemData,
    device: str,
    seed: int,
    power_iters: int,
) -> dict[str, Any]:
    m = d.num_candidates
    A = dense_conflict_matrix(d, d.num_riders, d.num_orders)
    deg = A.sum(axis=1)
    J = -0.25 * A
    h = d.reward.astype(np.float64) / 2.0 - deg / 4.0
    eig = np.linalg.eigvalsh(J) if m else np.zeros(0)
    exact_spec = float(np.max(np.abs(eig))) if m else 0.0
    exact_scale = max(
        exact_spec,
        float(np.max(np.abs(h))) if m else 0.0,
        1e-12,
    )

    soft = WarmGroupSoftSpin(
        d,
        d.num_riders,
        d.num_orders,
        device,
        normalization="spectral_power",
        power_iters=power_iters,
        power_seed=seed,
    )
    degree_err = (
        float(np.max(np.abs(soft.degree_np - deg)))
        if m
        else 0.0
    )

    rng = np.random.default_rng(seed)
    x_np = rng.normal(size=(4, m)).astype(np.float32)
    x_t = torch.as_tensor(x_np, device=soft.device)
    implicit = (
        soft.raw_field(x_t)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    dense = x_np.astype(np.float64) @ J + h[None, :]
    field_err = (
        float(np.max(np.abs(implicit - dense)))
        if m
        else 0.0
    )

    Qs = 0.5 * A
    np.fill_diagonal(Qs, -d.reward.astype(np.float64))
    J2 = -0.5 * Qs
    np.fill_diagonal(J2, 0.0)
    h2 = -0.5 * Qs.sum(axis=1)
    offset = float(
        0.25 * Qs.sum()
        + 0.25 * np.trace(Qs)
    )
    energy_err = 0.0
    for _ in range(16):
        b = rng.integers(
            0, 2, size=m, dtype=np.int8
        ).astype(np.float64)
        s = 2.0 * b - 1.0
        q = float(b @ Qs @ b)
        ie = float(
            -0.5 * s @ J2 @ s
            - h2 @ s
            + offset
        )
        energy_err = max(
            energy_err,
            abs(q - ie),
        )

    rel_spec_err = (
        abs(soft.spectral_j_estimate - exact_spec)
        / max(exact_spec, 1e-12)
    )
    rel_scale_err = (
        abs(soft.scale - exact_scale)
        / max(exact_scale, 1e-12)
    )
    passed = (
        degree_err <= 1e-9
        and field_err <= 2e-5
        and energy_err <= 1e-7 * max(1.0, m)
        and rel_spec_err <= 0.08
    )
    return {
        "num_candidates": m,
        "degree_max_abs_error": degree_err,
        "field_max_abs_error": field_err,
        "qubo_ising_max_abs_error": energy_err,
        "dense_spectral_radius": exact_spec,
        "implicit_spectral_estimate": soft.spectral_j_estimate,
        "spectral_relative_error": rel_spec_err,
        "dense_normalization_scale": exact_scale,
        "implicit_normalization_scale": soft.scale,
        "normalization_relative_error": rel_scale_err,
        "passed": bool(passed),
    }


# -----------------------------------------------------------------------------
# PT with ladder diagnostics / replica round trips
# -----------------------------------------------------------------------------


def _swap_rows(arr: np.ndarray, mask: np.ndarray, j: int) -> None:
    if not np.any(mask):
        return
    tmp = arr[mask, j].copy()
    arr[mask, j] = arr[mask, j + 1]
    arr[mask, j + 1] = tmp


def run_sparse_pt_diagnostics(
    *,
    start: np.ndarray,
    d: ProblemData,
    nr: int,
    no: int,
    tmin: float,
    tmax: float,
    ntemps: int,
    ensembles: int,
    sweeps: int,
    seed: int,
    max_wall_s: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    m = len(start)
    temps = np.geomspace(max(tmin, 1e-12), max(tmax, tmin * 1.0001), ntemps)
    beta = 1.0 / temps
    states = np.broadcast_to(start, (ensembles, ntemps, m)).copy().astype(np.int8)
    occ = replicate_occ(start_occupancy(start, d, nr, no), (ensembles, ntemps))
    a0, d0, _ = eval_bits(start, d)
    assigned = np.full((ensembles, ntemps), a0, dtype=np.int32)
    dist = np.full((ensembles, ntemps), d0, dtype=np.float64)
    e0 = -float(start @ d.reward)
    energy = np.full((ensembles, ntemps), e0, dtype=np.float64)

    best_bits = start.copy()
    best_a, best_d = a0, d0
    first_improve = -1
    completed = 0
    accepted_by_temp = np.zeros(ntemps, dtype=np.int64)
    attempted_by_temp = np.zeros(ntemps, dtype=np.int64)
    swap_acc_edge = np.zeros(max(0, ntemps - 1), dtype=np.int64)
    swap_att_edge = np.zeros(max(0, ntemps - 1), dtype=np.int64)

    # Track physical walkers as swaps exchange temperature positions.
    nwalkers = ensembles * ntemps
    walker_id = np.arange(nwalkers, dtype=np.int32).reshape(ensembles, ntemps)
    # phase: 0 = has not established a cold start; 1 = seen cold, waiting hot;
    #        2 = seen hot after cold, waiting to return cold.
    phase = np.zeros(nwalkers, dtype=np.int8)
    cycle_start = np.full(nwalkers, -1, dtype=np.int32)
    roundtrips = np.zeros(nwalkers, dtype=np.int32)
    roundtrip_durations: list[int] = []
    for wid in walker_id[:, 0]:
        phase[int(wid)] = 1
        cycle_start[int(wid)] = 0

    t0 = time.perf_counter()
    tempmat = temps[None, :]

    for sw in range(sweeps):
        for i in rng.permutation(m):
            b = states[:, :, i].astype(np.int16)
            nc = neighbor_count(b, occ, d, int(i)).astype(np.float64)
            delta = (1 - 2 * b) * (-float(d.reward[i]) + nc)
            acc = delta <= 0.0
            up = ~acc
            if np.any(up):
                T = np.broadcast_to(tempmat, delta.shape)
                p = np.exp(-np.minimum(delta[up] / T[up], 700.0))
                acc[up] = rng.random(np.sum(up)) < p
            dx = np.where(acc, 1 - 2 * b, 0).astype(np.int16)
            attempted_by_temp += ensembles
            accepted_by_temp += np.sum(acc, axis=0).astype(np.int64)
            if np.any(dx):
                states[:, :, i] = (states[:, :, i].astype(np.int16) + dx).astype(np.int8)
                apply_flip_update(occ, d, int(i), dx)
                assigned += dx * int(d.count[i])
                dist += dx * float(d.distance[i])
                energy += np.where(acc, delta, 0.0)

        parity = sw % 2
        for j in range(parity, ntemps - 1, 2):
            log_ratio = (beta[j] - beta[j + 1]) * (energy[:, j] - energy[:, j + 1])
            swap = np.log(rng.random(ensembles)) < np.minimum(0.0, log_ratio)
            swap_att_edge[j] += ensembles
            swap_acc_edge[j] += int(np.sum(swap))
            if np.any(swap):
                _swap_rows(states, swap, j)
                _swap_rows(occ.rider, swap, j)
                _swap_rows(occ.order, swap, j)
                _swap_rows(occ.rider_order, swap, j)
                _swap_rows(occ.order_pair, swap, j)
                _swap_rows(assigned, swap, j)
                _swap_rows(dist, swap, j)
                _swap_rows(energy, swap, j)
                _swap_rows(walker_id, swap, j)

        # Update true cold -> hot -> cold walker cycles.
        sweep_no = sw + 1
        for wid0 in walker_id[:, 0].tolist():
            wid = int(wid0)
            if phase[wid] == 0:
                phase[wid] = 1
                cycle_start[wid] = sweep_no
            elif phase[wid] == 2:
                roundtrips[wid] += 1
                if cycle_start[wid] >= 0:
                    roundtrip_durations.append(sweep_no - int(cycle_start[wid]))
                phase[wid] = 1
                cycle_start[wid] = sweep_no
        for wid0 in walker_id[:, -1].tolist():
            wid = int(wid0)
            if phase[wid] == 1:
                phase[wid] = 2

        completed = sweep_no
        feas = feasible_mask(occ)
        er, et = np.where(feas)
        for ee, tt in zip(er.tolist(), et.tolist()):
            aa, dd = int(assigned[ee, tt]), float(dist[ee, tt])
            if lex_better(aa, dd, best_a, best_d):
                best_a, best_d = aa, dd
                best_bits = states[ee, tt].copy()
            if first_improve < 0 and aa == a0 and dd < d0 - 1e-10:
                first_improve = sweep_no

        if time.perf_counter() - t0 >= max_wall_s:
            break

    wall = time.perf_counter() - t0
    acc_rate_temp = accepted_by_temp / np.maximum(1, attempted_by_temp)
    swap_rate_edge = swap_acc_edge / np.maximum(1, swap_att_edge)
    return {
        "bits": best_bits,
        "assigned": best_a,
        "distance": best_d,
        "wall_ms": 1000.0 * wall,
        "sweeps_completed": completed,
        "first_same_card_improvement_sweep": first_improve,
        "same_card_improved": bool(best_a == a0 and best_d < d0 - 1e-10),
        "acceptance_rate": float(np.sum(accepted_by_temp) / max(1, np.sum(attempted_by_temp))),
        "acceptance_cold": float(acc_rate_temp[0]),
        "acceptance_hot": float(acc_rate_temp[-1]),
        "acceptance_by_temp_json": json.dumps(acc_rate_temp.tolist()),
        "swap_rate": float(np.sum(swap_acc_edge) / max(1, np.sum(swap_att_edge))),
        "swap_rate_min_edge": float(np.min(swap_rate_edge)) if len(swap_rate_edge) else math.nan,
        "swap_rate_median_edge": float(np.median(swap_rate_edge)) if len(swap_rate_edge) else math.nan,
        "swap_rate_max_edge": float(np.max(swap_rate_edge)) if len(swap_rate_edge) else math.nan,
        "swap_rate_by_edge_json": json.dumps(swap_rate_edge.tolist()),
        "roundtrips_total": int(np.sum(roundtrips)),
        "walkers_with_roundtrip": int(np.sum(roundtrips > 0)),
        "roundtrip_fraction_walkers": float(np.mean(roundtrips > 0)),
        "median_roundtrip_sweeps": float(np.median(roundtrip_durations)) if roundtrip_durations else math.nan,
        "min_roundtrip_sweeps": float(np.min(roundtrip_durations)) if roundtrip_durations else math.nan,
        "tmin": float(tmin),
        "tmax": float(tmax),
        "temperature_span": float(tmax / max(tmin, 1e-15)),
        "ntemps": int(ntemps),
        "ensembles": int(ensembles),
    }


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------


def aggregate_scale(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted(
        {(r["scale"], r["solver"], r.get("budget_label", "")) for r in rows if r.get("phase") == "test"}
    )
    for scale, solver, bud in keys:
        z = [
            r for r in rows
            if r.get("phase") == "test"
            and r["scale"] == scale
            and r["solver"] == solver
            and r.get("budget_label", "") == bud
        ]
        if not z:
            continue
        walls = np.asarray([float(r["wall_ms"]) for r in z], dtype=float)
        gaps = np.asarray([float(r["cardinality_gap"]) for r in z], dtype=float)
        dg = np.asarray(
            [float(r["distance_gap_pct"]) for r in z if math.isfinite(float(r["distance_gap_pct"]))],
            dtype=float,
        )
        rec = np.asarray(
            [float(r.get("refinement_fraction", math.nan)) for r in z if math.isfinite(float(r.get("refinement_fraction", math.nan)))],
            dtype=float,
        )
        row = {
            "scale": scale,
            "solver": solver,
            "budget_label": bud,
            "n": len(z),
            "median_candidates": float(np.median([r["num_candidates"] for r in z])),
            "median_wall_ms": float(np.median(walls)),
            "p90_wall_ms": float(np.quantile(walls, 0.9)),
            "mean_cardinality_gap": float(np.mean(gaps)),
            "pct_best_cardinality": float(np.mean(gaps == 0)),
            "median_distance_gap_pct_when_best_card": float(np.median(dg)) if len(dg) else math.nan,
            "mean_refinement_fraction": float(np.mean(rec)) if len(rec) else math.nan,
            "median_refinement_fraction": float(np.median(rec)) if len(rec) else math.nan,
            "pct_matches_reference": float(np.mean([bool(r["matches_reference"]) for r in z])),
        }
        for key in ("scip_build_ms", "scip_warm_start_ms", "scip_optimize_ms", "pt_acceptance_rate", "pt_swap_rate"):
            vals = [float(r[key]) for r in z if key in r and math.isfinite(float(r[key]))]
            if vals:
                row[f"median_{key}"] = float(np.median(vals))
        out.append(row)
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def _parse_map(s: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        k, val = item.split("=", 1)
        out[k.strip()] = val.strip()
    return out


def _discover_public_sets(prefix: str) -> dict[str, str]:
    root = REPO / "instances" / "raw"
    out: dict[str, str] = {}
    if not root.exists():
        return out
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith(prefix):
            scale = p.name[len(prefix):]
            if scale:
                out[scale] = p.name
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-sets",
        default="",
        help=(
            "Comma-separated scale=set map. If omitted, directories "
            "instances/raw/raw_scaling_train_* are auto-discovered."
        ),
    )
    ap.add_argument(
        "--test-sets",
        default="",
        help=(
            "Comma-separated scale=set map. If omitted, directories "
            "instances/raw/raw_scaling_test_* are auto-discovered."
        ),
    )
    ap.add_argument(
        "--validation-set",
        default="raw_scaling_validation",
    )
    ap.add_argument(
        "--output",
        default="raw_scaling/results",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--validation-limit", type=int, default=2)
    ap.add_argument("--train-limit", type=int, default=3)
    ap.add_argument("--test-limit", type=int, default=8)
    ap.add_argument("--reference-scip-seconds", type=float, default=8.0)
    ap.add_argument("--scip-deadlines", default="0.1,0.25,0.5,1,2,5")

    ap.add_argument("--pt-sweeps", default="1,2,4,8")
    ap.add_argument("--pt-ntemps", type=int, default=16)
    ap.add_argument("--pt-ensembles", type=int, default=2)
    ap.add_argument("--pt-tmin", type=float, default=0.0005)
    ap.add_argument("--pt-tmax", type=float, default=1.5)
    ap.add_argument("--pt-max-wall", type=float, default=60.0)
    ap.add_argument("--pt-diagnostic-limit", type=int, default=1)
    ap.add_argument("--pt-diagnostic-ntemps", default="8,16,32")
    ap.add_argument("--pt-diagnostic-sweeps", type=int, default=64)
    ap.add_argument("--pt-diagnostic-ensembles", type=int, default=2)
    ap.add_argument("--pt-diagnostic-max-wall", type=float, default=120.0)

    ap.add_argument(
        "--simcim-normalization",
        choices=["spectral_power", "row_sum", "none"],
        default="spectral_power",
    )
    ap.add_argument("--simcim-power-iters", type=int, default=24)
    ap.add_argument("--simcim-train-steps", type=int, default=250)
    ap.add_argument("--simcim-train-replicas", type=int, default=24)
    ap.add_argument("--simcim-step-budgets", default="100,250,500,1000")
    ap.add_argument("--simcim-replicas", type=int, default=32)
    ap.add_argument("--simcim-checkpoints", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2200000)
    args = ap.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    train_map = (
        _parse_map(args.train_sets)
        if args.train_sets.strip()
        else _discover_public_sets("raw_scaling_train_")
    )
    test_map = (
        _parse_map(args.test_sets)
        if args.test_sets.strip()
        else _discover_public_sets("raw_scaling_test_")
    )
    if not train_map:
        raise ValueError(
            "No raw-scaling training sets found. Supply --train-sets "
            "or create instances/raw/raw_scaling_train_<scale>/."
        )
    if not test_map:
        raise ValueError(
            "No raw-scaling test sets found. Supply --test-sets "
            "or create instances/raw/raw_scaling_test_<scale>/."
        )
    if set(train_map) != set(test_map):
        raise ValueError(
            "train/test scales must match; "
            f"train={sorted(train_map)}, test={sorted(test_map)}"
        )

    validation_rows: list[dict[str, Any]] = []
    print("\n=== DENSE vs IMPLICIT VALIDATION ===", flush=True)
    validation = load_problem_set(args.validation_set)[: args.validation_limit]
    for ii, d in enumerate(validation):
        row = validate_dense_implicit(
            d,
            args.device,
            seed=args.seed + ii,
            power_iters=args.simcim_power_iters,
        )
        row.update(
            {
                "instance_index": int(d.source_index),
                "instance_id": d.instance_id,
            }
        )
        validation_rows.append(row)
        print(
            f"  {ii}: m={row['num_candidates']} "
            f"field_err={row['field_max_abs_error']:.3e} "
            f"spec_rel={row['spectral_relative_error']:.3e} "
            f"passed={row['passed']}",
            flush=True,
        )
    write_csv(outdir / "validation_dense_implicit.csv", validation_rows)
    if not validation_rows or not all(bool(r["passed"]) for r in validation_rows):
        raise RuntimeError(
            "Dense-vs-implicit validation failed. Refusing to run "
            "large-instance SimCIM. See "
            f"{outdir/'validation_dense_implicit.csv'}"
        )

    tuning_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    raw_rows: list[dict[str, Any]] = []
    inst_rows: list[dict[str, Any]] = []
    pt_diag_rows: list[dict[str, Any]] = []

    for si, scale in enumerate(train_map):
        print(
            f"\n=== SCALE {scale}: warm SimCIM TRAINING ===",
            flush=True,
        )
        train = load_problem_set(train_map[scale])[: args.train_limit]
        train_probs = []
        for ii, d in enumerate(train):
            greedy = greedy_pair_first(d)
            ga, gd, _ = eval_bits(greedy, d)
            ref = warm_scip(
                d,
                greedy,
                args.reference_scip_seconds,
                args.seed + si * 10000 + ii,
            )
            ra, rd = ref["assigned"], ref["distance"]
            if lex_better(ga, gd, ra, rd):
                ra, rd = ga, gd
            soft = WarmGroupSoftSpin(
                d,
                d.num_riders,
                d.num_orders,
                args.device,
                normalization=args.simcim_normalization,
                power_iters=args.simcim_power_iters,
                power_seed=args.seed + 1000 * si + ii,
            )
            train_probs.append(
                (d, greedy, ga, gd, ra, rd, soft)
            )

        cfg_scores: list[tuple[float, int]] = []
        for ci, cfg in enumerate(SOFT_GRID):
            scores: list[float] = []
            for ii, (d, greedy, ga, gd, ra, rd, soft) in enumerate(train_probs):
                z = soft.solve(
                    cfg,
                    warm_start=greedy,
                    steps=args.simcim_train_steps,
                    replicas=args.simcim_train_replicas,
                    seed=args.seed + 100000 * si + 1000 * ci + ii,
                    checkpoint_count=args.simcim_checkpoints,
                )
                score = tuning_score(
                    ga,
                    gd,
                    z["assigned"],
                    z["distance"],
                    ra,
                    rd,
                )
                scores.append(score)
                tuning_rows.append(
                    {
                        "scale": scale,
                        "config_id": ci,
                        **asdict(cfg),
                        "instance_index": int(d.source_index),
                        "instance_id": d.instance_id,
                        "num_candidates": d.num_candidates,
                        "greedy_assigned": ga,
                        "greedy_distance": gd,
                        "assigned": z["assigned"],
                        "distance": z["distance"],
                        "ref_assigned": ra,
                        "ref_distance": rd,
                        "refinement_fraction": refinement_fraction(
                            ga,
                            gd,
                            z["assigned"],
                            z["distance"],
                            ra,
                            rd,
                        ),
                        "tuning_score": score,
                        "wall_ms": z["wall_ms"],
                        "best_step": z["best_step"],
                        "first_same_card_improvement_step": z[
                            "first_same_card_improvement_step"
                        ],
                        "feasible_replica_rate": z[
                            "final_checkpoint_feasible_replica_rate"
                        ],
                        "normalization_scale": z["normalization_scale"],
                        "spectral_j_estimate": z["spectral_j_estimate"],
                    }
                )
            mean_score = (
                float(np.mean(scores))
                if scores
                else -math.inf
            )
            cfg_scores.append((mean_score, ci))
            print(
                f"  config {ci:02d}: "
                f"mean tuning score={mean_score:.5f}",
                flush=True,
            )
            write_csv(outdir / "simcim_tuning.csv", tuning_rows)

        _, best_ci = max(cfg_scores)
        best_cfg = SOFT_GRID[best_ci]
        selected[scale] = {
            "config_id": best_ci,
            **asdict(best_cfg),
        }
        write_json(outdir / "simcim_selected.json", selected)
        print(
            f"Selected config {best_ci}: {best_cfg}",
            flush=True,
        )

        print(
            f"=== SCALE {scale}: HELD-OUT RAW TEST ===",
            flush=True,
        )
        test = load_problem_set(test_map[scale])[: args.test_limit]
        for ii, d in enumerate(test):
            tg = time.perf_counter()
            greedy = greedy_pair_first(d)
            greedy_ms = 1000.0 * (time.perf_counter() - tg)
            ga, gd, _ = eval_bits(greedy, d)

            ref = warm_scip(
                d,
                greedy,
                args.reference_scip_seconds,
                args.seed + 500000 + si * 10000 + ii,
            )
            ra, rd = ref["assigned"], ref["distance"]
            if lex_better(ga, gd, ra, rd):
                ra, rd = ga, gd

            inst_rows.append(
                {
                    "scale": scale,
                    "instance_index": int(d.source_index),
                    "instance_id": d.instance_id,
                    "num_riders": d.num_riders,
                    "num_orders": d.num_orders,
                    "num_candidates": d.num_candidates,
                    "candidate_generation_included": False,
                    "greedy_assigned": ga,
                    "greedy_distance": gd,
                    "greedy_ms": greedy_ms,
                    "reference_assigned": ra,
                    "reference_distance": rd,
                    "reference_scip_status": ref["status"],
                    "reference_scip_wall_ms": ref["wall_ms"],
                    "reference_scip_build_ms": ref["build_ms"],
                    "reference_scip_warm_start_ms": ref["warm_start_ms"],
                    "reference_scip_optimize_ms": ref["optimize_ms"],
                    "reference_scip_nodes": ref["nodes"],
                }
            )

            def addrow(
                solver: str,
                budget: str,
                a: int,
                dd: float,
                wall_ms: float,
                extra: dict[str, Any] | None = None,
            ) -> None:
                cardgap = ra - a
                distgap = (
                    100.0 * (dd - rd) / max(abs(rd), 1e-12)
                    if a == ra
                    else math.nan
                )
                row: dict[str, Any] = {
                    "phase": "test",
                    "scale": scale,
                    "instance_index": int(d.source_index),
                    "instance_id": d.instance_id,
                    "num_candidates": d.num_candidates,
                    "solver": solver,
                    "budget_label": str(budget),
                    "assigned": a,
                    "distance": dd,
                    "wall_ms": wall_ms,
                    "reference_assigned": ra,
                    "reference_distance": rd,
                    "cardinality_gap": cardgap,
                    "distance_gap_pct": distgap,
                    "refinement_fraction": refinement_fraction(
                        ga, gd, a, dd, ra, rd
                    ),
                    "matches_reference": bool(
                        cardgap == 0
                        and abs(dd - rd) <= 1e-8
                    ),
                }
                if extra:
                    row.update(extra)
                raw_rows.append(row)

            addrow("greedy", "single", ga, gd, greedy_ms)
            addrow(
                "warm_scip_reference",
                f"{args.reference_scip_seconds:g}s",
                ref["assigned"],
                ref["distance"],
                ref["wall_ms"],
                {
                    "scip_status": ref["status"],
                    "scip_gap": ref["gap"],
                    "scip_nodes": ref["nodes"],
                    "scip_build_ms": ref["build_ms"],
                    "scip_warm_start_ms": ref["warm_start_ms"],
                    "scip_optimize_ms": ref["optimize_ms"],
                },
            )

            for dl in [
                float(x)
                for x in args.scip_deadlines.split(",")
                if x.strip()
            ]:
                z = warm_scip(
                    d,
                    greedy,
                    dl,
                    args.seed
                    + 600000
                    + si * 10000
                    + ii
                    + int(dl * 1000),
                )
                addrow(
                    "warm_scip",
                    f"{dl:g}s",
                    z["assigned"],
                    z["distance"],
                    z["wall_ms"],
                    {
                        "scip_status": z["status"],
                        "scip_gap": z["gap"],
                        "scip_nodes": z["nodes"],
                        "scip_build_ms": z["build_ms"],
                        "scip_warm_start_ms": z["warm_start_ms"],
                        "scip_optimize_ms": z["optimize_ms"],
                    },
                )
                write_csv(outdir / "raw_runs.csv", raw_rows)

            for sw in [
                int(x)
                for x in args.pt_sweeps.split(",")
                if x.strip()
            ]:
                z = run_sparse_pt_diagnostics(
                    start=greedy,
                    d=d,
                    nr=d.num_riders,
                    no=d.num_orders,
                    tmin=args.pt_tmin,
                    tmax=args.pt_tmax,
                    ntemps=args.pt_ntemps,
                    ensembles=args.pt_ensembles,
                    sweeps=sw,
                    seed=args.seed
                    + 700000
                    + si * 10000
                    + ii * 100
                    + sw,
                    max_wall_s=args.pt_max_wall,
                )
                addrow(
                    "pt",
                    f"{sw}sweeps",
                    z["assigned"],
                    z["distance"],
                    z["wall_ms"],
                    {
                        "pt_sweeps_completed": z["sweeps_completed"],
                        "pt_swap_rate": z["swap_rate"],
                        "pt_acceptance_rate": z["acceptance_rate"],
                        "pt_acceptance_cold": z["acceptance_cold"],
                        "pt_acceptance_hot": z["acceptance_hot"],
                        "pt_first_improve_sweep": z[
                            "first_same_card_improvement_sweep"
                        ],
                        "pt_roundtrips_total": z["roundtrips_total"],
                    },
                )
                write_csv(outdir / "raw_runs.csv", raw_rows)

            soft = WarmGroupSoftSpin(
                d,
                d.num_riders,
                d.num_orders,
                args.device,
                normalization=args.simcim_normalization,
                power_iters=args.simcim_power_iters,
                power_seed=args.seed
                + 800000
                + si * 10000
                + ii,
            )
            for steps in [
                int(x)
                for x in args.simcim_step_budgets.split(",")
                if x.strip()
            ]:
                z = soft.solve(
                    best_cfg,
                    warm_start=greedy,
                    steps=steps,
                    replicas=args.simcim_replicas,
                    seed=args.seed
                    + 810000
                    + si * 10000
                    + ii * 100
                    + steps,
                    checkpoint_count=args.simcim_checkpoints,
                )
                addrow(
                    "simcim_warm",
                    f"{steps}steps",
                    z["assigned"],
                    z["distance"],
                    z["wall_ms"],
                    {
                        "simcim_steps": steps,
                        "simcim_replicas": args.simcim_replicas,
                        "simcim_config_id": best_ci,
                        "simcim_best_step": z["best_step"],
                        "simcim_first_improve_step": z[
                            "first_same_card_improvement_step"
                        ],
                        "simcim_same_card_improved": z[
                            "same_card_improved"
                        ],
                        "simcim_final_feasible_replica_rate": z[
                            "final_checkpoint_feasible_replica_rate"
                        ],
                        "simcim_normalization_scale": z[
                            "normalization_scale"
                        ],
                        "simcim_spectral_j_estimate": z[
                            "spectral_j_estimate"
                        ],
                    },
                )
                write_csv(outdir / "raw_runs.csv", raw_rows)

            if ii < args.pt_diagnostic_limit:
                for nt in [
                    int(x)
                    for x in args.pt_diagnostic_ntemps.split(",")
                    if x.strip()
                ]:
                    z = run_sparse_pt_diagnostics(
                        start=greedy,
                        d=d,
                        nr=d.num_riders,
                        no=d.num_orders,
                        tmin=args.pt_tmin,
                        tmax=args.pt_tmax,
                        ntemps=nt,
                        ensembles=args.pt_diagnostic_ensembles,
                        sweeps=args.pt_diagnostic_sweeps,
                        seed=args.seed
                        + 900000
                        + si * 10000
                        + ii * 100
                        + nt,
                        max_wall_s=args.pt_diagnostic_max_wall,
                    )
                    pt_diag_rows.append(
                        {
                            "scale": scale,
                            "instance_index": int(d.source_index),
                            "instance_id": d.instance_id,
                            "num_candidates": d.num_candidates,
                            "greedy_assigned": ga,
                            "greedy_distance": gd,
                            "reference_assigned": ra,
                            "reference_distance": rd,
                            "ntemps": nt,
                            "requested_sweeps": args.pt_diagnostic_sweeps,
                            "sweeps_completed": z["sweeps_completed"],
                            "ensembles": args.pt_diagnostic_ensembles,
                            "tmin": args.pt_tmin,
                            "tmax": args.pt_tmax,
                            "assigned": z["assigned"],
                            "distance": z["distance"],
                            "refinement_fraction": refinement_fraction(
                                ga,
                                gd,
                                z["assigned"],
                                z["distance"],
                                ra,
                                rd,
                            ),
                            "wall_ms": z["wall_ms"],
                            "acceptance_rate": z["acceptance_rate"],
                            "acceptance_cold": z["acceptance_cold"],
                            "acceptance_hot": z["acceptance_hot"],
                            "swap_rate": z["swap_rate"],
                            "swap_rate_min_edge": z[
                                "swap_rate_min_edge"
                            ],
                            "swap_rate_median_edge": z[
                                "swap_rate_median_edge"
                            ],
                            "swap_rate_max_edge": z[
                                "swap_rate_max_edge"
                            ],
                            "roundtrips_total": z["roundtrips_total"],
                            "walkers_with_roundtrip": z[
                                "walkers_with_roundtrip"
                            ],
                            "roundtrip_fraction_walkers": z[
                                "roundtrip_fraction_walkers"
                            ],
                            "median_roundtrip_sweeps": z[
                                "median_roundtrip_sweeps"
                            ],
                            "min_roundtrip_sweeps": z[
                                "min_roundtrip_sweeps"
                            ],
                            "first_same_card_improvement_sweep": z[
                                "first_same_card_improvement_sweep"
                            ],
                            "acceptance_by_temp_json": z[
                                "acceptance_by_temp_json"
                            ],
                            "swap_rate_by_edge_json": z[
                                "swap_rate_by_edge_json"
                            ],
                        }
                    )
                    write_csv(
                        outdir / "pt_diagnostics.csv",
                        pt_diag_rows,
                    )

            write_csv(outdir / "instance_summary.csv", inst_rows)
            print(
                f"  test {ii}: m={d.num_candidates} "
                f"greedy=({ga},{gd:.3f}) "
                f"ref=({ra},{rd:.3f})",
                flush=True,
            )

    write_csv(outdir / "validation_dense_implicit.csv", validation_rows)
    write_csv(outdir / "simcim_tuning.csv", tuning_rows)
    write_json(outdir / "simcim_selected.json", selected)
    write_csv(outdir / "raw_runs.csv", raw_rows)
    write_csv(outdir / "pt_diagnostics.csv", pt_diag_rows)
    write_csv(outdir / "instance_summary.csv", inst_rows)
    scale_summary = aggregate_scale(raw_rows)
    write_csv(outdir / "scale_summary.csv", scale_summary)

    public_config = vars(args).copy()
    public_config["resolved_train_sets"] = train_map
    public_config["resolved_test_sets"] = test_map
    public_config["candidate_generation_included"] = False
    write_json(outdir / "run_config.json", public_config)

    summary = {
        "validation_all_passed": bool(
            validation_rows
            and all(bool(r["passed"]) for r in validation_rows)
        ),
        "validation_cases": len(validation_rows),
        "scales": list(train_map.keys()),
        "selected_simcim_configs": selected,
        "n_test_rows": len(raw_rows),
        "n_pt_diagnostic_rows": len(pt_diag_rows),
        "candidate_generation_included": False,
    }
    write_json(outdir / "summary.json", summary)
    print(f"\nWrote raw-scaling results to {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
