from __future__ import annotations

"""
EulerQ — Large-Instance Flux Landscape Hardness Analysis
=========================================================

Purpose
-------
This experiment studies whether high-quality incumbent solutions of complete
two-order Flux candidate-selection problems are separated from better feasible
solutions by landscape structure that is difficult for local single-variable
thermal dynamics.

The analysis is performed on representative large Flux instances and focuses on
the relationship between solution improvement, Hamming distance, conflict
penalties, and intermediate QUBO barriers.

Experiments
-----------
A. Thermal-search analysis
   - run simulated annealing and parallel tempering on complete Flux problems;
   - calibrate temperature ranges using the relevant energy scales of the QUBO;
   - ensure that the cold end can resolve improvements between competing
     feasible solutions while the hot end can cross conflict penalties.

B. Nearest better feasible solution
   - use a mixed-integer optimization model to determine the minimum Hamming
     distance from the incumbent to any strictly better feasible solution;
   - report this distance as a measure of the number of coordinated candidate
     changes required for improvement.

C. Exact local barrier analysis
   - for sufficiently small nearest-better transitions, enumerate the
     corresponding Hamming subcube;
   - compute the minimum one-variable-path barrier between the incumbent and
     the improved feasible configuration;
   - compare the intermediate barrier with the final energy improvement.

D. Coherent local transition test
   - for sufficiently small extracted subproblems, perform an ideal
     transverse-field reverse-annealing state-vector simulation;
   - evaluate coherent transfer between the incumbent configuration and the
     improved configuration.

Interpretation
--------------
The local quantum simulation is a mechanism experiment and does not constitute
an end-to-end quantum-advantage benchmark. The active variables defining the
small local subproblem are identified using classical analysis.

The primary outputs of this experiment are the coordinated transition width,
the improvement in objective energy, the minimum local barrier, and the
barrier-to-improvement ratio for representative Flux-derived transitions.

Implementation
--------------
The large-instance calculations use an implicit representation of the Flux
QUBO. No dense global QUBO matrix or dense global conflict-adjacency matrix is
constructed.
"""

import argparse
import csv
import heapq
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pyscipopt import Model, quicksum

HERE = Path(__file__).resolve()
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


from common.problem_data import ProblemData, load_problem_set


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")


# -----------------------------------------------------------------------------
# Candidate arrays and exact business/QUBO scaling
# -----------------------------------------------------------------------------

def eval_bits(bits: np.ndarray, d: ProblemData) -> tuple[int, float, float]:
    b = np.asarray(bits, dtype=np.int8)
    assigned = int(b @ d.count.astype(np.int32))
    distance = float(b @ d.distance)
    business = float(d.M * assigned - distance)
    return assigned, distance, business


def greedy_pair_first(d: ProblemData) -> np.ndarray:
    m = d.num_candidates
    out = np.zeros(m, dtype=np.int8)
    used_riders: set[int] = set()
    used_orders: set[int] = set()
    order = sorted(
        range(m),
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


# -----------------------------------------------------------------------------
# Full SCIP
# -----------------------------------------------------------------------------

def run_scip(
    d: ProblemData,
    limit_s: float,
    seed: int,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    model = Model("flux_candidate_selection")
    model.hideOutput()
    try:
        model.setIntParam("randomization/randomseedshift", int(seed) % 2147483647)
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
        quicksum(float(d.raw_reward[i]) * x[i] for i in range(len(x))), "maximize"
    )
    build_ms = 1000.0 * (time.perf_counter() - t0)
    model.setRealParam("limits/time", float(limit_s))
    ts = time.perf_counter()
    model.optimize()
    opt_ms = 1000.0 * (time.perf_counter() - ts)
    status = str(model.getStatus())
    bits = np.zeros(len(x), dtype=np.int8)
    if model.getNSols() > 0:
        sol = model.getBestSol()
        for i, var in enumerate(x):
            bits[i] = int(model.getSolVal(sol, var) > 0.5)
    a, dist, score = eval_bits(bits, d)
    try:
        gap = float(model.getGap())
    except Exception:
        gap = math.nan
    try:
        nodes = int(model.getNNodes())
    except Exception:
        nodes = -1
    return {
        "bits": bits,
        "status": status,
        "optimal": status.lower() == "optimal",
        "gap": gap,
        "nodes": nodes,
        "assigned": a,
        "distance": dist,
        "business": score,
        "build_ms": build_ms,
        "optimize_ms": opt_ms,
        "solver_ms": build_ms + opt_ms,
    }


# -----------------------------------------------------------------------------
# Sparse/implicit unique-conflict thermal dynamics
# -----------------------------------------------------------------------------

@dataclass
class Occ:
    rider: np.ndarray
    order: np.ndarray
    rider_order: np.ndarray
    order_pair: np.ndarray


def start_occupancy(bits: np.ndarray, d: ProblemData, nr: int, no: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def replicate_occ(base: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], shape: tuple[int, ...]) -> Occ:
    ro, oo, r_o, op = base
    def rep(arr: np.ndarray) -> np.ndarray:
        return np.broadcast_to(arr, shape + arr.shape).copy()
    return Occ(rep(ro), rep(oo), rep(r_o), rep(op))


def neighbor_count(states_i: np.ndarray, occ: Occ, d: ProblemData, i: int) -> np.ndarray:
    """Exact number of selected UNIQUE conflicting neighbors of candidate i.

    Uses inclusion-exclusion over the rider/order conflict sets; no adjacency
    matrix is needed.
    """
    r, a, b = int(d.rider[i]), int(d.o1[i]), int(d.o2[i])
    if b < 0:
        union = occ.rider[..., r] + occ.order[..., a] - occ.rider_order[..., r, a]
    else:
        # R union Oa union Ob. Triple intersection is the exact same pair
        # candidate (r,a,b), therefore equals states_i for this unique key.
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


def apply_flip_update(occ: Occ, d: ProblemData, i: int, dx: np.ndarray) -> None:
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
    return (np.max(occ.rider, axis=-1) <= 1) & (np.max(occ.order, axis=-1) <= 1)


def run_sparse_sa(
    *, start: np.ndarray, d: ProblemData, nr: int, no: int,
    temperature: float, runs: int, sweeps: int, seed: int, max_wall_s: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    m = len(start)
    states = np.broadcast_to(start, (runs, m)).copy().astype(np.int8)
    occ = replicate_occ(start_occupancy(start, d, nr, no), (runs,))
    assigned0, dist0, _ = eval_bits(start, d)
    assigned = np.full(runs, assigned0, dtype=np.int32)
    dist = np.full(runs, dist0, dtype=np.float64)
    energy0 = -float(start @ d.reward)
    energy = np.full(runs, energy0, dtype=np.float64)

    best_bits = start.copy()
    best_a, best_d = assigned0, dist0
    first_improve = -1
    accepted = 0
    attempted = 0
    completed = 0
    t0 = time.perf_counter()

    for sw in range(sweeps):
        for i in rng.permutation(m):
            b = states[:, i].astype(np.int16)
            nc = neighbor_count(b, occ, d, int(i)).astype(np.float64)
            delta = (1 - 2 * b) * (-float(d.reward[i]) + nc)
            acc = delta <= 0.0
            up = ~acc
            if np.any(up):
                p = np.exp(-np.minimum(delta[up] / max(temperature, 1e-15), 700.0))
                acc[up] = rng.random(np.sum(up)) < p
            dx = np.where(acc, 1 - 2 * b, 0).astype(np.int16)
            attempted += runs
            accepted += int(np.sum(acc))
            if np.any(dx):
                states[:, i] = (states[:, i].astype(np.int16) + dx).astype(np.int8)
                apply_flip_update(occ, d, int(i), dx)
                assigned += dx * int(d.count[i])
                dist += dx * float(d.distance[i])
                energy += np.where(acc, delta, 0.0)

        completed = sw + 1
        feas = feasible_mask(occ)
        for rr in np.flatnonzero(feas):
            aa, dd = int(assigned[rr]), float(dist[rr])
            better = aa > best_a or (aa == best_a and dd < best_d - 1e-10)
            if better:
                best_a, best_d = aa, dd
                best_bits = states[rr].copy()
            if first_improve < 0 and aa == assigned0 and dd < dist0 - 1e-10:
                first_improve = sw + 1
        if time.perf_counter() - t0 >= max_wall_s:
            break

    wall = time.perf_counter() - t0
    return {
        "bits": best_bits,
        "assigned": best_a,
        "distance": best_d,
        "wall_ms": 1000.0 * wall,
        "sweeps_completed": completed,
        "first_same_card_improvement_sweep": first_improve,
        "same_card_improved": bool(best_a == assigned0 and best_d < dist0 - 1e-10),
        "acceptance_rate": accepted / max(1, attempted),
        "temperature": float(temperature),
    }


def _swap_rows(arr: np.ndarray, eidx: np.ndarray, j: int) -> None:
    if not np.any(eidx):
        return
    tmp = arr[eidx, j].copy()
    arr[eidx, j] = arr[eidx, j + 1]
    arr[eidx, j + 1] = tmp


def run_sparse_pt(
    *, start: np.ndarray, d: ProblemData, nr: int, no: int,
    tmin: float, tmax: float, ntemps: int, ensembles: int,
    sweeps: int, seed: int, max_wall_s: float,
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
    accepted = attempted = swaps_acc = swaps_att = 0
    completed = 0
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
            attempted += ensembles * ntemps
            accepted += int(np.sum(acc))
            if np.any(dx):
                states[:, :, i] = (states[:, :, i].astype(np.int16) + dx).astype(np.int8)
                apply_flip_update(occ, d, int(i), dx)
                assigned += dx * int(d.count[i])
                dist += dx * float(d.distance[i])
                energy += np.where(acc, delta, 0.0)

        # Parallel-tempering swaps.
        parity = sw % 2
        for j in range(parity, ntemps - 1, 2):
            log_ratio = (beta[j] - beta[j + 1]) * (energy[:, j] - energy[:, j + 1])
            swap = np.log(rng.random(ensembles)) < np.minimum(0.0, log_ratio)
            swaps_att += ensembles
            swaps_acc += int(np.sum(swap))
            if np.any(swap):
                _swap_rows(states, swap, j)
                _swap_rows(occ.rider, swap, j)
                _swap_rows(occ.order, swap, j)
                _swap_rows(occ.rider_order, swap, j)
                _swap_rows(occ.order_pair, swap, j)
                _swap_rows(assigned, swap, j)
                _swap_rows(dist, swap, j)
                _swap_rows(energy, swap, j)

        completed = sw + 1
        feas = feasible_mask(occ)
        er, et = np.where(feas)
        for ee, tt in zip(er.tolist(), et.tolist()):
            aa, dd = int(assigned[ee, tt]), float(dist[ee, tt])
            better = aa > best_a or (aa == best_a and dd < best_d - 1e-10)
            if better:
                best_a, best_d = aa, dd
                best_bits = states[ee, tt].copy()
            if first_improve < 0 and aa == a0 and dd < d0 - 1e-10:
                first_improve = sw + 1
        if time.perf_counter() - t0 >= max_wall_s:
            break

    wall = time.perf_counter() - t0
    return {
        "bits": best_bits,
        "assigned": best_a,
        "distance": best_d,
        "wall_ms": 1000.0 * wall,
        "sweeps_completed": completed,
        "first_same_card_improvement_sweep": first_improve,
        "same_card_improved": bool(best_a == a0 and best_d < d0 - 1e-10),
        "acceptance_rate": accepted / max(1, attempted),
        "swap_rate": swaps_acc / max(1, swaps_att),
        "tmin": float(tmin),
        "tmax": float(tmax),
        "ntemps": int(ntemps),
    }


# -----------------------------------------------------------------------------
# Nearest strictly better feasible state (MILP)
# -----------------------------------------------------------------------------

def nearest_better_milp(
    *,
    d: ProblemData, start: np.ndarray,
    scip_assigned: int, scip_distance: float, time_limit_s: float,
) -> dict[str, Any]:
    start_a, start_d, _ = eval_bits(start, d)
    model = Model("nearest_better_full")
    model.hideOutput()
    x = [model.addVar(vtype="B", name=f"x_{i}") for i in range(len(start))]
    for ids in d.by_rider:
        if len(ids) > 0:
            model.addCons(quicksum(x[int(i)] for i in ids) <= 1)
    for ids in d.by_order:
        if len(ids) > 0:
            model.addCons(quicksum(x[int(i)] for i in ids) <= 1)

    assigned_expr = quicksum(float(d.count[i]) * x[i] for i in range(len(x)))
    distance_expr = quicksum(float(d.distance[i]) * x[i] for i in range(len(x)))

    mode = "same_card_lower_distance"
    if scip_assigned > start_a:
        mode = "higher_cardinality"
        model.addCons(assigned_expr >= start_a + 1)
    else:
        model.addCons(assigned_expr == start_a)
        # Require a numerically meaningful strict improvement.
        eps = max(1e-7, 1e-8 * max(1.0, abs(start_d)))
        model.addCons(distance_expr <= start_d - eps)

    selected = np.flatnonzero(start)
    unselected = np.flatnonzero(1 - start)
    hamming = (
        quicksum(1.0 - x[int(i)] for i in selected)
        + quicksum(x[int(i)] for i in unselected)
    )
    model.setObjective(hamming, "minimize")
    model.setRealParam("limits/time", float(time_limit_s))
    t0 = time.perf_counter()
    model.optimize()
    ms = 1000.0 * (time.perf_counter() - t0)
    status = str(model.getStatus())
    bits = np.zeros(len(start), dtype=np.int8)
    if model.getNSols() > 0:
        sol = model.getBestSol()
        for i, var in enumerate(x):
            bits[i] = int(model.getSolVal(sol, var) > 0.5)
        aa, dd, _ = eval_bits(bits, d)
        k = int(np.sum(bits != start))
    else:
        aa, dd, k = -1, math.nan, -1
    try:
        gap = float(model.getGap())
    except Exception:
        gap = math.nan
    return {
        "bits": bits,
        "mode": mode,
        "status": status,
        "optimal": status.lower() == "optimal",
        "gap": gap,
        "k_min": k,
        "target_assigned": aa,
        "target_distance": dd,
        "solve_ms": ms,
    }


# -----------------------------------------------------------------------------
# Controlled subcube: exact barrier + coherent transverse-field test
# -----------------------------------------------------------------------------

def conflict(i: int, j: int, d: ProblemData) -> bool:
    if int(d.rider[i]) == int(d.rider[j]):
        return True
    ai = {int(d.o1[i])}
    if int(d.o2[i]) >= 0:
        ai.add(int(d.o2[i]))
    aj = {int(d.o1[j])}
    if int(d.o2[j]) >= 0:
        aj.add(int(d.o2[j]))
    return bool(ai & aj)


def bits_from_ids(ids: np.ndarray, k: int) -> np.ndarray:
    return ((ids[:, None] >> np.arange(k, dtype=np.uint64)[None, :]) & 1).astype(np.int8)


def bit_index(bits: np.ndarray) -> int:
    idx = 0
    for q, b in enumerate(np.asarray(bits, dtype=np.int8)):
        if int(b):
            idx |= (1 << q)
    return idx


def enumerate_controlled(start: np.ndarray, target: np.ndarray, d: ProblemData, nr: int, no: int) -> dict[str, Any]:
    active = np.flatnonzero(start != target).astype(np.int32)
    k = len(active)
    if k > 20:
        raise ValueError("controlled core too large")
    outside = np.flatnonzero(start & ~np.isin(np.arange(len(start)), active)).astype(np.int32)
    # More reliable outside construction.
    amask = np.zeros(len(start), dtype=bool)
    amask[active] = True
    outside = np.flatnonzero((start > 0) & (~amask)).astype(np.int32)

    dim = 1 << k
    abits = bits_from_ids(np.arange(dim, dtype=np.uint64), k)
    base_reward = float(np.sum(d.reward[outside])) if len(outside) else 0.0
    base_a = int(np.sum(d.count[outside])) if len(outside) else 0
    base_dist = float(np.sum(d.distance[outside])) if len(outside) else 0.0

    outside_conf = np.zeros(k, dtype=np.int16)
    for q, i in enumerate(active):
        outside_conf[q] = sum(1 for j in outside if conflict(int(i), int(j), d))
    A = np.zeros((k, k), dtype=np.int8)
    for q in range(k):
        for p in range(q + 1, k):
            if conflict(int(active[q]), int(active[p]), d):
                A[q, p] = A[p, q] = 1

    penalties = abits @ outside_conf.astype(np.int32)
    penalties = penalties.astype(np.float64) + 0.5 * np.einsum(
        "bi,ij,bj->b", abits.astype(np.float64), A.astype(np.float64), abits.astype(np.float64), optimize=True
    )
    energies = -base_reward - abits @ d.reward[active] + penalties
    assigned = base_a + abits @ d.count[active].astype(np.int32)
    distances = base_dist + abits @ d.distance[active]

    # Feasibility can be checked by resources because outside is feasible.
    bro, boo, br_o, bop = start_occupancy(np.where(amask, 0, start), d, nr, no)
    rmat = np.broadcast_to(bro, (dim, nr)).copy()
    omat = np.broadcast_to(boo, (dim, no)).copy()
    for q, i in enumerate(active):
        vals = abits[:, q]
        rmat[:, int(d.rider[i])] += vals
        omat[:, int(d.o1[i])] += vals
        if int(d.o2[i]) >= 0:
            omat[:, int(d.o2[i])] += vals
    feasible = (np.max(rmat, axis=1) <= 1) & (np.max(omat, axis=1) <= 1)

    sidx = bit_index(start[active])
    tidx = bit_index(target[active])
    return {
        "active": active,
        "k": k,
        "active_bits": abits,
        "energies": energies,
        "assigned": assigned,
        "distances": distances,
        "feasible": feasible,
        "start_idx": sidx,
        "target_idx": tidx,
    }


def minimax_barrier(energies: np.ndarray, start_idx: int, target_idx: int, k: int) -> tuple[float, int]:
    best = np.full(len(energies), np.inf, dtype=np.float64)
    best[start_idx] = energies[start_idx]
    heap: list[tuple[float, int]] = [(float(best[start_idx]), int(start_idx))]
    while heap:
        cost, u = heapq.heappop(heap)
        if cost != best[u]:
            continue
        if u == target_idx:
            break
        for q in range(k):
            v = u ^ (1 << q)
            nc = max(cost, float(energies[v]))
            if nc < best[v] - 1e-14:
                best[v] = nc
                heapq.heappush(heap, (nc, int(v)))
    return float(best[target_idx] - energies[start_idx]), int(np.sum(np.isclose(energies, best[target_idx], atol=1e-12)))


def apply_driver_rotation(psi: np.ndarray, theta: float, k: int) -> np.ndarray:
    out = psi.copy()
    c = math.cos(theta)
    s = 1j * math.sin(theta)
    n = len(out)
    for q in range(k):
        half = 1 << q
        block = half << 1
        view = out.reshape(-1, block)
        a = view[:, :half].copy()
        b = view[:, half:].copy()
        view[:, :half] = c * a + s * b
        view[:, half:] = s * a + c * b
    return out


def reverse_s(u: float, smin: float) -> float:
    if u <= 0.5:
        return 1.0 - 2.0 * u * (1.0 - smin)
    return smin + 2.0 * (u - 0.5) * (1.0 - smin)


def reverse_quantum(cube: dict[str, Any], smin: float, total_time: float, steps: int) -> dict[str, Any]:
    E = cube["energies"]
    feas = cube["feasible"]
    sidx, tidx, k = int(cube["start_idx"]), int(cube["target_idx"]), int(cube["k"])
    problem = E - float(E[sidx])
    psi = np.zeros(len(E), dtype=np.complex128)
    psi[sidx] = 1.0
    dt = total_time / steps
    t0 = time.perf_counter()
    for step in range(steps):
        s = reverse_s((step + 0.5) / steps, smin)
        driver = 1.0 - s
        psi = apply_driver_rotation(psi, 0.5 * dt * driver, k)
        psi *= np.exp(-1j * dt * s * problem)
        psi = apply_driver_rotation(psi, 0.5 * dt * driver, k)
    p = np.abs(psi) ** 2
    p /= np.sum(p)
    lower_feasible = feas & (E < E[sidx] - 1e-12)
    return {
        "smin": smin,
        "total_time": total_time,
        "steps": steps,
        "p_target": float(p[tidx]),
        "p_lower_feasible": float(np.sum(p[lower_feasible])),
        "p_feasible": float(np.sum(p[feas])),
        "p_start": float(p[sidx]),
        "wall_ms": 1000.0 * (time.perf_counter() - t0),
    }




# -----------------------------------------------------------------------------
# Quality-fraction geometry, controlled thermal, and coherent holds
# -----------------------------------------------------------------------------

def overlap_stats(
    d: ProblemData,
) -> dict[str, float]:
    m = max(1, d.num_candidates)

    rb = np.asarray(
        [len(x) for x in d.by_rider],
        dtype=np.float64,
    )

    ob = np.asarray(
        [len(x) for x in d.by_order],
        dtype=np.float64,
    )
    pair_proxy = float(np.sum(rb * (rb - 1.0) / 2.0) + np.sum(ob * (ob - 1.0) / 2.0))
    denom = m * (m - 1.0) / 2.0
    return {
        "rider_bucket_mean": float(np.mean(rb)) if len(rb) else 0.0,
        "rider_bucket_max": float(np.max(rb)) if len(rb) else 0.0,
        "order_bucket_mean": float(np.mean(ob)) if len(ob) else 0.0,
        "order_bucket_min": float(np.min(ob)) if len(ob) else 0.0,
        "order_bucket_max": float(np.max(ob)) if len(ob) else 0.0,
        "orders_degree_le_5": int(np.sum(ob <= 5)),
        "orders_degree_le_10": int(np.sum(ob <= 10)),
        "overlap_pair_proxy": pair_proxy,
        "overlap_density_proxy": float(pair_proxy / denom) if denom > 0 else 0.0,
        "pair_fraction": float(np.mean(d.count == 2)) if len(d.count) else 0.0,
    }


def nearest_quality_fraction_milp(
    *, d: ProblemData, start: np.ndarray,
    optimum_assigned: int, optimum_distance: float, fraction: float,
    time_limit_s: float, seed: int,
) -> dict[str, Any]:
    """Minimum Hamming distance to a feasible state achieving a fraction of the
    greedy->SCIP quality improvement. For same-cardinality refinement this means
    a fraction of the distance reduction. If SCIP has higher cardinality, this
    asks for the higher-cardinality basin instead.
    """
    start_a, start_d, _ = eval_bits(start, d)
    model = Model(f"nearest_quality_{fraction:g}")
    model.hideOutput()
    try:
        model.setIntParam("randomization/randomseedshift", int(seed) % 2147483647)
    except Exception:
        pass
    x = [model.addVar(vtype="B", name=f"x_{i}") for i in range(len(start))]
    for ids in d.by_rider:
        if len(ids) > 0:
            model.addCons(quicksum(x[int(i)] for i in ids) <= 1)
    for ids in d.by_order:
        if len(ids) > 0:
            model.addCons(quicksum(x[int(i)] for i in ids) <= 1)
    assigned_expr = quicksum(float(d.count[i]) * x[i] for i in range(len(x)))
    distance_expr = quicksum(float(d.distance[i]) * x[i] for i in range(len(x)))

    if optimum_assigned > start_a:
        mode = "higher_cardinality"
        model.addCons(assigned_expr >= start_a + 1)
        required_gain = math.nan
        target_distance_threshold = math.nan
    else:
        mode = "same_card_distance_fraction"
        model.addCons(assigned_expr == start_a)
        full_gain = max(0.0, float(start_d - optimum_distance))
        required_gain = float(fraction * full_gain)
        target_distance_threshold = float(start_d - required_gain)
        tol = max(1e-9, 1e-10 * max(1.0, abs(start_d)))
        model.addCons(distance_expr <= target_distance_threshold + tol)

    selected = np.flatnonzero(start)
    unselected = np.flatnonzero(1 - start)
    hamming = quicksum(1.0 - x[int(i)] for i in selected) + quicksum(x[int(i)] for i in unselected)
    model.setObjective(hamming, "minimize")
    model.setRealParam("limits/time", float(time_limit_s))
    t0 = time.perf_counter()
    model.optimize()
    ms = 1000.0 * (time.perf_counter() - t0)
    status = str(model.getStatus())
    bits = np.zeros(len(start), dtype=np.int8)
    if model.getNSols() > 0:
        sol = model.getBestSol()
        for i, var in enumerate(x):
            bits[i] = int(model.getSolVal(sol, var) > 0.5)
        aa, dd, _ = eval_bits(bits, d)
        k = int(np.sum(bits != start))
        achieved = float(start_d - dd) if aa == start_a else math.nan
    else:
        aa, dd, k, achieved = -1, math.nan, -1, math.nan
    try:
        gap = float(model.getGap())
    except Exception:
        gap = math.nan
    return {
        "bits": bits,
        "fraction": float(fraction),
        "mode": mode,
        "status": status,
        "optimal": status.lower() == "optimal",
        "gap": gap,
        "k_min": k,
        "target_assigned": aa,
        "target_distance": dd,
        "required_distance_gain": required_gain,
        "achieved_distance_gain": achieved,
        "distance_threshold": target_distance_threshold,
        "solve_ms": ms,
    }


def controlled_sa(
    cube: dict[str, Any], *, target_energy: float, temperature: float,
    runs: int, sweeps: int, seed: int,
) -> dict[str, Any]:
    E = cube["energies"]
    feas = cube["feasible"]
    sidx = int(cube["start_idx"])
    k = int(cube["k"])
    rng = np.random.default_rng(seed)
    states = np.full(runs, sidx, dtype=np.int64)
    ever = np.zeros(runs, dtype=bool)
    ever_target = np.zeros(runs, dtype=bool)
    tidx = int(cube["target_idx"])
    first = np.full(runs, -1, dtype=np.int32)
    t0 = time.perf_counter()
    for sw in range(sweeps):
        for q in rng.permutation(k):
            prop = states ^ (1 << int(q))
            de = E[prop] - E[states]
            acc = de <= 0
            up = ~acc
            if np.any(up):
                p = np.exp(-np.minimum(de[up] / max(float(temperature), 1e-15), 700.0))
                acc[up] = rng.random(np.sum(up)) < p
            states[acc] = prop[acc]
        good = feas[states] & (E[states] <= target_energy + 1e-12)
        new = good & (~ever)
        first[new] = sw + 1
        ever |= good
        ever_target |= states == tidx
    return {
        "solver": "controlled_sa",
        "temperature": float(temperature),
        "runs": int(runs),
        "sweeps": int(sweeps),
        "quality_success_rate": float(np.mean(ever)),
        "exact_target_hit_rate": float(np.mean(ever_target)),
        "median_first_hit_sweep": float(np.median(first[first > 0])) if np.any(first > 0) else math.nan,
        "wall_ms": 1000.0 * (time.perf_counter() - t0),
    }


def controlled_pt(
    cube: dict[str, Any], *, target_energy: float,
    tmin: float, tmax: float, ntemps: int, ensembles: int,
    sweeps: int, seed: int,
) -> dict[str, Any]:
    E = cube["energies"]
    feas = cube["feasible"]
    sidx = int(cube["start_idx"])
    tidx = int(cube["target_idx"])
    k = int(cube["k"])
    rng = np.random.default_rng(seed)
    temps = np.geomspace(max(tmin, 1e-10), max(tmax, tmin * 1.001), ntemps)
    beta = 1.0 / temps
    states = np.full((ensembles, ntemps), sidx, dtype=np.int64)
    ever = np.zeros(ensembles, dtype=bool)
    ever_target = np.zeros(ensembles, dtype=bool)
    first = np.full(ensembles, -1, dtype=np.int32)
    swaps_a = swaps_n = 0
    t0 = time.perf_counter()
    for sw in range(sweeps):
        for q in rng.permutation(k):
            prop = states ^ (1 << int(q))
            de = E[prop] - E[states]
            acc = de <= 0
            up = ~acc
            if np.any(up):
                T = np.broadcast_to(temps[None, :], states.shape)
                p = np.exp(-np.minimum(de[up] / T[up], 700.0))
                acc[up] = rng.random(np.sum(up)) < p
            states[acc] = prop[acc]
        for parity in (0, 1):
            for j in range(parity, ntemps - 1, 2):
                loga = (beta[j] - beta[j + 1]) * (E[states[:, j]] - E[states[:, j + 1]])
                acc = np.log(np.maximum(rng.random(ensembles), 1e-300)) < np.minimum(0.0, loga)
                swaps_n += ensembles
                swaps_a += int(np.sum(acc))
                if np.any(acc):
                    tmp = states[acc, j].copy()
                    states[acc, j] = states[acc, j + 1]
                    states[acc, j + 1] = tmp
        # Success for an ensemble if any replica has reached target quality.
        good = np.any(feas[states] & (E[states] <= target_energy + 1e-12), axis=1)
        targ = np.any(states == tidx, axis=1)
        new = good & (~ever)
        first[new] = sw + 1
        ever |= good
        ever_target |= targ
    return {
        "solver": "controlled_pt",
        "tmin": float(tmin), "tmax": float(tmax), "ntemps": int(ntemps),
        "ensembles": int(ensembles), "sweeps": int(sweeps),
        "quality_success_rate": float(np.mean(ever)),
        "exact_target_hit_rate": float(np.mean(ever_target)),
        "median_first_hit_sweep": float(np.median(first[first > 0])) if np.any(first > 0) else math.nan,
        "swap_rate": swaps_a / max(1, swaps_n),
        "wall_ms": 1000.0 * (time.perf_counter() - t0),
    }


def reverse_quantum_quality(cube: dict[str, Any], *, target_energy: float, smin: float, total_time: float, steps: int) -> dict[str, Any]:
    E = cube["energies"]
    feas = cube["feasible"]
    sidx, tidx, k = int(cube["start_idx"]), int(cube["target_idx"]), int(cube["k"])
    problem = E - float(E[sidx])
    psi = np.zeros(len(E), dtype=np.complex128)
    psi[sidx] = 1.0
    dt = float(total_time) / int(steps)
    t0 = time.perf_counter()
    for step in range(int(steps)):
        s = reverse_s((step + 0.5) / steps, float(smin))
        driver = 1.0 - s
        psi = apply_driver_rotation(psi, 0.5 * dt * driver, k)
        psi *= np.exp(-1j * dt * s * problem)
        psi = apply_driver_rotation(psi, 0.5 * dt * driver, k)
    p = np.abs(psi) ** 2
    p /= np.sum(p)
    lower = feas & (E < E[sidx] - 1e-12)
    targetq = feas & (E <= float(target_energy) + 1e-12)
    return {
        "smin": float(smin), "total_time": float(total_time), "steps": int(steps),
        "p_target": float(p[tidx]),
        "p_target_quality_or_better": float(np.sum(p[targetq])),
        "p_any_lower_feasible": float(np.sum(p[lower])),
        "p_feasible": float(np.sum(p[feas])),
        "p_start": float(p[sidx]),
        "wall_ms": 1000.0 * (time.perf_counter() - t0),
    }


def _evolve_fixed(psi: np.ndarray, problem: np.ndarray, k: int, s: float, duration: float, steps: int) -> np.ndarray:
    if duration <= 0:
        return psi
    dt = duration / max(1, steps)
    for _ in range(max(1, steps)):
        drv = 1.0 - s
        psi = apply_driver_rotation(psi, 0.5 * dt * drv, k)
        psi *= np.exp(-1j * dt * s * problem)
        psi = apply_driver_rotation(psi, 0.5 * dt * drv, k)
    return psi


def reverse_hold_quality(
    cube: dict[str, Any], *, target_energy: float, s_hold: float,
    ramp_time: float, hold_time: float, ramp_steps: int = 80,
) -> dict[str, Any]:
    E = cube["energies"]
    feas = cube["feasible"]
    sidx, tidx, k = int(cube["start_idx"]), int(cube["target_idx"]), int(cube["k"])
    problem = E - float(E[sidx])
    psi = np.zeros(len(E), dtype=np.complex128)
    psi[sidx] = 1.0
    # Linear down ramp.
    dt = ramp_time / max(1, ramp_steps)
    for step in range(max(1, ramp_steps)):
        u = (step + 0.5) / max(1, ramp_steps)
        s = 1.0 - u * (1.0 - s_hold)
        psi = apply_driver_rotation(psi, 0.5 * dt * (1.0 - s), k)
        psi *= np.exp(-1j * dt * s * problem)
        psi = apply_driver_rotation(psi, 0.5 * dt * (1.0 - s), k)
    hold_steps = max(1, int(math.ceil(20.0 * hold_time))) if hold_time > 0 else 1
    psi = _evolve_fixed(psi, problem, k, s_hold, hold_time, hold_steps)
    # Linear up ramp.
    for step in range(max(1, ramp_steps)):
        u = (step + 0.5) / max(1, ramp_steps)
        s = s_hold + u * (1.0 - s_hold)
        psi = apply_driver_rotation(psi, 0.5 * dt * (1.0 - s), k)
        psi *= np.exp(-1j * dt * s * problem)
        psi = apply_driver_rotation(psi, 0.5 * dt * (1.0 - s), k)
    p = np.abs(psi) ** 2
    p /= np.sum(p)
    targetq = feas & (E <= target_energy + 1e-12)
    return {
        "s_hold": float(s_hold), "ramp_time": float(ramp_time), "hold_time": float(hold_time),
        "p_target": float(p[tidx]),
        "p_target_quality_or_better": float(np.sum(p[targetq])),
        "p_feasible": float(np.sum(p[feas])),
    }


def core_local_minima_stats(cube: dict[str, Any]) -> dict[str, Any]:
    E = cube["energies"]
    feas = cube["feasible"]
    k = int(cube["k"])
    n = len(E)
    local = np.ones(n, dtype=bool)
    for q in range(k):
        nbr = np.arange(n, dtype=np.int64) ^ (1 << q)
        local &= E <= E[nbr] + 1e-12
    sE = float(E[int(cube["start_idx"])])
    return {
        "num_local_minima_all": int(np.sum(local)),
        "num_feasible_local_minima": int(np.sum(local & feas)),
        "num_lower_feasible_states": int(np.sum(feas & (E < sE - 1e-12))),
        "feasible_fraction": float(np.mean(feas)),
    }


# -----------------------------------------------------------------------------
# Main — adaptive 90-minute hardness hunt
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
            "--sets",
            default="landscape_15x40,landscape_20x50",
        )
    ap.add_argument(
        "--output",
        default="landscape_hardness/results",
    )
    ap.add_argument("--deadline-minutes", type=float, default=82.0)
    ap.add_argument("--screen-budget-minutes", type=float, default=18.0)
    ap.add_argument("--scip-seconds", type=float, default=6.0)
    ap.add_argument("--selected-per-scale", type=int, default=5)
    ap.add_argument("--nearest-seconds", type=float, default=25.0)
    ap.add_argument("--thermal-max-wall", type=float, default=45.0)
    ap.add_argument("--sa-runs", type=int, default=8)
    ap.add_argument("--sa-sweeps", type=int, default=32)
    ap.add_argument("--pt-ensembles", type=int, default=3)
    ap.add_argument("--pt-sweeps", type=int, default=24)
    ap.add_argument("--barrier-max-k", type=int, default=16)
    ap.add_argument("--quantum-max-k", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1800000)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t_global = time.perf_counter()
    deadline = t_global + 60.0 * args.deadline_minutes
    screen_deadline = min(deadline, t_global + 60.0 * args.screen_budget_minutes)

    def left() -> float:
        return deadline - time.perf_counter()

    def can_start(estimated_s: float = 5.0) -> bool:
        return left() > estimated_s + 20.0

    config = vars(args).copy()
    config["started_unix"] = time.time()
    write_json(out / "run_config.json", config)

    screen_rows: list[dict[str, Any]] = []
    nearest_rows: list[dict[str, Any]] = []
    thermal_rows: list[dict[str, Any]] = []
    barrier_rows: list[dict[str, Any]] = []
    controlled_classical_rows: list[dict[str, Any]] = []
    quantum_rows: list[dict[str, Any]] = []
    hold_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    set_names = [s.strip() for s in args.sets.split(",") if s.strip()]
    loaded: dict[str, list[ProblemData]] = {
        s: load_problem_set(s)
        for s in set_names
    }

    print("=" * 104)
    print("EulerQ — Flux landscape hardness analysis")
    print(f"Internal deadline: {args.deadline_minutes:.1f} min | screen budget: {args.screen_budget_minutes:.1f} min")
    print("Full 2-order Flux only; no dense global QUBO; all files checkpoint after every result.")
    print("=" * 104)

    # ------------------------------------------------------------------
    # Phase A: broad full-problem screen.
    # ------------------------------------------------------------------
    for set_name in set_names:
        print(f"\nSCREEN {set_name}: {len(loaded[set_name])} instances")
        for idx, d in enumerate(loaded[set_name]):
            if (
                time.perf_counter() >= screen_deadline
                or not can_start(args.scip_seconds + 5)
            ):
                print(
                    "Screen budget reached; preserving time "
                    "for deep analysis."
                )
                break

            tg = time.perf_counter()
            gbits = greedy_pair_first(d)
            greedy_ms = 1000.0 * (
                time.perf_counter() - tg
            )

            ga, gd, _ = eval_bits(gbits, d)

            sc = run_scip(
                d,
                args.scip_seconds,
                args.seed
                + 1000 * len(screen_rows)
                + idx,
            )
            same = sc["assigned"] == ga
            dgain = float(gd - sc["distance"]) if same and sc["assigned"] >= 0 else math.nan
            gap_pct = 100.0 * dgain / max(abs(sc["distance"]), 1e-12) if same else math.nan
            e_g = -float(gbits @ d.reward)
            e_s = -float(sc["bits"] @ d.reward) if sc["assigned"] >= 0 else math.nan
            egain = float(e_g - e_s) if np.isfinite(e_s) else math.nan
            kopt = int(np.sum(gbits != sc["bits"])) if sc["assigned"] >= 0 else -1
            st = overlap_stats(d)
            # A ranking score, not a hardness claim: prefer real greedy gaps,
            # wider rearrangements, more B&B work, and slower exact solves.
            hardness_score = (
                (max(0.0, gap_pct) if np.isfinite(gap_pct) else 0.0) * 3.0
                + math.log1p(max(0, sc["nodes"]))
                + 0.5 * math.log1p(max(0.0, sc["solver_ms"]))
                + 0.04 * max(0, kopt)
            )
            row = {
                "set_name": set_name, "instance_index": idx, "instance_id": d.instance_id,
                "num_riders": int(d.num_riders), "num_orders": int(d.num_orders),
                "num_candidates": d.num_candidates,
                "single_candidates": int(np.sum(d.count == 1)), "pair_candidates": int(np.sum(d.count == 2)),
                "greedy_ms": greedy_ms,
                "greedy_assigned": ga, "greedy_distance": gd,
                "scip_status": sc["status"], "scip_optimal": sc["optimal"], "scip_gap": sc["gap"],
                "scip_nodes": sc["nodes"], "scip_assigned": sc["assigned"], "scip_distance": sc["distance"],
                "scip_optimize_ms": sc["optimize_ms"], "scip_solver_ms": sc["solver_ms"],
                "same_cardinality": same, "greedy_minus_scip_distance": dgain,
                "greedy_distance_gap_pct": gap_pct,
                "greedy_to_scip_energy_gain_norm": egain, "greedy_to_scip_hamming": kopt,
                "lambda_raw": d.lam, "hardness_rank_score": hardness_score,
                **st,
            }
            screen_rows.append(row)
            write_csv(out / "screening.csv", screen_rows)
            print(
                f"[{set_name} {idx:02d}] m={d.num_candidates:6d} "
                f"G={ga:2d}/{gd:8.2f} S={sc['assigned']:2d}/{sc['distance']:8.2f} "
                f"dD={dgain:7.3f} k={kopt:3d} nodes={sc['nodes']:6d} SCIP={sc['solver_ms']:7.1f}ms"
            )
        if time.perf_counter() >= screen_deadline:
            break

    # Select hard same-cardinality refinement cases per scale.
    selected: list[tuple[str, int]] = []
    for set_name in set_names:
        eligible = [
            r for r in screen_rows
            if r["set_name"] == set_name and r["same_cardinality"]
            and np.isfinite(float(r["greedy_minus_scip_distance"]))
            and float(r["greedy_minus_scip_distance"]) > 1e-7
        ]
        eligible.sort(key=lambda r: float(r["hardness_rank_score"]), reverse=True)
        selected.extend((set_name, int(r["instance_index"])) for r in eligible[: args.selected_per_scale])
    write_json(out / "selected.json", {"selected": selected})
    print("\nSELECTED FOR DEEP ANALYSIS:", selected)

    # ------------------------------------------------------------------
    # Phase B: exact quality-width geometry + robust full PT/SA.
    # ------------------------------------------------------------------
    core_cache: list[dict[str, Any]] = []
    for sel_rank, (set_name, idx) in enumerate(selected):
        if not can_start(60):
            break
        d = loaded[set_name][idx]
        gbits = greedy_pair_first(d)
        ga, gd, _ = eval_bits(gbits, d)
        # Recompute SCIP with the normal screen limit so the deep record is self-contained.
        sc = run_scip(d, args.scip_seconds, args.seed + 50000 + sel_rank)
        if sc["assigned"] != ga or not np.isfinite(sc["distance"]) or gd <= sc["distance"] + 1e-8:
            continue
        full_gain_d = float(gd - sc["distance"])
        energy_gain = float((-gbits @ d.reward) - (-sc["bits"] @ d.reward))
        print(f"\nDEEP #{sel_rank} {set_name}/{idx}: m={len(gbits)} ΔD={full_gain_d:.6f} ΔE={energy_gain:.6g} left={left()/60:.1f}m")

        frac_results: list[dict[str, Any]] = []
        for fi, frac in enumerate((0.10, 0.50, 0.90)):
            if not can_start(args.nearest_seconds + 10):
                break
            nb = nearest_quality_fraction_milp(
                d=d, start=gbits, optimum_assigned=int(sc["assigned"]),
                optimum_distance=float(sc["distance"]), fraction=frac,
                time_limit_s=min(args.nearest_seconds, max(2.0, left() - 20.0)),
                seed=args.seed + 60000 + 100 * sel_rank + fi,
            )
            frac_results.append(nb)
            nearest_rows.append({
                "set_name": set_name, "instance_index": idx, "instance_id": d.instance_id,
                "num_candidates": len(gbits), "start_assigned": ga, "start_distance": gd,
                "scip_distance": sc["distance"], "full_distance_gain": full_gain_d,
                "scip_hamming": int(np.sum(gbits != sc["bits"])),
                **{k: v for k, v in nb.items() if k != "bits"},
            })
            write_csv(out / "nearest_quality_widths.csv", nearest_rows)
            print(f"  width f={frac:.1f}: k={nb['k_min']} status={nb['status']} D={nb['target_distance']:.5f}")

        # Full thermal tests: three ladders, two independent seeds when time permits.
        cal_tmin = max(1e-6, energy_gain / 20.0)
        thermal_cfgs = [
            ("sa_gap_div10", "sa", max(1e-6, energy_gain / 10.0), None, None),
            ("sa_0.1", "sa", 0.1, None, None),
            ("pt_resolving", "pt", cal_tmin, 1.2, 20),
            ("pt_wide", "pt", cal_tmin, 3.0, 24),
            ("pt_ultracold", "pt", max(1e-7, energy_gain / 50.0), 1.2, 24),
        ]
        for ci, (tag, typ, p1, p2, p3) in enumerate(thermal_cfgs):
            for rep in range(2 if typ == "pt" else 1):
                if not can_start(args.thermal_max_wall + 20):
                    break
                seed = args.seed + 70000 + 1000 * sel_rank + 20 * ci + rep
                if typ == "sa":
                    rr = run_sparse_sa(
                        start=gbits, d=d, nr=int(d.num_riders), no=int(d.num_orders),
                        temperature=float(p1), runs=args.sa_runs, sweeps=args.sa_sweeps,
                        seed=seed, max_wall_s=min(args.thermal_max_wall, max(5.0, left() - 20.0)),
                    )
                else:
                    rr = run_sparse_pt(
                        start=gbits, d=d, nr=int(d.num_riders), no=int(d.num_orders),
                        tmin=float(p1), tmax=float(p2), ntemps=int(p3), ensembles=args.pt_ensembles,
                        sweeps=args.pt_sweeps, seed=seed,
                        max_wall_s=min(args.thermal_max_wall, max(5.0, left() - 20.0)),
                    )
                frac_ach = (
                    max(0.0, min(1.5, (gd - float(rr["distance"])) / max(full_gain_d, 1e-12)))
                    if int(rr["assigned"]) == ga else math.nan
                )
                thermal_rows.append({
                    "set_name": set_name, "instance_index": idx, "instance_id": d.instance_id,
                    "solver": tag, "rep": rep, "num_candidates": len(gbits),
                    "start_assigned": ga, "start_distance": gd, "scip_distance": sc["distance"],
                    "full_distance_gain": full_gain_d, "energy_gain_norm": energy_gain,
                    "fraction_of_scip_improvement_achieved": frac_ach,
                    **{k: v for k, v in rr.items() if k != "bits"},
                })
                write_csv(out / "full_thermal_runs.csv", thermal_rows)
                print(f"  {tag:14s} rep{rep}: frac={frac_ach:.3f} D={rr['distance']:.4f} wall={rr['wall_ms']/1000:.1f}s")

        # Pick the highest quality fraction whose exact nearest target is small enough.
        eligible_core = [x for x in frac_results if x["optimal"] and 0 < int(x["k_min"]) <= args.barrier_max_k]
        if eligible_core:
            # Prefer the highest substantial target that still fits exact coherent
            # simulation.  If none fits, retain the highest target that fits exact
            # barrier enumeration.  This avoids skipping quantum analysis merely
            # because the 90% target is a little wider than the statevector cap.
            quantum_fit = [x for x in eligible_core if int(x["k_min"]) <= args.quantum_max_k]
            target_nb = sorted(quantum_fit if quantum_fit else eligible_core, key=lambda x: float(x["fraction"]), reverse=True)[0]
            cube = enumerate_controlled(gbits, target_nb["bits"], d, int(d.num_riders), int(d.num_orders))
            barrier, topdeg = minimax_barrier(cube["energies"], cube["start_idx"], cube["target_idx"], cube["k"])
            cstats = core_local_minima_stats(cube)
            target_energy = float(cube["energies"][cube["target_idx"]])
            gain = float(cube["energies"][cube["start_idx"]] - target_energy)
            br = {
                "set_name": set_name, "instance_index": idx, "instance_id": d.instance_id,
                "quality_fraction": target_nb["fraction"], "k": cube["k"],
                "num_states": len(cube["energies"]), "num_feasible_states": int(np.sum(cube["feasible"])),
                "start_energy": float(cube["energies"][cube["start_idx"]]), "target_energy": target_energy,
                "target_energy_gain": gain, "minimax_barrier_norm": barrier,
                "barrier_over_target_gain": barrier / max(gain, 1e-15),
                "barrier_top_degeneracy": topdeg, **cstats,
            }
            barrier_rows.append(br)
            write_csv(out / "controlled_barriers.csv", barrier_rows)
            print(f"  CORE f={target_nb['fraction']:.1f} k={cube['k']} B={barrier:.6g} B/gain={br['barrier_over_target_gain']:.1f}")
            core_cache.append({
                "set_name": set_name, "idx": idx, "d": d,
                "gbits": gbits, "sc": sc, "cube": cube, "barrier": barrier,
                "target_energy": target_energy, "quality_fraction": target_nb["fraction"],
                "full_gain_d": full_gain_d, "energy_gain": energy_gain,
            })

            # Classical controlled-core effort curve on exactly the same landscape.
            c_tmin = max(1e-5, gain / 20.0)
            c_tmax = max(1.2, min(4.0, 1.5 * max(barrier, 0.1)))
            for sweeps in (32, 128, 512):
                if not can_start(3):
                    break
                for tag, res in [
                    ("sa_low", controlled_sa(cube, target_energy=target_energy, temperature=max(1e-4, gain/10.0), runs=256, sweeps=sweeps, seed=args.seed+80000+sel_rank*100+sweeps)),
                    ("sa_mid", controlled_sa(cube, target_energy=target_energy, temperature=0.1, runs=256, sweeps=sweeps, seed=args.seed+81000+sel_rank*100+sweeps)),
                    ("pt_cal", controlled_pt(cube, target_energy=target_energy, tmin=c_tmin, tmax=c_tmax, ntemps=16, ensembles=128, sweeps=sweeps, seed=args.seed+82000+sel_rank*100+sweeps)),
                ]:
                    controlled_classical_rows.append({
                        "set_name": set_name, "instance_index": idx, "instance_id": d.instance_id,
                        "quality_fraction": target_nb["fraction"], "k": cube["k"],
                        "barrier_norm": barrier, "target_gain": gain, "variant": tag, **res,
                    })
                write_csv(out / "controlled_classical.csv", controlled_classical_rows)

            # Coherent mechanism test if exact statevector is still small.
            if cube["k"] <= args.quantum_max_k and can_start(10):
                best_q = None
                for smin in (0.15, 0.25, 0.35, 0.50):
                    for tt in (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0):
                        if not can_start(2):
                            break
                        steps = max(120, int(32 * tt))
                        qr = reverse_quantum_quality(cube, target_energy=target_energy, smin=smin, total_time=tt, steps=steps)
                        rowq = {
                            "set_name": set_name, "instance_index": idx, "instance_id": d.instance_id,
                            "quality_fraction": target_nb["fraction"], "k": cube["k"],
                            "barrier_norm": barrier, "target_gain": gain, **qr,
                        }
                        quantum_rows.append(rowq)
                        if best_q is None or qr["p_target_quality_or_better"] > best_q["p_target_quality_or_better"]:
                            best_q = qr
                        write_csv(out / "controlled_quantum_time_sweep.csv", quantum_rows)
                if best_q is not None and can_start(10):
                    # Hold spectroscopy: phase-sensitive oscillation is a coherence diagnostic,
                    # not a computational-advantage claim.
                    for tau in np.linspace(0.0, 15.0, 31):
                        if not can_start(1):
                            break
                        hr = reverse_hold_quality(
                            cube, target_energy=target_energy, s_hold=float(best_q["smin"]),
                            ramp_time=max(1.0, float(best_q["total_time"]) / 2.0), hold_time=float(tau), ramp_steps=80,
                        )
                        hold_rows.append({
                            "set_name": set_name, "instance_index": idx, "instance_id": d.instance_id,
                            "quality_fraction": target_nb["fraction"], "k": cube["k"],
                            "barrier_norm": barrier, "target_gain": gain, **hr,
                        })
                    write_csv(out / "controlled_quantum_hold.csv", hold_rows)

    # ------------------------------------------------------------------
    # Phase C: use remaining time to stress the most thermally resistant
    # full instances with independent calibrated PT seeds.
    # ------------------------------------------------------------------
    def current_pt_fraction(set_name: str, idx: int) -> float:
        vals = [float(r["fraction_of_scip_improvement_achieved"]) for r in thermal_rows
                if r["set_name"] == set_name and int(r["instance_index"]) == idx
                and str(r["solver"]).startswith("pt_")
                and np.isfinite(float(r["fraction_of_scip_improvement_achieved"]))]
        return max(vals) if vals else 0.0

    revisit = []
    for set_name, idx in selected:
        sr = next((r for r in screen_rows if r["set_name"] == set_name and int(r["instance_index"]) == idx), None)
        if sr is None:
            continue
        revisit.append((current_pt_fraction(set_name, idx), -float(sr["hardness_rank_score"]), set_name, idx))
    revisit.sort()
    rep_counter = 0
    while can_start(args.thermal_max_wall + 25) and revisit:
        _, _, set_name, idx = revisit[rep_counter % len(revisit)]
        d = loaded[set_name][idx]
        gbits = greedy_pair_first(d)
        ga, gd, _ = eval_bits(gbits, d)
        sc = run_scip(d, min(args.scip_seconds, 4.0), args.seed + 90000 + rep_counter)
        if sc["assigned"] != ga or gd <= sc["distance"] + 1e-8:
            rep_counter += 1
            continue
        full_gain_d = gd - sc["distance"]
        egain = float((-gbits @ d.reward) - (-sc["bits"] @ d.reward))
        rr = run_sparse_pt(
            start=gbits, d=d, nr=int(d.num_riders), no=int(d.num_orders),
            tmin=max(1e-7, egain / 50.0), tmax=3.0, ntemps=28, ensembles=max(3, args.pt_ensembles),
            sweeps=max(32, args.pt_sweeps), seed=args.seed + 100000 + rep_counter,
            max_wall_s=min(args.thermal_max_wall, max(5.0, left() - 20.0)),
        )
        frac_ach = max(0.0, min(1.5, (gd - float(rr["distance"])) / max(full_gain_d, 1e-12))) if rr["assigned"] == ga else math.nan
        thermal_rows.append({
            "set_name": set_name, "instance_index": idx, "instance_id": d.instance_id,
            "solver": "pt_revisit_wide28", "rep": rep_counter, "num_candidates": len(gbits),
            "start_assigned": ga, "start_distance": gd, "scip_distance": sc["distance"],
            "full_distance_gain": full_gain_d, "energy_gain_norm": egain,
            "fraction_of_scip_improvement_achieved": frac_ach,
            **{k: v for k, v in rr.items() if k != "bits"},
        })
        write_csv(out / "full_thermal_runs.csv", thermal_rows)
        print(f"REVISIT {set_name}/{idx} rep{rep_counter}: PT fraction={frac_ach:.3f}, left={left()/60:.1f}m")
        rep_counter += 1

    # Candidate quantum-mechanism table: this is a ranking aid, not a claim.
    for br in barrier_rows:
        set_name = br["set_name"]; idx = int(br["instance_index"])
        ptvals = [float(r["fraction_of_scip_improvement_achieved"]) for r in thermal_rows
                  if r["set_name"] == set_name and int(r["instance_index"]) == idx
                  and str(r["solver"]).startswith("pt_") and np.isfinite(float(r["fraction_of_scip_improvement_achieved"]))]
        qvals = [float(r["p_target_quality_or_better"]) for r in quantum_rows
                 if r["set_name"] == set_name and int(r["instance_index"]) == idx]
        hvals = [float(r["p_target_quality_or_better"]) for r in hold_rows
                 if r["set_name"] == set_name and int(r["instance_index"]) == idx]
        candidate_rows.append({
            "set_name": set_name, "instance_index": idx, "instance_id": br["instance_id"],
            "quality_fraction": br["quality_fraction"], "k": br["k"],
            "barrier_norm": br["minimax_barrier_norm"], "barrier_over_target_gain": br["barrier_over_target_gain"],
            "best_full_pt_fraction_of_scip_gain": max(ptvals) if ptvals else math.nan,
            "best_quantum_quality_probability": max(qvals) if qvals else math.nan,
            "quantum_hold_probability_range": (max(hvals) - min(hvals)) if hvals else math.nan,
            "mechanism_candidate": bool(ptvals and max(ptvals) < 0.10 and qvals and max(qvals) > 0.10),
        })
    write_csv(out / "mechanism_candidates.csv", candidate_rows)

    summary = {
        "elapsed_minutes": (time.perf_counter() - t_global) / 60.0,
        "screened": len(screen_rows), "selected": selected,
        "nearest_width_rows": len(nearest_rows), "full_thermal_rows": len(thermal_rows),
        "controlled_barriers": len(barrier_rows), "controlled_classical_rows": len(controlled_classical_rows),
        "quantum_time_rows": len(quantum_rows), "quantum_hold_rows": len(hold_rows),
        "mechanism_candidates": int(sum(bool(r.get("mechanism_candidate")) for r in candidate_rows)),
        "interpretation_note": (
            "A mechanism_candidate means calibrated full PT failed to obtain 10% of the SCIP refinement while ideal "
            "coherent dynamics on a classically extracted controlled core put >10% probability on target-quality states. "
            "This is mechanistic evidence only, not quantum advantage."
        ),
    }
    write_json(out / "summary.json", summary)
    print("\nDONE", json.dumps(summary, indent=2))
    print("Results ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
