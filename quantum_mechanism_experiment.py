from __future__ import annotations

"""
EulerQ — Quantum Mechanism Evidence on Flux-Derived Subproblems
===============================================================

Purpose
-------
This script evaluates quantum-specific mechanisms on Flux-derived optimization
subproblems. The experiments are designed to test whether coherent quantum
dynamics can support useful transitions between competing Flux configurations
and to distinguish these effects from purely classical or incoherent behavior.

The experiments focus on the following questions:

1. COHERENCE CONTROL
   Re-run the same Flux-derived transition and reverse-annealing schedule while
   progressively introducing computational-basis dephasing. Independent
   quantum-trajectory batches are used to estimate the variation in useful-state
   probability.

2. TRANSVERSE-DRIVER CONTROL
   Hold the problem instance and schedule fixed while varying the transverse
   coupling strength. At zero transverse coupling, a computational-basis start
   state cannot transition to another bit string. The experiment measures how
   useful-state probability changes as coherent transverse coupling is introduced.

3. COHERENT TRANSFER ACROSS FLUX-DERIVED TRANSITIONS
   For each selected Flux-derived local subproblem, evolve under a static
   transverse-field Hamiltonian at several mixing points and measure coherent
   transfer from the initial configuration to the improved target configuration
   as a function of evolution time.

   Barrier-matched comparisons are included where appropriate so that transition
   width can be examined without directly conflating it with barrier severity.

4. PARALLEL-TEMPERING CONTROL
   On representative complete Flux subproblems, compare alternative
   parallel-tempering temperature ladders using the same replica count and
   computational budget. This provides a stronger classical thermal-mixing
   control for the corresponding Flux instances.

Interpretation
--------------
These experiments are mechanism studies and do not establish end-to-end quantum
computational advantage. The coherent-dynamics experiments necessarily operate
on small Flux-derived local subproblems because exact quantum simulation scales
exponentially with the number of simulated qubits.

The primary purpose is to test whether useful Flux transitions exhibit behavior
that depends on coherent transverse coupling and whether that behavior is
suppressed when coherence is removed.

Dependencies
------------
This script uses shared Flux instance-generation, landscape-analysis, and raw
scaling utilities contained in the accompanying experiment files and result
sets in this repository.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy import sparse
    from scipy.sparse.linalg import expm_multiply
    from scipy.stats import spearmanr
except Exception as exc:  # pragma: no cover
    raise RuntimeError("V12 requires scipy") from exc

HERE = Path(__file__).resolve()
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.instance_source import load_instance_set  # noqa: E402
from solvers.candidate_pool import CandidatePool  # noqa: E402
from experiments import eulerq_quantum_hardness_hunt_v8 as v8  # noqa: E402
from experiments import eulerq_quantum_mechanism_v11b as v11  # noqa: E402
from experiments import eulerq_raw_scaling_v10b as v10b  # noqa: E402


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")


def parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def safe_spearman(xs: list[float], ys: list[float]) -> dict[str, Any]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 4:
        return {"n": len(pairs), "rho": math.nan, "pvalue": math.nan}
    x = np.asarray([a for a, _ in pairs], dtype=float)
    y = np.asarray([b for _, b in pairs], dtype=float)
    z = spearmanr(x, y)
    return {"n": len(pairs), "rho": float(z.statistic), "pvalue": float(z.pvalue)}


# -----------------------------------------------------------------------------
# Reconstruct exactly the V11b natural Flux cores and schedules
# -----------------------------------------------------------------------------


def reconstruct_cases(v8dir: Path, v11dir: Path, nearest_seconds: float, seed: int) -> list[dict[str, Any]]:
    case_rows = read_csv(v11dir / "cases.csv")
    barrier_rows = read_csv(v8dir / "controlled_barriers.csv")
    screen_rows = read_csv(v8dir / "screening.csv")
    screen_map = {(r["set_name"], int(r["instance_index"])): r for r in screen_rows}

    bmap: dict[tuple[str, int, str], dict[str, str]] = {}
    for r in barrier_rows:
        key = (r["set_name"], int(r["instance_index"]), f"{float(r['quality_fraction']):.12g}")
        bmap[key] = r

    sched_json = json.loads((v11dir / "quantum_selected_schedules.json").read_text(encoding="utf-8"))
    per_case = sched_json["per_case"]

    out: list[dict[str, Any]] = []
    for j, cr in enumerate(case_rows):
        ci = int(cr["case_id"])
        key = (cr["set_name"], int(cr["instance_index"]), f"{float(cr['quality_fraction']):.12g}")
        br = bmap.get(key)
        if br is None:
            raise RuntimeError(f"Cannot find V8 barrier row for {key}")
        sr = screen_map[(cr["set_name"], int(cr["instance_index"]))]
        c = v11.load_case(br, sr, seed + 10000 * j, nearest_seconds)
        cube = v11.make_cube(c["terms"], 1.0)
        geom = v11.cube_geometry(cube)
        sj = per_case[str(ci)]
        schedule = v11.QSchedule(
            float(sj["smin"]), float(sj["ramp_time"]), float(sj["hold_time"]), float(sj["driver_scale"])
        )
        c.update({"case_id": ci, "cube": cube, "geom": geom, "schedule": schedule})
        out.append(c)
    out.sort(key=lambda c: int(c["case_id"]))
    return out


# -----------------------------------------------------------------------------
# Experiment 1: replicated coherence destruction
# -----------------------------------------------------------------------------


def run_coherence_replication(
    cases: list[dict[str, Any]],
    gammas: list[float],
    repeats: int,
    trajectories: int,
    ramp_steps: int,
    hold_steps_per_unit: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw: list[dict[str, Any]] = []
    for c in cases:
        ci = int(c["case_id"])
        cube = c["cube"]
        sch = c["schedule"]
        target_e = float(cube["energies"][cube["target_idx"]])
        p_uniform = 1.0 / float(1 << int(cube["k"]))

        # Exact coherent baseline — no Monte Carlo uncertainty.
        coh = v11.coherent_quantum(
            cube, schedule=sch, target_energy_raw=target_e,
            ramp_steps=ramp_steps, hold_steps_per_unit=hold_steps_per_unit,
        )
        raw.append({
            "case_id": ci, "set_name": c["set_name"], "instance_index": c["instance_index"],
            "k": cube["k"], "barrier_over_gain": c["geom"]["barrier_over_gain"],
            "gamma": 0.0, "repeat": -1, "trajectories": 0,
            "p_target_quality_or_better": coh["p_target_quality_or_better"],
            "p_target": coh["p_target"], "p_feasible": coh["p_feasible"],
            "p_uniform_target": p_uniform,
            "target_enrichment_vs_uniform": coh["p_target"] / p_uniform,
            "schedule_duration": coh["schedule_duration"],
            "monte_carlo_se": 0.0,
        })

        for gamma in gammas:
            if gamma <= 0:
                continue
            for rep in range(repeats):
                z = v11.dephased_quantum_trajectories(
                    cube, schedule=sch, target_energy_raw=target_e,
                    gamma=float(gamma), trajectories=trajectories,
                    ramp_steps=ramp_steps, hold_steps_per_unit=hold_steps_per_unit,
                    seed=seed + ci * 100000 + int(round(gamma * 10000)) * 100 + rep,
                )
                raw.append({
                    "case_id": ci, "set_name": c["set_name"], "instance_index": c["instance_index"],
                    "k": cube["k"], "barrier_over_gain": c["geom"]["barrier_over_gain"],
                    "gamma": gamma, "repeat": rep, "trajectories": trajectories,
                    "p_target_quality_or_better": z["p_target_quality_or_better"],
                    "p_target": z["p_target"], "p_feasible": z["p_feasible"],
                    "p_uniform_target": p_uniform,
                    "target_enrichment_vs_uniform": z["p_target"] / p_uniform,
                    "schedule_duration": z["schedule_duration"],
                    "dephasing_dose_gamma_t": gamma * z["schedule_duration"],
                    "monte_carlo_se": z["p_target_quality_se"],
                })

    agg: list[dict[str, Any]] = []
    for c in cases:
        ci = int(c["case_id"])
        base = next(r for r in raw if r["case_id"] == ci and r["gamma"] == 0.0)
        p0 = float(base["p_target_quality_or_better"])
        for gamma in [0.0] + [g for g in gammas if g > 0]:
            rr = [r for r in raw if r["case_id"] == ci and abs(float(r["gamma"]) - gamma) < 1e-15]
            vals = np.asarray([float(r["p_target_quality_or_better"]) for r in rr], dtype=float)
            if gamma == 0:
                mean = p0
                se = 0.0
            else:
                mean = float(np.mean(vals))
                # Combine between-batch spread and the trajectory SEs conservatively.
                between = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
                within = math.sqrt(sum(float(r["monte_carlo_se"]) ** 2 for r in rr)) / max(1, len(rr))
                se = max(between, within)
            agg.append({
                "case_id": ci, "set_name": c["set_name"], "instance_index": c["instance_index"],
                "k": c["cube"]["k"], "barrier_over_gain": c["geom"]["barrier_over_gain"],
                "gamma": gamma, "mean_p_quality": mean, "se_p_quality": se,
                "ci95_low": max(0.0, mean - 1.96 * se),
                "ci95_high": min(1.0, mean + 1.96 * se),
                "retention_vs_coherent": mean / p0 if p0 > 0 else math.nan,
                "suppression_fraction": 1.0 - mean / p0 if p0 > 0 else math.nan,
                "n_batches": len(rr),
            })
    return raw, agg


# -----------------------------------------------------------------------------
# Experiment 2: transverse-driver strength control
# -----------------------------------------------------------------------------


def run_driver_sweep(
    cases: list[dict[str, Any]],
    driver_scales: list[float],
    ramp_steps: int,
    hold_steps_per_unit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in cases:
        cube = c["cube"]
        base = c["schedule"]
        target_e = float(cube["energies"][cube["target_idx"]])
        p_uniform = 1.0 / float(1 << int(cube["k"]))
        for ds in driver_scales:
            sch = v11.QSchedule(base.smin, base.ramp_time, base.hold_time, float(ds))
            z = v11.coherent_quantum(
                cube, schedule=sch, target_energy_raw=target_e,
                ramp_steps=ramp_steps, hold_steps_per_unit=hold_steps_per_unit,
            )
            rows.append({
                "case_id": c["case_id"], "set_name": c["set_name"], "instance_index": c["instance_index"],
                "k": cube["k"], "barrier_over_gain": c["geom"]["barrier_over_gain"],
                "driver_scale": ds, "base_smin": base.smin, "base_ramp": base.ramp_time,
                "base_hold": base.hold_time,
                "p_target": z["p_target"],
                "p_quality": z["p_target_quality_or_better"],
                "p_feasible": z["p_feasible"], "p_start": z["p_start"],
                "target_enrichment_vs_uniform": z["p_target"] / p_uniform,
            })
    return rows


# -----------------------------------------------------------------------------
# Experiment 3: direct static coherent transfer / width fingerprint
# -----------------------------------------------------------------------------


def static_transfer(
    cube: dict[str, Any],
    *,
    s: float,
    driver_scale: float,
    hold_max: float,
    hold_points: int,
) -> dict[str, Any]:
    problem, _ = v11.normalized_problem(cube)
    k = int(cube["k"])
    dim = len(problem)
    si, ti = int(cube["start_idx"]), int(cube["target_idx"])
    feasible = np.asarray(cube["feasible"], dtype=bool)
    Eraw = np.asarray(cube["energies"], dtype=float)
    target_e = float(Eraw[ti])
    quality = feasible & (Eraw <= target_e + 1e-12)

    Hd = v11.transverse_driver_sparse(k)
    Hp = sparse.diags(problem, offsets=0, format="csr")
    H = float(driver_scale) * (1.0 - float(s)) * Hd + float(s) * Hp
    psi0 = np.zeros(dim, dtype=np.complex128)
    psi0[si] = 1.0
    t0 = time.perf_counter()
    psis = expm_multiply(
        (-1j) * H, psi0,
        start=0.0, stop=float(hold_max), num=int(hold_points), endpoint=True,
    )
    times = np.linspace(0.0, float(hold_max), int(hold_points))
    probs = np.abs(psis) ** 2
    pt = probs[:, ti]
    pq = np.sum(probs[:, quality], axis=1)
    pf = np.sum(probs[:, feasible], axis=1)

    imax = int(np.argmax(pt))
    iq = int(np.argmax(pq))
    # Time-averaged probability after the trivial t=0 point.
    start_idx = 1 if len(times) > 1 else 0
    return {
        "s": float(s), "driver_scale": float(driver_scale),
        "max_p_target": float(pt[imax]), "time_of_max_target": float(times[imax]),
        "max_p_quality": float(pq[iq]), "time_of_max_quality": float(times[iq]),
        "mean_p_target": float(np.mean(pt[start_idx:])),
        "mean_p_quality": float(np.mean(pq[start_idx:])),
        "max_p_feasible": float(np.max(pf)),
        "final_p_target": float(pt[-1]),
        "wall_ms": 1000.0 * (time.perf_counter() - t0),
    }


def run_static_width(
    cases: list[dict[str, Any]],
    s_values: list[float],
    driver_scales: list[float],
    hold_max: float,
    hold_points: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for c in cases:
        p_uniform = 1.0 / float(1 << int(c["cube"]["k"]))
        cbest: dict[str, Any] | None = None
        for s in s_values:
            for ds in driver_scales:
                z = static_transfer(c["cube"], s=s, driver_scale=ds, hold_max=hold_max, hold_points=hold_points)
                row = {
                    "case_id": c["case_id"], "set_name": c["set_name"], "instance_index": c["instance_index"],
                    "k": c["cube"]["k"], "barrier": c["geom"]["barrier"],
                    "gain": c["geom"]["gain"], "barrier_over_gain": c["geom"]["barrier_over_gain"],
                    "p_uniform_target": p_uniform,
                    "target_enrichment_at_max": z["max_p_target"] / p_uniform,
                    **z,
                }
                rows.append(row)
                if cbest is None or float(row["max_p_target"]) > float(cbest["max_p_target"]):
                    cbest = row.copy()
        assert cbest is not None
        best_rows.append(cbest)
    return rows, best_rows


# -----------------------------------------------------------------------------
# Experiment 4: custom-temperature raw PT + exchange-rate adaptation
# -----------------------------------------------------------------------------


def _swap_rows(arr: np.ndarray, mask: np.ndarray, j: int) -> None:
    if not np.any(mask):
        return
    tmp = arr[mask, j].copy()
    arr[mask, j] = arr[mask, j + 1]
    arr[mask, j + 1] = tmp


def run_raw_pt_custom(
    *, start: np.ndarray, d: v8.PoolData, nr: int, no: int,
    temperatures: np.ndarray, ensembles: int, sweeps: int,
    seed: int, max_wall_s: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    m = len(start)
    temps = np.asarray(temperatures, dtype=float)
    nt = len(temps)
    beta = 1.0 / temps

    states = np.broadcast_to(start, (ensembles, nt, m)).copy().astype(np.int8)
    occ = v8.replicate_occ(v8.start_occupancy(start, d, nr, no), (ensembles, nt))
    a0, d0, _ = v8.eval_bits(start, d)
    assigned = np.full((ensembles, nt), a0, dtype=np.int32)
    dist = np.full((ensembles, nt), d0, dtype=np.float64)
    energy = np.full((ensembles, nt), -float(start @ d.reward), dtype=np.float64)

    best_a, best_d = a0, d0
    first_improve = -1
    accepted_by_temp = np.zeros(nt, dtype=np.int64)
    attempted_by_temp = np.zeros(nt, dtype=np.int64)
    swap_a = np.zeros(nt - 1, dtype=np.int64)
    swap_n = np.zeros(nt - 1, dtype=np.int64)

    nwalkers = ensembles * nt
    walker = np.arange(nwalkers, dtype=np.int32).reshape(ensembles, nt)
    phase = np.zeros(nwalkers, dtype=np.int8)
    start_sw = np.full(nwalkers, -1, dtype=np.int32)
    rt = np.zeros(nwalkers, dtype=np.int32)
    rt_durations: list[int] = []
    for wid in walker[:, 0]:
        phase[int(wid)] = 1
        start_sw[int(wid)] = 0

    t0 = time.perf_counter()
    completed = 0
    tempmat = temps[None, :]
    for sw in range(sweeps):
        # One complete random-permutation single-spin sweep.
        for i in rng.permutation(m):
            b = states[:, :, i].astype(np.int16)
            nc = v8.neighbor_count(b, occ, d, int(i)).astype(np.float64)
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
                v8.apply_flip_update(occ, d, int(i), dx)
                assigned += dx * int(d.count[i])
                dist += dx * float(d.distance[i])
                energy += np.where(acc, delta, 0.0)

        parity = sw & 1
        for j in range(parity, nt - 1, 2):
            log_ratio = (beta[j] - beta[j + 1]) * (energy[:, j] - energy[:, j + 1])
            mask = np.log(np.maximum(rng.random(ensembles), 1e-300)) < np.minimum(0.0, log_ratio)
            swap_n[j] += ensembles
            swap_a[j] += int(np.sum(mask))
            if np.any(mask):
                _swap_rows(states, mask, j)
                _swap_rows(occ.rider, mask, j)
                _swap_rows(occ.order, mask, j)
                _swap_rows(occ.rider_order, mask, j)
                _swap_rows(occ.order_pair, mask, j)
                _swap_rows(assigned, mask, j)
                _swap_rows(dist, mask, j)
                _swap_rows(energy, mask, j)
                _swap_rows(walker, mask, j)

        sweep_no = sw + 1
        for wid0 in walker[:, 0].tolist():
            wid = int(wid0)
            if phase[wid] == 0:
                phase[wid] = 1
                start_sw[wid] = sweep_no
            elif phase[wid] == 2:
                rt[wid] += 1
                if start_sw[wid] >= 0:
                    rt_durations.append(sweep_no - int(start_sw[wid]))
                phase[wid] = 1
                start_sw[wid] = sweep_no
        for wid0 in walker[:, -1].tolist():
            wid = int(wid0)
            if phase[wid] == 1:
                phase[wid] = 2

        feas = v8.feasible_mask(occ)
        er, et = np.where(feas)
        for ee, tt in zip(er.tolist(), et.tolist()):
            aa, dd = int(assigned[ee, tt]), float(dist[ee, tt])
            if aa > best_a or (aa == best_a and dd < best_d - 1e-12):
                best_a, best_d = aa, dd
            if first_improve < 0 and aa == a0 and dd < d0 - 1e-10:
                first_improve = sweep_no

        completed = sweep_no
        if time.perf_counter() - t0 >= max_wall_s:
            break

    spin_rates = accepted_by_temp / np.maximum(attempted_by_temp, 1)
    edge_rates = swap_a / np.maximum(swap_n, 1)
    return {
        "assigned": int(best_a), "distance": float(best_d),
        "same_card_improved": bool(best_a == a0 and best_d < d0 - 1e-10),
        "first_same_card_improvement_sweep": int(first_improve),
        "sweeps_completed": int(completed),
        "spin_acceptance_mean": float(np.mean(spin_rates)),
        "spin_acceptance_cold": float(spin_rates[0]),
        "spin_acceptance_hot": float(spin_rates[-1]),
        "swap_acceptance_mean": float(np.mean(edge_rates)),
        "swap_acceptance_min": float(np.min(edge_rates)),
        "blocked_edges_lt_1pct": int(np.sum(edge_rates < 0.01)),
        "roundtrips_total": int(np.sum(rt)),
        "walkers_with_roundtrip": int(np.sum(rt > 0)),
        "roundtrip_fraction_walkers": float(np.mean(rt > 0)),
        "median_roundtrip_sweeps": float(np.median(rt_durations)) if rt_durations else math.nan,
        "wall_ms": 1000.0 * (time.perf_counter() - t0),
        "temperatures_json": json.dumps(temps.tolist()),
        "edge_swap_rates_json": json.dumps(edge_rates.tolist()),
        "spin_acceptance_by_temp_json": json.dumps(spin_rates.tolist()),
    }


def adapt_raw_temperatures(
    *, start: np.ndarray, d: v8.PoolData, nr: int, no: int,
    tmin: float, tmax: float, ntemps: int, iterations: int,
    pilot_sweeps: int, seed: int, max_wall_s: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    temps = np.geomspace(tmin, tmax, ntemps)
    hist: list[dict[str, Any]] = []
    for it in range(iterations):
        z = run_raw_pt_custom(
            start=start, d=d, nr=nr, no=no, temperatures=temps,
            ensembles=1, sweeps=pilot_sweeps,
            seed=seed + 7919 * it, max_wall_s=max_wall_s,
        )
        rates = np.asarray(json.loads(z["edge_swap_rates_json"]), dtype=float)
        hist.append({
            "iteration": it, "min_swap": z["swap_acceptance_min"],
            "mean_swap": z["swap_acceptance_mean"],
            "blocked_edges_lt_1pct": z["blocked_edges_lt_1pct"],
            "roundtrips_total": z["roundtrips_total"],
            "temperatures_json": z["temperatures_json"],
            "edge_swap_rates_json": z["edge_swap_rates_json"],
        })
        if len(rates) == 0:
            break
        logt = np.log(temps)
        resistance = 1.0 / np.clip(rates, 0.002, 0.95)
        resistance = np.clip(resistance, 1.0, 100.0)
        cumulative = np.concatenate([[0.0], np.cumsum(resistance)])
        targets = np.linspace(0.0, cumulative[-1], ntemps)
        temps = np.exp(np.interp(targets, cumulative, logt))
        temps[0], temps[-1] = tmin, tmax
    return temps, hist


def run_raw_adaptive_pt(
    scales: list[str],
    set_map: dict[str, str],
    v10c_screening: Path,
    ntemps_list: list[int],
    tmin: float,
    tmax: float,
    adapt_iters: int,
    pilot_sweeps: int,
    final_sweeps: int,
    ensembles: int,
    max_wall_s: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    screen = read_csv(v10c_screening)
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []

    for scale in scales:
        sr = [r for r in screen if r["scale"] == scale and float(r["distance_gain"]) > 1e-9]
        if not sr:
            continue
        # Choose the case with the most expensive reference optimization.
        chosen = max(sr, key=lambda r: float(r["reference_optimize_ms"]))
        idx = int(chosen["instance_index"])
        inst = load_instance_set(set_map[scale])[idx]
        pool = CandidatePool(inst)
        d = v8.pool_data(inst, pool)
        start = v8.greedy_pair_first(pool, d)
        ga, gd, _ = v8.eval_bits(start, d)
        ref_a = int(float(chosen["reference_assigned"]))
        ref_d = float(chosen["reference_distance"])
        available = gd - ref_d if ga == ref_a else math.nan

        for nt in ntemps_list:
            geom = np.geomspace(tmin, tmax, nt)
            gz = run_raw_pt_custom(
                start=start, d=d, nr=inst.num_riders, no=inst.num_orders,
                temperatures=geom, ensembles=ensembles, sweeps=final_sweeps,
                seed=seed + idx * 1000 + nt, max_wall_s=max_wall_s,
            )
            rec = (gd - float(gz["distance"])) / available if available > 1e-12 and int(gz["assigned"]) == ga else 0.0
            rows.append({
                "scale": scale, "instance_index": idx, "instance_id": chosen["instance_id"],
                "num_candidates": len(pool.candidates), "variant": "geometric",
                "ntemps": nt, "tmin": tmin, "tmax": tmax,
                "greedy_distance": gd, "reference_distance": ref_d,
                "available_distance_gain": available, "refinement_fraction": rec,
                **gz,
            })

            at, hist = adapt_raw_temperatures(
                start=start, d=d, nr=inst.num_riders, no=inst.num_orders,
                tmin=tmin, tmax=tmax, ntemps=nt, iterations=adapt_iters,
                pilot_sweeps=pilot_sweeps,
                seed=seed + 500000 + idx * 1000 + nt,
                max_wall_s=max_wall_s,
            )
            for h in hist:
                history.append({
                    "scale": scale, "instance_index": idx, "ntemps": nt,
                    "tmin": tmin, "tmax": tmax, **h,
                })
            az = run_raw_pt_custom(
                start=start, d=d, nr=inst.num_riders, no=inst.num_orders,
                temperatures=at, ensembles=ensembles, sweeps=final_sweeps,
                seed=seed + 900000 + idx * 1000 + nt, max_wall_s=max_wall_s,
            )
            rec = (gd - float(az["distance"])) / available if available > 1e-12 and int(az["assigned"]) == ga else 0.0
            rows.append({
                "scale": scale, "instance_index": idx, "instance_id": chosen["instance_id"],
                "num_candidates": len(pool.candidates), "variant": "exchange_rate_adapted",
                "ntemps": nt, "tmin": tmin, "tmax": tmax,
                "greedy_distance": gd, "reference_distance": ref_d,
                "available_distance_gain": available, "refinement_fraction": rec,
                **az,
            })
    return rows, history


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------


def build_summary(
    cases: list[dict[str, Any]],
    coherence_agg: list[dict[str, Any]],
    driver_rows: list[dict[str, Any]],
    width_best: list[dict[str, Any]],
    raw_pt: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_cases": len(cases)}

    if coherence_agg:
        max_gamma = max(float(r["gamma"]) for r in coherence_agg)
        hi = [r for r in coherence_agg if abs(float(r["gamma"]) - max_gamma) < 1e-12]
        suppress = [float(r["suppression_fraction"]) for r in hi if finite(r["suppression_fraction"])]
        monotonic = 0
        for c in cases:
            rr = sorted([r for r in coherence_agg if int(r["case_id"]) == int(c["case_id"])], key=lambda r: float(r["gamma"]))
            vals = [float(r["mean_p_quality"]) for r in rr]
            if all(vals[i + 1] <= vals[i] + 2.0 * max(float(rr[i]["se_p_quality"]), float(rr[i + 1]["se_p_quality"])) for i in range(len(vals) - 1)):
                monotonic += 1
        summary["coherence"] = {
            "max_gamma": max_gamma,
            "median_suppression_at_max_gamma": float(np.median(suppress)) if suppress else math.nan,
            "mean_suppression_at_max_gamma": float(np.mean(suppress)) if suppress else math.nan,
            "cases_monotone_within_2se": monotonic,
            "n_cases": len(cases),
        }

    if driver_rows:
        no_driver = [float(r["p_quality"]) for r in driver_rows if abs(float(r["driver_scale"])) < 1e-15]
        bests = []
        for c in cases:
            rr = [r for r in driver_rows if int(r["case_id"]) == int(c["case_id"])]
            bests.append(max(float(r["p_quality"]) for r in rr))
        summary["driver_control"] = {
            "max_no_driver_p_quality": max(no_driver) if no_driver else math.nan,
            "median_best_with_driver_p_quality": float(np.median(bests)) if bests else math.nan,
        }

    if width_best:
        ks = [float(r["k"]) for r in width_best]
        ps = [float(r["max_p_target"]) for r in width_best]
        ts = [float(r["time_of_max_target"]) for r in width_best]
        br = [float(r["barrier_over_gain"]) for r in width_best]
        matched = [r for r in width_best if 150.0 <= float(r["barrier_over_gain"]) <= 500.0]
        summary["static_transfer"] = {
            "all_k_vs_max_p": safe_spearman(ks, ps),
            "all_k_vs_peak_time": safe_spearman(ks, ts),
            "barrier_ratio_vs_max_p": safe_spearman(br, ps),
            "matched_barrier_window": [150.0, 500.0],
            "matched_n": len(matched),
            "matched_k_vs_max_p": safe_spearman([float(r["k"]) for r in matched], [float(r["max_p_target"]) for r in matched]),
            "matched_k_vs_peak_time": safe_spearman([float(r["k"]) for r in matched], [float(r["time_of_max_target"]) for r in matched]),
        }

    if raw_pt:
        summary["raw_adaptive_pt"] = []
        for scale in sorted({r["scale"] for r in raw_pt}):
            rr = [r for r in raw_pt if r["scale"] == scale]
            summary["raw_adaptive_pt"].append({
                "scale": scale,
                "runs": len(rr),
                "best_refinement_fraction": max(float(r["refinement_fraction"]) for r in rr),
                "max_roundtrips": max(int(r["roundtrips_total"]) for r in rr),
                "min_blocked_edges": min(int(r["blocked_edges_lt_1pct"]) for r in rr),
                "best_variant": max(rr, key=lambda r: float(r["refinement_fraction"]))["variant"],
            })
    return summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8-results", default="experiments/results/quantum_hardness_hunt_v8")
    ap.add_argument("--v11-results", default="experiments/results/quantum_mechanism_v11b")
    ap.add_argument("--v10c-screening", default="experiments/results/raw_scaling_v10c/screening.csv")
    ap.add_argument("--output", default="experiments/results/quantum_evidence_v12")
    ap.add_argument("--modules", default="all", help="all or comma list: coherence,driver,width,rawpt")
    ap.add_argument("--nearest-seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=3100000)

    # Coherence replication.
    ap.add_argument("--dephasing-gammas", default="0.01,0.03,0.1,0.3,1.0")
    ap.add_argument("--dephasing-repeats", type=int, default=4)
    ap.add_argument("--dephasing-trajectories", type=int, default=128)
    ap.add_argument("--q-ramp-steps", type=int, default=80)
    ap.add_argument("--q-hold-steps-per-unit", type=int, default=16)

    # Driver control.
    ap.add_argument("--driver-scales", default="0,0.25,0.5,0.75,1,1.25,1.5,2")

    # Static coherent-transfer width test.
    ap.add_argument("--static-s-values", default="0.15,0.25,0.35,0.5")
    ap.add_argument("--static-driver-scales", default="0.75,1,1.25,1.5")
    ap.add_argument("--static-hold-max", type=float, default=20.0)
    ap.add_argument("--static-hold-points", type=int, default=81)

    # Raw adaptive PT.
    ap.add_argument("--raw-pt-scales", default="15x40,20x50")
    ap.add_argument("--raw-pt-sets", default="15x40=committee_test_15x40_v10,20x50=committee_test_20x50_v10")
    ap.add_argument("--raw-pt-ntemps", default="32,64")
    ap.add_argument("--raw-pt-tmin", type=float, default=0.0005)
    ap.add_argument("--raw-pt-tmax", type=float, default=20.0)
    ap.add_argument("--raw-pt-adapt-iters", type=int, default=2)
    ap.add_argument("--raw-pt-pilot-sweeps", type=int, default=12)
    ap.add_argument("--raw-pt-final-sweeps", type=int, default=64)
    ap.add_argument("--raw-pt-ensembles", type=int, default=1)
    ap.add_argument("--raw-pt-max-wall", type=float, default=90.0)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    v8dir = Path(args.v8_results)
    v11dir = Path(args.v11_results)
    modules = {x.strip() for x in args.modules.split(",") if x.strip()}
    if "all" in modules:
        modules = {"coherence", "driver", "width", "rawpt"}

    cases = reconstruct_cases(v8dir, v11dir, args.nearest_seconds, args.seed)
    print(f"Reconstructed {len(cases)} natural Flux cores", flush=True)
    write_csv(out / "cases.csv", [
        {
            "case_id": c["case_id"], "set_name": c["set_name"], "instance_index": c["instance_index"],
            "k": c["cube"]["k"], "gain": c["geom"]["gain"], "barrier": c["geom"]["barrier"],
            "barrier_over_gain": c["geom"]["barrier_over_gain"],
            "smin": c["schedule"].smin, "ramp_time": c["schedule"].ramp_time,
            "hold_time": c["schedule"].hold_time, "driver_scale": c["schedule"].driver_scale,
        }
        for c in cases
    ])

    coh_raw: list[dict[str, Any]] = []
    coh_agg: list[dict[str, Any]] = []
    driver_rows: list[dict[str, Any]] = []
    width_rows: list[dict[str, Any]] = []
    width_best: list[dict[str, Any]] = []
    raw_pt: list[dict[str, Any]] = []
    raw_pt_hist: list[dict[str, Any]] = []

    if "coherence" in modules:
        print("[1/4] replicated coherence-destruction experiment", flush=True)
        coh_raw, coh_agg = run_coherence_replication(
            cases, parse_floats(args.dephasing_gammas), args.dephasing_repeats,
            args.dephasing_trajectories, args.q_ramp_steps, args.q_hold_steps_per_unit, args.seed,
        )
        write_csv(out / "coherence_replication_raw.csv", coh_raw)
        write_csv(out / "coherence_replication_summary.csv", coh_agg)

    if "driver" in modules:
        print("[2/4] transverse-driver control", flush=True)
        driver_rows = run_driver_sweep(
            cases, parse_floats(args.driver_scales), args.q_ramp_steps, args.q_hold_steps_per_unit,
        )
        write_csv(out / "driver_strength_control.csv", driver_rows)

    if "width" in modules:
        print("[3/4] direct static coherent-transfer / width fingerprint", flush=True)
        width_rows, width_best = run_static_width(
            cases, parse_floats(args.static_s_values), parse_floats(args.static_driver_scales),
            args.static_hold_max, args.static_hold_points,
        )
        write_csv(out / "static_coherent_transfer.csv", width_rows)
        write_csv(out / "static_coherent_transfer_best.csv", width_best)
        matched = [r for r in width_best if 150.0 <= float(r["barrier_over_gain"]) <= 500.0]
        write_csv(out / "static_coherent_transfer_matched_barrier.csv", matched)

    if "rawpt" in modules:
        print("[4/4] strong raw adaptive-PT control", flush=True)
        set_map: dict[str, str] = {}
        for item in args.raw_pt_sets.split(","):
            k, v = item.split("=", 1)
            set_map[k.strip()] = v.strip()
        raw_pt, raw_pt_hist = run_raw_adaptive_pt(
            [x.strip() for x in args.raw_pt_scales.split(",") if x.strip()], set_map,
            Path(args.v10c_screening), parse_ints(args.raw_pt_ntemps),
            args.raw_pt_tmin, args.raw_pt_tmax, args.raw_pt_adapt_iters,
            args.raw_pt_pilot_sweeps, args.raw_pt_final_sweeps,
            args.raw_pt_ensembles, args.raw_pt_max_wall, args.seed,
        )
        write_csv(out / "raw_adaptive_pt.csv", raw_pt)
        write_csv(out / "raw_adaptive_pt_history.csv", raw_pt_hist)

    summary = build_summary(cases, coh_agg, driver_rows, width_best, raw_pt)
    summary["interpretation_guardrail"] = (
        "Mechanism evidence only: exact quantum simulations are small-core controls; "
        "do not describe these results as end-to-end quantum speedup."
    )
    write_json(out / "committee_evidence_summary.json", summary)
    write_json(out / "run_config.json", vars(args))
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
