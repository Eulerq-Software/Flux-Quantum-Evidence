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
This public version uses only exported Flux-derived local subproblems and the
public quantum utilities in common.quantum_core. Production Flux candidate
generation, routing, and solver code are not required.
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

from common import quantum_core as qc  # noqa: E402


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
# Load already-exported local Flux subproblems
# -----------------------------------------------------------------------------


def load_cases(set_name: str) -> list[dict[str, Any]]:
    return qc.load_local_cases(set_name)


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
        coh = qc.coherent_quantum(
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
                z = qc.dephased_quantum_trajectories(
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
            sch = qc.QSchedule(base.smin, base.ramp_time, base.hold_time, float(ds))
            z = qc.coherent_quantum(
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
    problem, _ = qc.normalized_problem(cube)
    k = int(cube["k"])
    dim = len(problem)
    si, ti = int(cube["start_idx"]), int(cube["target_idx"])
    feasible = np.asarray(cube["feasible"], dtype=bool)
    Eraw = np.asarray(cube["energies"], dtype=float)
    target_e = float(Eraw[ti])
    quality = feasible & (Eraw <= target_e + 1e-12)

    Hd = qc.transverse_driver_sparse(k)
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
# Summary
# -----------------------------------------------------------------------------


def build_summary(
    cases: list[dict[str, Any]],
    coherence_agg: list[dict[str, Any]],
    driver_rows: list[dict[str, Any]],
    width_best: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_cases": len(cases)}

    if coherence_agg:
        max_gamma = max(float(r["gamma"]) for r in coherence_agg)
        hi = [
            r
            for r in coherence_agg
            if abs(float(r["gamma"]) - max_gamma) < 1e-12
        ]
        suppress = [
            float(r["suppression_fraction"])
            for r in hi
            if finite(r["suppression_fraction"])
        ]
        monotonic = 0
        for c in cases:
            rr = sorted(
                [
                    r
                    for r in coherence_agg
                    if int(r["case_id"]) == int(c["case_id"])
                ],
                key=lambda r: float(r["gamma"]),
            )
            vals = [float(r["mean_p_quality"]) for r in rr]
            if all(
                vals[i + 1]
                <= vals[i]
                + 2.0
                * max(
                    float(rr[i]["se_p_quality"]),
                    float(rr[i + 1]["se_p_quality"]),
                )
                for i in range(len(vals) - 1)
            ):
                monotonic += 1

        summary["coherence"] = {
            "max_gamma": max_gamma,
            "median_suppression_at_max_gamma": (
                float(np.median(suppress))
                if suppress
                else math.nan
            ),
            "mean_suppression_at_max_gamma": (
                float(np.mean(suppress))
                if suppress
                else math.nan
            ),
            "cases_monotone_within_2se": monotonic,
            "n_cases": len(cases),
        }

    if driver_rows:
        no_driver = [
            float(r["p_quality"])
            for r in driver_rows
            if abs(float(r["driver_scale"])) < 1e-15
        ]
        bests = []
        for c in cases:
            rr = [
                r
                for r in driver_rows
                if int(r["case_id"]) == int(c["case_id"])
            ]
            bests.append(
                max(float(r["p_quality"]) for r in rr)
            )
        summary["driver_control"] = {
            "max_no_driver_p_quality": (
                max(no_driver)
                if no_driver
                else math.nan
            ),
            "median_best_with_driver_p_quality": (
                float(np.median(bests))
                if bests
                else math.nan
            ),
        }

    if width_best:
        ks = [float(r["k"]) for r in width_best]
        ps = [float(r["max_p_target"]) for r in width_best]
        ts = [float(r["time_of_max_target"]) for r in width_best]
        br = [
            float(r["barrier_over_gain"])
            for r in width_best
        ]
        matched = [
            r
            for r in width_best
            if 150.0
            <= float(r["barrier_over_gain"])
            <= 500.0
        ]
        summary["static_transfer"] = {
            "all_k_vs_max_p": safe_spearman(ks, ps),
            "all_k_vs_peak_time": safe_spearman(ks, ts),
            "barrier_ratio_vs_max_p": safe_spearman(br, ps),
            "matched_barrier_window": [150.0, 500.0],
            "matched_n": len(matched),
            "matched_k_vs_max_p": safe_spearman(
                [float(r["k"]) for r in matched],
                [float(r["max_p_target"]) for r in matched],
            ),
            "matched_k_vs_peak_time": safe_spearman(
                [float(r["k"]) for r in matched],
                [float(r["time_of_max_target"]) for r in matched],
            ),
        }

    return summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cases",
        default="quantum_mechanism",
        help=(
            "Directory name under instances/local_subproblems/ "
            "containing exported V12 local cases."
        ),
    )
    ap.add_argument(
        "--output",
        default="quantum_mechanism/results",
    )
    ap.add_argument(
        "--modules",
        default="all",
        help="all or comma list: coherence,driver,width",
    )
    ap.add_argument("--seed", type=int, default=3100000)

    ap.add_argument(
        "--dephasing-gammas",
        default="0.01,0.03,0.1,0.3,1.0",
    )
    ap.add_argument(
        "--dephasing-repeats",
        type=int,
        default=4,
    )
    ap.add_argument(
        "--dephasing-trajectories",
        type=int,
        default=128,
    )
    ap.add_argument(
        "--q-ramp-steps",
        type=int,
        default=80,
    )
    ap.add_argument(
        "--q-hold-steps-per-unit",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--driver-scales",
        default="0,0.25,0.5,0.75,1,1.25,1.5,2",
    )

    ap.add_argument(
        "--static-s-values",
        default="0.15,0.25,0.35,0.5",
    )
    ap.add_argument(
        "--static-driver-scales",
        default="0.75,1,1.25,1.5",
    )
    ap.add_argument(
        "--static-hold-max",
        type=float,
        default=20.0,
    )
    ap.add_argument(
        "--static-hold-points",
        type=int,
        default=81,
    )
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    modules = {
        x.strip()
        for x in args.modules.split(",")
        if x.strip()
    }
    if "all" in modules:
        modules = {"coherence", "driver", "width"}
    unknown = modules - {"coherence", "driver", "width"}
    if unknown:
        raise ValueError(
            f"Unknown modules: {sorted(unknown)}"
        )

    cases = load_cases(args.cases)
    print(
        f"Loaded {len(cases)} exported Flux-derived local cases",
        flush=True,
    )
    write_csv(
        out / "cases.csv",
        [
            {
                "case_id": c["case_id"],
                "set_name": c["set_name"],
                "instance_index": c["instance_index"],
                "k": c["cube"]["k"],
                "gain": c["geom"]["gain"],
                "barrier": c["geom"]["barrier"],
                "barrier_over_gain": c["geom"]["barrier_over_gain"],
                "smin": c["schedule"].smin,
                "ramp_time": c["schedule"].ramp_time,
                "hold_time": c["schedule"].hold_time,
                "driver_scale": c["schedule"].driver_scale,
            }
            for c in cases
        ],
    )

    coh_raw: list[dict[str, Any]] = []
    coh_agg: list[dict[str, Any]] = []
    driver_rows: list[dict[str, Any]] = []
    width_rows: list[dict[str, Any]] = []
    width_best: list[dict[str, Any]] = []

    if "coherence" in modules:
        print(
            "[1/3] replicated coherence-destruction experiment",
            flush=True,
        )
        coh_raw, coh_agg = run_coherence_replication(
            cases,
            parse_floats(args.dephasing_gammas),
            args.dephasing_repeats,
            args.dephasing_trajectories,
            args.q_ramp_steps,
            args.q_hold_steps_per_unit,
            args.seed,
        )
        write_csv(
            out / "coherence_replication_raw.csv",
            coh_raw,
        )
        write_csv(
            out / "coherence_replication_summary.csv",
            coh_agg,
        )

    if "driver" in modules:
        print(
            "[2/3] transverse-driver control",
            flush=True,
        )
        driver_rows = run_driver_sweep(
            cases,
            parse_floats(args.driver_scales),
            args.q_ramp_steps,
            args.q_hold_steps_per_unit,
        )
        write_csv(
            out / "driver_strength_control.csv",
            driver_rows,
        )

    if "width" in modules:
        print(
            "[3/3] direct static coherent-transfer / width fingerprint",
            flush=True,
        )
        width_rows, width_best = run_static_width(
            cases,
            parse_floats(args.static_s_values),
            parse_floats(args.static_driver_scales),
            args.static_hold_max,
            args.static_hold_points,
        )
        write_csv(
            out / "static_coherent_transfer.csv",
            width_rows,
        )
        write_csv(
            out / "static_coherent_transfer_best.csv",
            width_best,
        )
        matched = [
            r
            for r in width_best
            if 150.0
            <= float(r["barrier_over_gain"])
            <= 500.0
        ]
        write_csv(
            out / "static_coherent_transfer_matched_barrier.csv",
            matched,
        )

    summary = build_summary(
        cases,
        coh_agg,
        driver_rows,
        width_best,
    )
    summary["interpretation_guardrail"] = (
        "Mechanism evidence only: exact quantum simulations are "
        "small-core controls; do not describe these results as "
        "end-to-end quantum speedup."
    )
    summary["raw_pt_note"] = (
        "The legacy V12 raw-PT module is intentionally not part of "
        "this public quantum-mechanism script. Full-problem PT "
        "diagnostics are reproduced in raw_scaling/experiment.py."
    )

    write_json(
        out / "committee_evidence_summary.json",
        summary,
    )
    write_json(
        out / "run_config.json",
        vars(args),
    )
    print(
        json.dumps(summary, indent=2, allow_nan=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
