from __future__ import annotations

"""
EulerQ — Flux-Structured Quantum Driver Experiment
===================================================

Purpose
-------
This experiment tests whether a quantum driver constructed from the Flux
candidate-conflict structure can couple correlated reassignment configurations
more effectively than a generic single-variable transverse-field driver.

The driver construction uses only the Flux conflict structure and does not use
information about the known improved target configuration.

Drivers
-------
The following quantum drivers are compared on the same Flux-derived local
subproblems:

    X
        Generic transverse-field driver based on single-variable transitions.

    XY_FLUX
        Exchange driver constructed from edges of the Flux candidate-conflict
        graph.

    X_PLUS_XY_FLUX
        Combined transverse-field and Flux-structured exchange driver.

    X_PLUS_XY_RAND
        Control driver using the same number of exchange interactions placed
        randomly rather than according to the Flux conflict structure.

The exchange interaction implements

    |10> <-> |01>,

corresponding to the coherent replacement of one selected assignment candidate
by a competing candidate. Such replacements naturally arise in rider/order
reassignment structures.

Method
------
For comparison, each driver is normalized to the same operator norm before
being multiplied by a common driver-strength parameter. The system is evolved
under a static mixing Hamiltonian of the form

    H = s H_problem + driver_scale (1 - s) H_driver.

The evolution begins from the incumbent assignment configuration. Coherent
transfer probability to a known improved configuration is measured over a
range of evolution times.

The randomized exchange-driver control is included to determine whether any
improvement arises merely from introducing additional two-variable couplings,
or specifically from using the Flux conflict structure.

Interpretation
--------------
This is a mechanism experiment on small Flux-derived subproblems and does not
constitute an end-to-end quantum speedup claim. Its purpose is to determine
whether the exchange structure present in Flux can be exploited by
problem-aware quantum dynamics.

A positive result motivates further investigation of structured exchange
interactions in quantum optimization methods such as reverse annealing and
alternating-operator approaches.
"""

import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, expm_multiply

HERE = Path(__file__).resolve()
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import eulerq_quantum_mechanism_v11b as v11  # noqa: E402
from experiments import eulerq_quantum_evidence_v12 as v12  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k); seen.add(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding='utf-8')


def parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(',') if x.strip()]


def x_driver(k: int) -> sparse.csr_matrix:
    # v11 driver is already -sum X_i.
    return v11.transverse_driver_sparse(k).tocsr()


def xy_exchange_driver(k: int, edges: list[tuple[int, int]]) -> sparse.csr_matrix:
    """Return -sum_edges (XX+YY)/2.

    On each edge this only couples 01 <-> 10 with matrix element -1; 00 and 11
    are unchanged. Thus it coherently exchanges occupation between alternatives.
    """
    dim = 1 << k
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for state in range(dim):
        for i, j in edges:
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi == bj:
                continue
            nxt = state ^ (1 << i) ^ (1 << j)
            rows.append(state); cols.append(nxt); data.append(-1.0)
    return sparse.coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=float).tocsr()


def op_norm(H: sparse.csr_matrix) -> float:
    if H.nnz == 0:
        return 0.0
    n = H.shape[0]
    if n <= 256:
        vals = np.linalg.eigvalsh(H.toarray())
        return float(np.max(np.abs(vals)))
    # Symmetric real Hamiltonian. Largest magnitude eigenvalue.
    try:
        hi = float(eigsh(H, k=1, which='LA', return_eigenvectors=False, tol=1e-7)[0])
        lo = float(eigsh(H, k=1, which='SA', return_eigenvectors=False, tol=1e-7)[0])
        return max(abs(hi), abs(lo))
    except Exception:
        # Conservative fallback for tiny k only.
        vals = np.linalg.eigvalsh(H.toarray())
        return float(np.max(np.abs(vals)))


def normalize_driver(H: sparse.csr_matrix) -> tuple[sparse.csr_matrix, float]:
    nrm = op_norm(H)
    if nrm <= 1e-14:
        return H.copy(), nrm
    return (H * (1.0 / nrm)).tocsr(), nrm


def conflict_edges(terms: dict[str, Any]) -> list[tuple[int, int]]:
    A = np.asarray(terms['A'], dtype=np.int8)
    k = A.shape[0]
    return [(i, j) for i in range(k) for j in range(i + 1, k) if A[i, j] != 0]


def random_edges(k: int, n_edges: int, seed: int) -> list[tuple[int, int]]:
    all_edges = [(i, j) for i in range(k) for j in range(i + 1, k)]
    if n_edges >= len(all_edges):
        return all_edges
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(all_edges), size=n_edges, replace=False)
    return [all_edges[int(x)] for x in pick]


def neighbors_for_driver(state: int, k: int, driver_name: str, xy_edges: list[tuple[int, int]]) -> list[int]:
    out: list[int] = []
    if driver_name in {'X', 'X_PLUS_XY_FLUX', 'X_PLUS_XY_RAND'}:
        out.extend(state ^ (1 << q) for q in range(k))
    if driver_name in {'XY_FLUX', 'X_PLUS_XY_FLUX', 'X_PLUS_XY_RAND'}:
        for i, j in xy_edges:
            if ((state >> i) & 1) != ((state >> j) & 1):
                out.append(state ^ (1 << i) ^ (1 << j))
    return out


def graph_distance(start: int, target: int, k: int, driver_name: str, xy_edges: list[tuple[int, int]]) -> int:
    if start == target:
        return 0
    dim = 1 << k
    seen = np.zeros(dim, dtype=bool)
    seen[start] = True
    q = deque([(start, 0)])
    while q:
        u, d = q.popleft()
        for v in neighbors_for_driver(u, k, driver_name, xy_edges):
            if seen[v]:
                continue
            if v == target:
                return d + 1
            seen[v] = True
            q.append((v, d + 1))
    return -1


def static_transfer_driver(
    cube: dict[str, Any], Hdriver: sparse.csr_matrix,
    *, s: float, driver_scale: float, hold_max: float, hold_points: int,
) -> dict[str, Any]:
    problem, _ = v11.normalized_problem(cube)
    dim = len(problem)
    si, ti = int(cube['start_idx']), int(cube['target_idx'])
    feasible = np.asarray(cube['feasible'], dtype=bool)
    Eraw = np.asarray(cube['energies'], dtype=float)
    target_e = float(Eraw[ti])
    quality = feasible & (Eraw <= target_e + 1e-12)
    lower = feasible & (Eraw < Eraw[si] - 1e-12)

    Hp = sparse.diags(problem, offsets=0, format='csr')
    H = float(s) * Hp + float(driver_scale) * (1.0 - float(s)) * Hdriver
    psi0 = np.zeros(dim, dtype=np.complex128); psi0[si] = 1.0

    t0 = time.perf_counter()
    psis = expm_multiply((-1j) * H, psi0, start=0.0, stop=float(hold_max),
                        num=int(hold_points), endpoint=True)
    times = np.linspace(0.0, float(hold_max), int(hold_points))
    probs = np.abs(psis) ** 2
    pt = probs[:, ti]
    pq = np.sum(probs[:, quality], axis=1)
    pl = np.sum(probs[:, lower], axis=1)
    pf = np.sum(probs[:, feasible], axis=1)
    it = int(np.argmax(pt)); iq = int(np.argmax(pq))
    return {
        'max_p_target': float(pt[it]), 'time_of_max_target': float(times[it]),
        'max_p_quality': float(pq[iq]), 'time_of_max_quality': float(times[iq]),
        'max_p_any_lower_feasible': float(np.max(pl)),
        'mean_p_target': float(np.mean(pt[1:] if len(pt)>1 else pt)),
        'mean_p_quality': float(np.mean(pq[1:] if len(pq)>1 else pq)),
        'max_p_feasible': float(np.max(pf)),
        'final_p_target': float(pt[-1]),
        'wall_ms': 1000.0 * (time.perf_counter() - t0),
    }


def build_drivers(case: dict[str, Any], seed: int) -> dict[str, dict[str, Any]]:
    cube = case['cube']; terms = case['terms']; k = int(cube['k'])
    fedges = conflict_edges(terms)
    redges = random_edges(k, len(fedges), seed)

    X = x_driver(k)
    XYf = xy_exchange_driver(k, fedges)
    XYr = xy_exchange_driver(k, redges)

    raw = {
        'X': (X, []),
        'XY_FLUX': (XYf, fedges),
        'X_PLUS_XY_FLUX': (X + XYf, fedges),
        'X_PLUS_XY_RAND': (X + XYr, redges),
    }
    out: dict[str, dict[str, Any]] = {}
    for name, (H, edges) in raw.items():
        Hn, nrm = normalize_driver(H.tocsr())
        out[name] = {'H': Hn, 'raw_norm': nrm, 'xy_edges': edges}
    return out


def summarize(best_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[int, dict[str, dict[str, Any]]] = {}
    for r in best_rows:
        by_case.setdefault(int(r['case_id']), {})[str(r['driver'])] = r

    comparisons: list[dict[str, Any]] = []
    structured_wins_x = 0
    structured_wins_rand = 0
    structured_wins_both = 0
    ratios_x: list[float] = []
    ratios_rand: list[float] = []
    distance_reductions: list[float] = []
    for ci, dd in sorted(by_case.items()):
        if 'X_PLUS_XY_FLUX' not in dd or 'X' not in dd or 'X_PLUS_XY_RAND' not in dd:
            continue
        sf, x, rr = dd['X_PLUS_XY_FLUX'], dd['X'], dd['X_PLUS_XY_RAND']
        ps = float(sf['max_p_target']); px = float(x['max_p_target']); pr = float(rr['max_p_target'])
        rx = ps / max(px, 1e-15); rrati = ps / max(pr, 1e-15)
        ratios_x.append(rx); ratios_rand.append(rrati)
        wx = ps > px * 1.05; wr = ps > pr * 1.05
        structured_wins_x += int(wx); structured_wins_rand += int(wr); structured_wins_both += int(wx and wr)
        gd_x = int(x['driver_graph_distance']); gd_s = int(sf['driver_graph_distance'])
        if gd_x > 0 and gd_s >= 0:
            distance_reductions.append((gd_x - gd_s) / gd_x)
        comparisons.append({
            'case_id': ci, 'k': int(sf['k']), 'barrier_over_gain': float(sf['barrier_over_gain']),
            'p_target_X': px, 'p_target_structured': ps, 'p_target_random_control': pr,
            'structured_over_X': rx, 'structured_over_random': rrati,
            'graph_distance_X': gd_x, 'graph_distance_structured': gd_s,
        })

    return {
        'n_cases_compared': len(comparisons),
        'structured_beats_X_cases_gt5pct': structured_wins_x,
        'structured_beats_random_cases_gt5pct': structured_wins_rand,
        'structured_beats_both_cases_gt5pct': structured_wins_both,
        'median_structured_over_X': float(np.median(ratios_x)) if ratios_x else math.nan,
        'median_structured_over_random': float(np.median(ratios_rand)) if ratios_rand else math.nan,
        'median_fractional_driver_graph_distance_reduction_vs_X': float(np.median(distance_reductions)) if distance_reductions else math.nan,
        'comparisons': comparisons,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--v8-results', default='experiments/results/quantum_hardness_hunt_v8')
    ap.add_argument('--v11-results', default='experiments/results/quantum_mechanism_v11b')
    ap.add_argument('--out', default='experiments/results/structured_quantum_driver_v13')
    ap.add_argument('--mode', choices=['quick','full'], default='quick')
    ap.add_argument('--seed', type=int, default=131313)
    ap.add_argument('--nearest-seconds', type=float, default=8.0)
    args = ap.parse_args()

    v8dir = Path(args.v8_results); v11dir = Path(args.v11_results); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cases = v12.reconstruct_cases(v8dir, v11dir, args.nearest_seconds, args.seed)
    # Keep exact-statevector-safe natural cases only.
    cases = [c for c in cases if int(c['cube']['k']) <= 12]

    if args.mode == 'quick':
        s_values = [0.30, 0.50, 0.70]
        driver_scales = [0.75, 1.5]
        hold_max, hold_points = 24.0, 81
    else:
        s_values = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
        driver_scales = [0.5, 0.75, 1.0, 1.5, 2.0]
        hold_max, hold_points = 40.0, 161

    rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []

    for ci, c in enumerate(cases):
        cube = c['cube']; k = int(cube['k'])
        p_uniform = 1.0 / float(1 << k)
        drivers = build_drivers(c, args.seed + 1000 * ci)
        si, ti = int(cube['start_idx']), int(cube['target_idx'])

        print(f"[V13] case={c['case_id']} k={k} B/gain={c['geom']['barrier_over_gain']:.3g}", flush=True)
        for dname, dobj in drivers.items():
            edges = dobj['xy_edges']
            gd = graph_distance(si, ti, k, dname, edges)
            d_best: dict[str, Any] | None = None
            for s in s_values:
                for g in driver_scales:
                    z = static_transfer_driver(
                        cube, dobj['H'], s=s, driver_scale=g,
                        hold_max=hold_max, hold_points=hold_points,
                    )
                    r = {
                        'case_id': c['case_id'], 'set_name': c['set_name'],
                        'instance_index': c['instance_index'], 'k': k,
                        'barrier': c['geom']['barrier'], 'gain': c['geom']['gain'],
                        'barrier_over_gain': c['geom']['barrier_over_gain'],
                        'driver': dname, 'driver_raw_norm': dobj['raw_norm'],
                        'n_xy_edges': len(edges), 'driver_graph_distance': gd,
                        's': s, 'driver_scale': g,
                        'p_uniform_target': p_uniform,
                        **z,
                    }
                    r['target_enrichment_vs_uniform'] = r['max_p_target'] / p_uniform
                    rows.append(r)
                    if d_best is None or float(r['max_p_target']) > float(d_best['max_p_target']):
                        d_best = r.copy()
            assert d_best is not None
            best_rows.append(d_best)

    write_csv(out / 'structured_driver_sweep.csv', rows)
    write_csv(out / 'structured_driver_best.csv', best_rows)
    summary = summarize(best_rows)
    summary['mode'] = args.mode
    summary['n_cases'] = len(cases)
    summary['interpretation_guardrail'] = (
        'Positive structured-driver results are mechanism evidence for problem-aware coherent exchange mixers; '
        'they are not an end-to-end quantum speedup claim. The random-edge control tests whether Flux structure, '
        'rather than simply adding two-qubit couplings, is responsible.'
    )
    write_json(out / 'committee_summary.json', summary)
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)


if __name__ == '__main__':
    main()
