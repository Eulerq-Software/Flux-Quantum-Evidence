from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTANCE_ROOT = REPO_ROOT / "instances" / "raw"


@dataclass
class ProblemData:
    """
    Public representation of a derived Flux candidate-selection problem.

    This contains only the combinatorial optimization problem used by the
    experiments. It does not contain the proprietary candidate-generation
    or routing implementation.
    """

    instance_id: str
    source_set: str
    source_index: int

    num_riders: int
    num_orders: int

    rider: np.ndarray
    o1: np.ndarray
    o2: np.ndarray
    count: np.ndarray
    distance: np.ndarray

    M: float
    raw_reward: np.ndarray
    lam: float
    reward: np.ndarray

    # Candidate indices grouped by conflicting resource.
    by_rider: tuple[np.ndarray, ...]
    by_order: tuple[np.ndarray, ...]

    @property
    def num_candidates(self) -> int:
        return len(self.rider)


def _build_groups(
    rider: np.ndarray,
    o1: np.ndarray,
    o2: np.ndarray,
    num_riders: int,
    num_orders: int,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """
    Reconstruct candidate conflict buckets.

    Candidates conflict when they use the same rider or share an order.
    """

    rider_groups: list[list[int]] = [
        [] for _ in range(num_riders)
    ]

    order_groups: list[list[int]] = [
        [] for _ in range(num_orders)
    ]

    for i in range(len(rider)):
        r = int(rider[i])
        a = int(o1[i])
        b = int(o2[i])

        rider_groups[r].append(i)
        order_groups[a].append(i)

        if b >= 0:
            order_groups[b].append(i)

    by_rider = tuple(
        np.asarray(group, dtype=np.int32)
        for group in rider_groups
    )

    by_order = tuple(
        np.asarray(group, dtype=np.int32)
        for group in order_groups
    )

    return by_rider, by_order


def _validate_problem(problem: ProblemData) -> None:
    m = problem.num_candidates

    arrays = {
        "rider": problem.rider,
        "o1": problem.o1,
        "o2": problem.o2,
        "count": problem.count,
        "distance": problem.distance,
        "raw_reward": problem.raw_reward,
        "reward": problem.reward,
    }

    for name, arr in arrays.items():
        if len(arr) != m:
            raise ValueError(
                f"{name} has length {len(arr)}, "
                f"expected {m}"
            )

    if np.any(problem.rider < 0):
        raise ValueError("Negative rider index found.")

    if np.any(problem.rider >= problem.num_riders):
        raise ValueError("Rider index exceeds num_riders.")

    if np.any(problem.o1 < 0):
        raise ValueError("Primary order index cannot be negative.")

    if np.any(problem.o1 >= problem.num_orders):
        raise ValueError("Primary order index exceeds num_orders.")

    valid_o2 = problem.o2 >= 0

    if np.any(problem.o2[valid_o2] >= problem.num_orders):
        raise ValueError("Secondary order index exceeds num_orders.")

    if np.any(
        ~np.isin(problem.count, np.asarray([1, 2]))
    ):
        raise ValueError(
            "Candidate count must be either 1 or 2."
        )

    if problem.lam <= 0:
        raise ValueError("QUBO penalty scale lam must be positive.")

    expected_reward = problem.raw_reward / problem.lam

    if not np.allclose(
        problem.reward,
        expected_reward,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "reward is inconsistent with raw_reward / lam."
        )


def load_problem(path: str | Path) -> ProblemData:
    """
    Load one exported candidate-selection problem.
    """

    path = Path(path)

    with np.load(path, allow_pickle=False) as data:
        rider = data["rider"].astype(np.int32)
        o1 = data["o1"].astype(np.int32)
        o2 = data["o2"].astype(np.int32)

        num_riders = int(data["num_riders"])
        num_orders = int(data["num_orders"])

        by_rider, by_order = _build_groups(
            rider=rider,
            o1=o1,
            o2=o2,
            num_riders=num_riders,
            num_orders=num_orders,
        )

        problem = ProblemData(
            instance_id=str(
                data["instance_id"].item()
            ),
            source_set=str(
                data["source_set"].item()
            ),
            source_index=int(
                data["source_index"]
            ),

            num_riders=num_riders,
            num_orders=num_orders,

            rider=rider,
            o1=o1,
            o2=o2,
            count=data["count"].astype(np.int8),
            distance=data["distance"].astype(
                np.float64
            ),

            M=float(data["M"]),
            raw_reward=data["raw_reward"].astype(
                np.float64
            ),
            lam=float(data["lam"]),
            reward=data["reward"].astype(
                np.float64
            ),

            by_rider=by_rider,
            by_order=by_order,
        )

    _validate_problem(problem)

    return problem


def list_problem_files(
    set_name: str,
) -> list[Path]:
    """
    Return all exported instances in a named public set.
    """

    directory = INSTANCE_ROOT / set_name

    if not directory.exists():
        raise FileNotFoundError(
            f"Problem set does not exist: {directory}"
        )

    files = sorted(
        directory.glob("instance_*.npz")
    )

    if not files:
        raise FileNotFoundError(
            f"No instance_*.npz files found in "
            f"{directory}"
        )

    return files


def load_problem_set(
    set_name: str,
) -> list[ProblemData]:
    """
    Load a complete named public problem set.
    """

    return [
        load_problem(path)
        for path in list_problem_files(set_name)
    ]