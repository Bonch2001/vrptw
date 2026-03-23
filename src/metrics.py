from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple
import csv
import json
from pathlib import Path

from src.instance import Instance
from src.validator import check_route


@dataclass
class RouteMetrics:
    route_index: int
    num_clients: int
    distance: float
    load: int
    load_ratio: float
    end_time: float
    depot_slack: float


@dataclass
class SolutionMetrics:
    num_routes: int
    total_distance: float
    total_demand: int
    avg_route_distance: float
    max_route_distance: float
    avg_load_ratio: float
    max_load_ratio: float
    avg_clients_per_route: float
    avg_end_time: float
    min_depot_slack: float
    avg_depot_slack: float
    route_metrics: List[RouteMetrics]


@dataclass
class RunSummary:
    instance_name: str
    method: str
    generator: str
    use_time_windows: bool
    params: Dict[str, Any]

    initial_routes: int
    initial_distance: float
    final_routes: int
    final_distance: float

    delta_routes: int
    delta_distance: float
    distance_improvement_pct: float

    runtime_seconds: float
    history_length: int

    counters: Dict[str, int]


def analyze_solution(inst: Instance, routes: List[List[int]], use_tw: bool = False) -> SolutionMetrics:
    """
    Compute detailed metrics for a complete solution.

    For each route, compute:
    - distance
    - load and load ratio
    - end time at depot
    - depot slack = due_depot - end_time

    Then aggregate these statistics over the whole solution.
    """
    route_infos: List[RouteMetrics] = []
    total_distance = 0.0
    total_demand = 0

    depot_due = inst.due[0] if use_tw else 0.0

    for idx, route in enumerate(routes, start=1):
        feas, dist, end_time, load = check_route(inst, route, use_tw=use_tw)
        if not feas:
            # keep values but mark impossible slack if route invalid
            dist = float("inf")
            end_time = float("inf")

        load_ratio = (load / inst.capacity) if inst.capacity > 0 else 0.0
        depot_slack = (depot_due - end_time) if use_tw and end_time != float("inf") else 0.0

        info = RouteMetrics(
            route_index=idx,
            num_clients=len(route),
            distance=dist,
            load=load,
            load_ratio=load_ratio,
            end_time=end_time,
            depot_slack=depot_slack
        )
        route_infos.append(info)

        total_distance += dist
        total_demand += load

    num_routes = len(route_infos)

    if num_routes == 0:
        return SolutionMetrics(
            num_routes=0,
            total_distance=0.0,
            total_demand=0,
            avg_route_distance=0.0,
            max_route_distance=0.0,
            avg_load_ratio=0.0,
            max_load_ratio=0.0,
            avg_clients_per_route=0.0,
            avg_end_time=0.0,
            min_depot_slack=0.0,
            avg_depot_slack=0.0,
            route_metrics=[]
        )

    distances = [r.distance for r in route_infos]
    load_ratios = [r.load_ratio for r in route_infos]
    clients_per_route = [r.num_clients for r in route_infos]
    end_times = [r.end_time for r in route_infos if r.end_time != float("inf")]
    depot_slacks = [r.depot_slack for r in route_infos if r.end_time != float("inf")]

    return SolutionMetrics(
        num_routes=num_routes,
        total_distance=total_distance,
        total_demand=total_demand,
        avg_route_distance=sum(distances) / num_routes,
        max_route_distance=max(distances),
        avg_load_ratio=sum(load_ratios) / num_routes,
        max_load_ratio=max(load_ratios),
        avg_clients_per_route=sum(clients_per_route) / num_routes,
        avg_end_time=(sum(end_times) / len(end_times)) if end_times else 0.0,
        min_depot_slack=min(depot_slacks) if depot_slacks else 0.0,
        avg_depot_slack=(sum(depot_slacks) / len(depot_slacks)) if depot_slacks else 0.0,
        route_metrics=route_infos
    )


def best_so_far_history(history: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """
    Convert a current accepted history into a best-so-far history
    using lexicographic minimization.
    """
    if not history:
        return []

    best = history[0]
    out = [best]

    for h in history[1:]:
        if h < best:
            best = h
        out.append(best)

    return out


def build_run_summary(
    instance_name: str,
    method: str,
    generator: str,
    use_time_windows: bool,
    params: Dict[str, Any],
    initial_metrics: SolutionMetrics,
    final_metrics: SolutionMetrics,
    runtime_seconds: float,
    history_length: int,
    counters: Dict[str, int]
) -> RunSummary:
    """
    Build a compact run summary for saving and comparison.
    """
    delta_routes = initial_metrics.num_routes - final_metrics.num_routes
    delta_distance = initial_metrics.total_distance - final_metrics.total_distance

    if initial_metrics.total_distance > 0:
        pct = 100.0 * delta_distance / initial_metrics.total_distance
    else:
        pct = 0.0

    return RunSummary(
        instance_name=instance_name,
        method=method,
        generator=generator,
        use_time_windows=use_time_windows,
        params=params,
        initial_routes=initial_metrics.num_routes,
        initial_distance=initial_metrics.total_distance,
        final_routes=final_metrics.num_routes,
        final_distance=final_metrics.total_distance,
        delta_routes=delta_routes,
        delta_distance=delta_distance,
        distance_improvement_pct=pct,
        runtime_seconds=runtime_seconds,
        history_length=history_length,
        counters=counters
    )


def save_run_summary_json(summary: RunSummary, save_path: str) -> None:
    """
    Save a run summary as JSON.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)


def append_run_summary_csv(summary: RunSummary, save_path: str) -> None:
    """
    Append a flattened run summary to a CSV file.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = asdict(summary)

    # flatten params
    params = row.pop("params", {})
    counters = row.pop("counters", {})

    for k, v in params.items():
        row["param_" + str(k)] = v
    for k, v in counters.items():
        row["count_" + str(k)] = v

    write_header = not path.exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)