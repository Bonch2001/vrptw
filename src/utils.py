"""
Utility functions: logging, output formatting, etc.
"""

import logging
from typing import List
from src.instance import Instance
from src.metrics import analyze_solution


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def print_solution(
    inst: Instance,
    routes: List[List[int]],
    is_valid: bool,
    total_cost: float,
    use_tw: bool = False
) -> None:
    """
    Print a richer solution summary and detailed per-route metrics.
    """
    metrics = analyze_solution(inst, routes, use_tw=use_tw)

    print("\n" + "=" * 100)
    print("SOLUTION SUMMARY")
    print("=" * 100)

    status = "✓ FEASIBLE" if is_valid else "✗ INFEASIBLE"
    print(f"Status: {status}")
    print(f"Number of routes: {metrics.num_routes}")
    print(f"Total cost: {metrics.total_distance:.2f}")
    print(f"Total demand served: {metrics.total_demand} / {sum(inst.demand[1:])}")
    print(f"Average route distance: {metrics.avg_route_distance:.2f}")
    print(f"Maximum route distance: {metrics.max_route_distance:.2f}")
    print(f"Average load ratio: {metrics.avg_load_ratio:.2%}")
    print(f"Maximum load ratio: {metrics.max_load_ratio:.2%}")
    print(f"Average clients per route: {metrics.avg_clients_per_route:.2f}")

    if use_tw:
        print(f"Average end time: {metrics.avg_end_time:.2f}")
        print(f"Minimum depot slack: {metrics.min_depot_slack:.2f}")
        print(f"Average depot slack: {metrics.avg_depot_slack:.2f}")

    print()
    print("-" * 100)
    print("PER-ROUTE DETAILS")
    print("-" * 100)

    for rm, route in zip(metrics.route_metrics, routes):
        route_str = " -> ".join(map(str, [0] + route + [0]))
        base = (
            f"Route {rm.route_index:2d}: {route_str:40s} | "
            f"Clients: {rm.num_clients:2d} | "
            f"Dist: {rm.distance:8.2f} | "
            f"Load: {rm.load:4d} ({rm.load_ratio:6.2%})"
        )

        if use_tw:
            base += (
                f" | End: {rm.end_time:8.2f}"
                f" | Slack: {rm.depot_slack:8.2f}"
            )

        print(base)

    print("=" * 100 + "\n")

def calculate_route_cost(inst: Instance, route: List[int]) -> float:
    """
    Compute the total travel distance of a single route.
    """
    if not route:
        return 0.0
    
    cost = 0.0
    prev = 0
    for c in route:
        cost += inst.dist(prev, c)
        prev = c
    cost += inst.dist(prev, 0)
    
    return cost
