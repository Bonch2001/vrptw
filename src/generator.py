"""
Solution generation heuristics: random, greedy, etc.
"""

import random
from typing import List, Optional
from src.instance import Instance
from src.validator import check_route


def random_solution(
    inst: Instance,
    seed: Optional[int] = None,
    use_tw: bool = True,
    fill_ratio: float = 0.65
) -> List[List[int]]:
    """
    Generate a random solution by shuffling clients and packing into routes.
    
    Construction uses cap_build = capacity * fill_ratio to leave slack,
    but feasibility check always uses full capacity.
    
    Args:
        inst: Instance object
        seed: Random seed
        use_tw: Whether to enforce time window constraints
        fill_ratio: Load fraction to use during construction (default 0.65)
    
    Returns:
        List of routes (each route is a list of client IDs)
    """
    rng = random.Random(seed)
    clients = list(range(1, inst.n))
    rng.shuffle(clients)

    cap_build = inst.capacity * fill_ratio
    routes: List[List[int]] = []
    cur: List[int] = []
    cur_load = 0

    for c in clients:
        # Try adding c to current route
        if cur_load + inst.demand[c] <= cap_build:
            trial = cur + [c]
            feas, *_ = check_route(inst, trial, use_tw=use_tw)
            if feas:
                cur = trial
                cur_load += inst.demand[c]
                continue

        # Close current route and start new one
        if cur:
            routes.append(cur)
        
        cur = [c]
        cur_load = inst.demand[c]

        # Check if single-client route is feasible
        feas, *_ = check_route(inst, cur, use_tw=use_tw)
        if not feas:
            # TODO: Repair logic - for now, skip or use fallback
            # In practice: try inserting into previous route, or split demand
            print(f"  Failed: TW violation for client {c}")  # Add this
            pass

    if cur:
        routes.append(cur)

    return routes


def greedy_nearest_neighbor(
    inst: Instance,
    start_node: int = 0,
    use_tw: bool = False,
    fill_ratio: float = 0.7
) -> List[List[int]]:
    """
    Greedy nearest neighbor heuristic: build routes by always visiting
    the nearest unserved client that fits in the current route.
    
    Args:
        inst: Instance object
        start_node: Depot node (default 0)
        use_tw: Whether to enforce time window constraints
    
    Returns:
        List of routes
    """
    unserved = set(range(1, inst.n))
    cap_build = inst.capacity * fill_ratio
    routes: List[List[int]] = []

    while unserved:
        route: List[int] = []
        route_load = 0
        current = start_node

        while unserved:
            # Find nearest unserved client that fits
            best_client = None
            best_dist = float("inf")

            for c in unserved:
                if route_load + inst.demand[c] <= cap_build:
                    trial = route + [c]
                    feas, *_ = check_route(inst, trial, use_tw=use_tw)
                    if feas:
                        d = inst.dist(current, c)
                        if d < best_dist:
                            best_dist = d
                            best_client = c

            if best_client is None:
                # No more clients fit in this route
                print(f"No client fits in route {len(routes)+1}, closing with {len(route)} clients")
                break

            route.append(best_client)
            route_load += inst.demand[best_client]
            unserved.remove(best_client)
            current = best_client

        if route:
            routes.append(route)

    return routes
