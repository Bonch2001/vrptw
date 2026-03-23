"""
Route validation and feasibility checking.
"""

from typing import List, Tuple
from src.instance import Instance


def check_route(
    inst: Instance,
    route: List[int],
    use_tw: bool = True
) -> Tuple[bool, float, float, int]:
    """
    Check feasibility of a single route.
    
    Args:
        inst: Instance object
        route: List of client IDs (depot 0 not included)
        use_tw: Whether to enforce time window constraints
    
    Returns:
        Tuple of (feasible, total_distance, end_time_at_depot, load)
        - feasible: True if route is valid
        - total_distance: Total distance traveled (route + return to depot)
        - end_time_at_depot: Time of return to depot
        - load: Total demand on route
    """
    # Check capacity
    load = sum(inst.demand[c] for c in route)
    if load > inst.capacity:
        return False, float("inf"), float("inf"), load

    total_dist = 0.0
    t = inst.ready[0]  # start time at depot
    prev = 0

    # Traverse route
    for c in route:
        # Travel to node c
        d = inst.dist(prev, c)
        total_dist += d
        
        if use_tw:
            t = t + inst.travel_time(prev, c)
            # Check time window at arrival
            if t < inst.ready[c]:
                t = inst.ready[c]  # wait until ready time
            if t > inst.due[c]:
                return False, float("inf"), float("inf"), load
            # Service at node c
            t += inst.service[c]
        
        prev = c

    # Return to depot
    d = inst.dist(prev, 0)
    total_dist += d
    if use_tw:
        t = t + inst.travel_time(prev, 0)
        # Check depot time window
        if t > inst.due[0]:
            return False, float("inf"), float("inf"), load

    if total_dist == float("inf"):
        return False, float("inf"), t, load

    return True, total_dist, t, load


def validate_solution(
    inst: Instance,
    routes: List[List[int]],
    use_tw: bool = True
) -> Tuple[bool, float, int]:
    """
    Validate a complete solution.
    
    Args:
        inst: Instance object
        routes: List of routes (each route is a list of client IDs)
        use_tw: Whether to enforce time window constraints
    
    Returns:
        Tuple of (is_valid, total_cost, num_routes)
    """
    visited = set()
    total_cost = 0.0
    
    for route in routes:
        # Check for duplicate clients
        for c in route:
            if c in visited:
                return False, float("inf"), len(routes)
            if c < 1 or c >= inst.n:
                return False, float("inf"), len(routes)
            visited.add(c)
        
        # Check route feasibility
        feas, dist, _, _ = check_route(inst, route, use_tw=use_tw)
        if not feas:
            if not feas or dist == float("inf"):
                return False, float("inf"), len(routes)
        total_cost += dist
    
    # Check all clients are served
    if len(visited) != inst.n - 1:
        return False, float("inf"), len(routes)
    
    return True, total_cost, len(routes)
