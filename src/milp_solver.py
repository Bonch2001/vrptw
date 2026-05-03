import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from src.instance import Instance
from src.utils import calculate_route_cost


def _build_arc_index(nodes: List[int]) -> Tuple[Dict[Tuple[int, int], int], List[Tuple[int, int]]]:
    """
    Build a variable index for directed arcs x_ij with i != j.
    """
    arc_index = {}
    arcs = []
    idx = 0
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            arc_index[(i, j)] = idx
            arcs.append((i, j))
            idx += 1
    return arc_index, arcs


def _reconstruct_routes(
    x_values: np.ndarray,
    arc_index: Dict[Tuple[int, int], int],
    clients: List[int]
) -> List[List[int]]:
    """
    Reconstruct routes from binary arc values.
    """
    succ = {}
    starts = []

    for (i, j), idx in arc_index.items():
        if x_values[idx] > 0.5:
            succ[i] = j
            if i == 0:
                starts.append(j)

    routes = []
    covered = []

    for start in starts:
        route = []
        current = start
        seen = set()

        while current != 0 and current not in seen:
            route.append(current)
            covered.append(current)
            seen.add(current)
            current = succ.get(current, 0)

        if route:
            routes.append(route)

    if sorted(covered) != sorted(clients):
        return []

    return routes


def _solve_fixed_k_cvrp(
    inst: Instance,
    k_vehicles: int,
    time_limit: Optional[float] = None,
    mip_gap: float = 0.0
) -> Tuple[Optional[List[List[int]]], Optional[float]]:
    """
    Solve the CVRP exactly for a fixed number of vehicles K
    using a MILP formulation with MTZ-like load variables.

    Variables:
        x_ij in {0,1}: arc i -> j is used
        u_i continuous: cumulated load after visiting client i

    Returns:
        (routes, total_distance) if a feasible solution is found,
        (None, None) otherwise.
    """
    nodes = list(range(inst.n))
    clients = list(range(1, inst.n))
    Q = inst.capacity

    arc_index, arcs = _build_arc_index(nodes)
    num_x = len(arcs)

    u_index = {}
    offset = num_x
    for pos, client in enumerate(clients):
        u_index[client] = offset + pos

    num_u = len(clients)
    num_vars = num_x + num_u

    c = np.zeros(num_vars, dtype=float)
    integrality = np.zeros(num_vars, dtype=int)

    lower_bounds = np.zeros(num_vars, dtype=float)
    upper_bounds = np.full(num_vars, np.inf, dtype=float)

    # x variables
    for (i, j), idx in arc_index.items():
        c[idx] = inst.dist(i, j)
        integrality[idx] = 1  # binary/integer
        lower_bounds[idx] = 0.0
        upper_bounds[idx] = 1.0

    # u variables
    for client in clients:
        idx = u_index[client]
        lower_bounds[idx] = inst.demand[client]
        upper_bounds[idx] = Q

    rows = []
    cols = []
    data = []
    lhs = []
    rhs = []

    row_id = 0

    def add_coeff(r: int, var_idx: int, value: float) -> None:
        rows.append(r)
        cols.append(var_idx)
        data.append(value)

    # 1) Each client has exactly one outgoing arc
    for i in clients:
        for j in nodes:
            if i == j:
                continue
            add_coeff(row_id, arc_index[(i, j)], 1.0)
        lhs.append(1.0)
        rhs.append(1.0)
        row_id += 1

    # 2) Each client has exactly one incoming arc
    for j in clients:
        for i in nodes:
            if i == j:
                continue
            add_coeff(row_id, arc_index[(i, j)], 1.0)
        lhs.append(1.0)
        rhs.append(1.0)
        row_id += 1

    # 3) Depot out-degree = K
    for j in clients:
        add_coeff(row_id, arc_index[(0, j)], 1.0)
    lhs.append(float(k_vehicles))
    rhs.append(float(k_vehicles))
    row_id += 1

    # 4) Depot in-degree = K
    for i in clients:
        add_coeff(row_id, arc_index[(i, 0)], 1.0)
    lhs.append(float(k_vehicles))
    rhs.append(float(k_vehicles))
    row_id += 1

    # 5) MTZ load constraints:
    #    u_i - u_j + Q * x_ij <= Q - demand_j
    for i in clients:
        for j in clients:
            if i == j:
                continue
            add_coeff(row_id, u_index[i], 1.0)
            add_coeff(row_id, u_index[j], -1.0)
            add_coeff(row_id, arc_index[(i, j)], float(Q))
            lhs.append(-np.inf)
            rhs.append(float(Q - inst.demand[j]))
            row_id += 1

    A = coo_matrix((data, (rows, cols)), shape=(row_id, num_vars)).tocsr()
    constraints = LinearConstraint(A, np.array(lhs, dtype=float), np.array(rhs, dtype=float))
    bounds = Bounds(lower_bounds, upper_bounds)

    options = {
        "disp": False,
        "mip_rel_gap": float(mip_gap),
    }
    if time_limit is not None:
        options["time_limit"] = float(time_limit)

    result = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=[constraints],
        options=options,
    )

    if result.x is None:
        return None, None

    routes = _reconstruct_routes(result.x[:num_x], arc_index, clients)
    if not routes:
        return None, None

    total_distance = sum(calculate_route_cost(inst, route) for route in routes)
    return routes, total_distance


def solve_cvrp_milp(
    inst: Instance,
    max_vehicles: Optional[int] = 20,
    time_limit: Optional[float] = 60.0,
    mip_gap: float = 0.0,
    return_history: bool = False
):
    """
    Solve the CVRP exactly by searching for the minimum feasible number
    of vehicles, then minimizing total distance for that number.

    This implementation handles capacity constraints only.
    Use it with --no-time-windows.

    Args:
        inst: Problem instance.
        max_vehicles: Optional upper bound on the number of vehicles.
        time_limit: Optional per-MILP time limit in seconds.
        mip_gap: Relative MIP gap for the solver.
        return_history: If True, also return a simple history.

    Returns:
        If return_history is False:
            routes
        If return_history is True:
            (routes, history)
    """
    clients = list(range(1, inst.n))
    if not clients:
        if return_history:
            return [], [(0, 0.0)]
        return []

    total_demand = sum(inst.demand[i] for i in clients)
    lower_k = math.ceil(total_demand / inst.capacity)
    upper_k = max_vehicles if max_vehicles is not None else len(clients)

    history = []

    for k in range(lower_k, upper_k + 1):
        routes, total_distance = _solve_fixed_k_cvrp(
            inst,
            k_vehicles=k,
            time_limit=time_limit,
            mip_gap=mip_gap
        )

        if routes is not None:
            history.append((len(routes), total_distance))
            if return_history:
                return routes, history
            return routes

        history.append((k, float("inf")))

    raise RuntimeError(
        "MILP did not find a feasible solution in the tested vehicle range. "
        "Try a smaller instance or increase --milp-max-vehicles / --milp-time-limit."
    )