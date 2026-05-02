import math
import random
from typing import Dict, List, Optional, Tuple
from src.instance import Instance
from src.validator import check_route
from src.utils import calculate_route_cost


def intra_2opt(inst: Instance, route: List[int], use_tw: bool = False, current_routes: List[List[int]] = None, route_index: int = None, history: Optional[list] = None) -> List[int]:
    """
    Improve a single route using the 2-opt neighborhood.

    The operator selects two positions in the route and reverses the
    intermediate segment. Only feasible improving moves are accepted.

    This is an intra-route operator: it changes the visit order inside
    one route but does not move clients between vehicles.

    Args:
        inst: Problem instance.
        route: One route represented as a list of client IDs.
        use_tw: If True, time-window feasibility is checked.

    Returns:
        A locally improved version of the input route.
    """
    if len(route) < 3:
        return route[:]
    
    improved = True
    best_route = route[:]
    
    while improved:
        improved = False
        _, best_dist, _, _ = check_route(inst, best_route, use_tw=use_tw)        
        # Try all segment reversals
        for i in range(len(best_route)):
            for j in range(i + 2, len(best_route)):
                # Reverse segment [i+1:j+1]
                new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                
                # Check feasibility
                feas, dist, _, _ = check_route(inst, new_route, use_tw=use_tw)
                
                if feas and dist < best_dist:
                    best_route = new_route
                    best_dist = dist
                    improved = True
                    if history is not None and current_routes is not None and route_index is not None:
                        candidate_routes = [r[:] for r in current_routes]
                        candidate_routes[route_index] = new_route
                        history.append(solution_score(inst, candidate_routes))
                    break
            
            if improved:
                break
    
    return best_route

def inter_relocate(inst: Instance, routes: List[List[int]], use_tw: bool = False, history: Optional[list] = None) -> List[List[int]]:
    """    
    Apply an inter-route relocate neighborhood on a complete solution.

    The operator removes one client from a route A and tries to insert it
    into every possible position of another route B. A move is accepted if:
    - both modified routes remain feasible,
    - and the global solution score improves.

    A route that becomes empty after relocation is removed automatically,
    which may reduce the number of vehicles.

    Args:
        inst: Problem instance.
        routes: Current solution as a list of routes.
        use_tw: If True, feasibility is checked with time windows.
                If False, only capacity is considered.
        history: Optional list used to store accepted global solution scores.

    Returns:
        An improved solution after repeated relocate moves until
        no improving move is found.
    """
    best_routes = [r[:] for r in routes if r]
    improved = True

    while improved:
        improved = False
        best_score = solution_score(inst, best_routes)

        for i in range(len(best_routes)):
            route_a = best_routes[i]

            for client_idx in range(len(route_a)):
                client = route_a[client_idx]
                new_route_a = route_a[:client_idx] + route_a[client_idx + 1:]

                # route A after removal must stay feasible if non-empty
                if new_route_a:
                    feas_a, _, _, _ = check_route(inst, new_route_a, use_tw=use_tw)
                    if not feas_a:
                        continue

                for j in range(len(best_routes)):
                    if i == j:
                        continue

                    route_b = best_routes[j]

                    for insert_pos in range(len(route_b) + 1):
                        new_route_b = route_b[:insert_pos] + [client] + route_b[insert_pos:]

                        feas_b, _, _, _ = check_route(inst, new_route_b, use_tw=use_tw)
                        if not feas_b:
                            continue

                        # Build candidate solution
                        candidate = []
                        for k, r in enumerate(best_routes):
                            if k == i:
                                if new_route_a:
                                    candidate.append(new_route_a)
                            elif k == j:
                                candidate.append(new_route_b)
                            else:
                                candidate.append(r[:])

                        candidate_score = solution_score(inst, candidate)

                        if candidate_score < best_score:
                            best_routes = candidate
                            best_score = candidate_score
                            improved = True
                            if history is not None:
                                history.append(best_score)
                            break

                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return best_routes

def inter_exchange(
    inst: Instance,
    routes: List[List[int]],
    use_tw: bool = False,
    history: Optional[list] = None
) -> List[List[int]]:
    """
    Apply an inter-route exchange neighborhood on a complete solution.

    The operator swaps one client from route A with one client from route B.
    A move is accepted if:
    - both modified routes remain feasible,
    - and the global solution score improves.

    Args:
        inst: Problem instance.
        routes: Current solution as a list of routes.
        use_tw: If True, feasibility is checked with time windows.
        history: Optional list used to store accepted global solution scores.

    Returns:
        An improved solution after repeated exchange moves until
        no improving move is found.
    """
    best_routes = [r[:] for r in routes if r]
    improved = True

    while improved:
        improved = False
        best_score = solution_score(inst, best_routes)

        for i in range(len(best_routes)):
            route_a = best_routes[i]

            for j in range(i + 1, len(best_routes)):
                route_b = best_routes[j]

                for pos_a in range(len(route_a)):
                    client_a = route_a[pos_a]

                    for pos_b in range(len(route_b)):
                        client_b = route_b[pos_b]

                        new_route_a = route_a[:]
                        new_route_b = route_b[:]

                        new_route_a[pos_a] = client_b
                        new_route_b[pos_b] = client_a

                        feas_a, _, _, _ = check_route(inst, new_route_a, use_tw=use_tw)
                        if not feas_a:
                            continue

                        feas_b, _, _, _ = check_route(inst, new_route_b, use_tw=use_tw)
                        if not feas_b:
                            continue

                        candidate = []
                        for k, r in enumerate(best_routes):
                            if k == i:
                                candidate.append(new_route_a)
                            elif k == j:
                                candidate.append(new_route_b)
                            else:
                                candidate.append(r[:])

                        candidate_score = solution_score(inst, candidate)

                        if candidate_score < best_score:
                            best_routes = candidate
                            best_score = candidate_score
                            improved = True
                            if history is not None:
                                history.append(best_score)
                            break

                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return best_routes
    
def solution_score(inst: Instance, routes: List[List[int]]) -> Tuple[int, float]:
    """
    Compute the lexicographic score of a complete solution.

    The optimization goal of the project is:
    1. minimize the number of routes (vehicles),
    2. then minimize the total traveled distance.

    Empty routes are ignored in the evaluation.

    Args:
        inst: Problem instance.
        routes: List of routes, each route being a list of client IDs.

    Returns:
        A tuple (num_routes, total_distance), compared lexicographically.
        A smaller tuple corresponds to a better solution.
    """
    non_empty = [route for route in routes if route]
    total_distance = sum(calculate_route_cost(inst, route) for route in non_empty)
    return len(non_empty), total_distance

def local_search(inst: Instance, routes: List[List[int]], use_tw: bool = False, return_history: bool = False):
    """
    Improve a solution by alternating intra-route and inter-route operators.

    The search applies:
    - intra-route 2-opt to improve the order of clients inside each route,
    - inter-route relocate to move clients between routes and possibly
      reduce the number of vehicles,
    - inter-route exchange to swap clients between routes,
    - then 2-opt again to clean the modified routes.

    The process repeats while the global solution score improves.

    Args:
        inst: Problem instance.
        routes: Initial solution as a list of routes.
        use_tw: If True, time windows are enforced during feasibility checks.
        return_history: If True, also return the evolution of the score.

    Returns:
        If return_history is False:
            A locally improved solution.
        If return_history is True:
            A tuple (improved_solution, history), where history is a list
            of (num_routes, total_distance) scores.
    """
    current = [r[:] for r in routes]
    history = [solution_score(inst, current)]

    improved = True
    while improved:
        improved = False
        old_score = solution_score(inst, current)

        # 2-opt on each route
        for idx in range(len(current)):
            current[idx] = intra_2opt(
                inst,
                current[idx],
                use_tw=use_tw,
                current_routes=current,
                route_index=idx,
                history=history
            )

        # relocate
        current = inter_relocate(
            inst,
            current,
            use_tw=use_tw,
            history=history
        )

        # exchange
        current = inter_exchange(
            inst,
            current,
            use_tw=use_tw,
            history=history
        )

        # cleanup 2-opt again
        for idx in range(len(current)):
            current[idx] = intra_2opt(
                inst,
                current[idx],
                use_tw=use_tw,
                current_routes=current,
                route_index=idx,
                history=history
            )

        if solution_score(inst, current) < old_score:
            improved = True

    if return_history:
        return current, history
    return current

def compute_vehicle_penalty(inst: Instance, multiplier: float = 10.0) -> float:
    """
    Compute the constant M used in the scalar energy for simulated annealing.

    The energy is defined as:
        E(S) = M * K(S) + D(S)
    where:
        K(S) = number of routes,
        D(S) = total distance.

    The penalty M must be large enough so that reducing the number
    of vehicles dominates ordinary distance improvements.

    Args:
        inst: Problem instance.
        multiplier: Safety factor applied to the depot-client distance sum.

    Returns:
        A positive scalar penalty M.
    """
    base = sum(inst.dist(0, client_id) for client_id in range(1, inst.n))
    return multiplier * max(base, 1.0)


def solution_energy(
    inst: Instance,
    routes: List[List[int]],
    vehicle_penalty: Optional[float] = None
) -> float:
    """
    Compute the scalar energy used by simulated annealing.

    Args:
        inst: Problem instance.
        routes: Complete solution.
        vehicle_penalty: Optional penalty M. If None, it is computed automatically.

    Returns:
        Scalar energy value.
    """
    if vehicle_penalty is None:
        vehicle_penalty = compute_vehicle_penalty(inst)

    num_routes, total_distance = solution_score(inst, routes)
    return vehicle_penalty * num_routes + total_distance


def _random_2opt_neighbor(
    inst: Instance,
    routes: List[List[int]],
    rng: random.Random,
    use_tw: bool = False,
    max_attempts: int = 100
) -> Optional[List[List[int]]]:
    """
    Generate one random feasible neighbor using an intra-route 2-opt move.

    Args:
        inst: Problem instance.
        routes: Current solution.
        rng: Random generator.
        use_tw: Whether to enforce time windows.
        max_attempts: Number of attempts to sample a feasible move.

    Returns:
        A new candidate solution if a feasible move is found, otherwise None.
    """
    valid_route_indices = [idx for idx, route in enumerate(routes) if len(route) >= 3]
    if not valid_route_indices:
        return None

    for _ in range(max_attempts):
        route_idx = rng.choice(valid_route_indices)
        route = routes[route_idx]

        i = rng.randint(0, len(route) - 3)
        j = rng.randint(i + 2, len(route) - 1)

        new_route = route[:i + 1] + route[i + 1:j + 1][::-1] + route[j + 1:]

        feasible, _, _, _ = check_route(inst, new_route, use_tw=use_tw)
        if not feasible:
            continue

        candidate = [r[:] for r in routes]
        candidate[route_idx] = new_route
        return [r for r in candidate if r]

    return None

def _random_relocate_neighbor(
    inst: Instance,
    routes: List[List[int]],
    rng: random.Random,
    use_tw: bool = False,
    max_attempts: int = 100
) -> Optional[List[List[int]]]:
    """
    Generate one random feasible neighbor using an inter-route relocate move.

    A client is removed from one route and inserted into another route.

    Args:
        inst: Problem instance.
        routes: Current solution.
        rng: Random generator.
        use_tw: Whether to enforce time windows.
        max_attempts: Number of attempts to sample a feasible move.

    Returns:
        A new candidate solution if a feasible move is found, otherwise None.
    """
    if len(routes) < 2:
        return None

    source_indices = [idx for idx, route in enumerate(routes) if len(route) >= 1]
    if not source_indices:
        return None

    for _ in range(max_attempts):
        source_idx = rng.choice(source_indices)

        target_candidates = [idx for idx in range(len(routes)) if idx != source_idx]
        if not target_candidates:
            return None

        target_idx = rng.choice(target_candidates)

        source_route = routes[source_idx]
        target_route = routes[target_idx]

        client_pos = rng.randrange(len(source_route))
        client = source_route[client_pos]

        new_source = source_route[:client_pos] + source_route[client_pos + 1:]
        insert_pos = rng.randrange(len(target_route) + 1)
        new_target = target_route[:insert_pos] + [client] + target_route[insert_pos:]

        if new_source:
            feasible_source, _, _, _ = check_route(inst, new_source, use_tw=use_tw)
            if not feasible_source:
                continue

        feasible_target, _, _, _ = check_route(inst, new_target, use_tw=use_tw)
        if not feasible_target:
            continue

        candidate = []
        for idx, route in enumerate(routes):
            if idx == source_idx:
                if new_source:
                    candidate.append(new_source)
            elif idx == target_idx:
                candidate.append(new_target)
            else:
                candidate.append(route[:])

        return [r for r in candidate if r]

    return None

def _random_exchange_neighbor(
    inst: Instance,
    routes: List[List[int]],
    rng: random.Random,
    use_tw: bool = False,
    max_attempts: int = 100
) -> Optional[List[List[int]]]:
    """
    Generate one random feasible neighbor using an inter-route exchange move.

    One client from a source route is swapped with one client from a target route.

    Args:
        inst: Problem instance.
        routes: Current solution.
        rng: Random generator.
        use_tw: Whether to enforce time windows.
        max_attempts: Number of attempts to sample a feasible move.

    Returns:
        A new candidate solution if a feasible move is found, otherwise None.
    """
    if len(routes) < 2:
        return None

    valid_route_indices = [idx for idx, route in enumerate(routes) if len(route) >= 1]
    if len(valid_route_indices) < 2:
        return None

    for _ in range(max_attempts):
        source_idx, target_idx = rng.sample(valid_route_indices, 2)

        source_route = routes[source_idx]
        target_route = routes[target_idx]

        pos_a = rng.randrange(len(source_route))
        pos_b = rng.randrange(len(target_route))

        client_a = source_route[pos_a]
        client_b = target_route[pos_b]

        new_source = source_route[:]
        new_target = target_route[:]

        new_source[pos_a] = client_b
        new_target[pos_b] = client_a

        feasible_source, _, _, _ = check_route(inst, new_source, use_tw=use_tw)
        if not feasible_source:
            continue

        feasible_target, _, _, _ = check_route(inst, new_target, use_tw=use_tw)
        if not feasible_target:
            continue

        candidate = []
        for idx, route in enumerate(routes):
            if idx == source_idx:
                candidate.append(new_source)
            elif idx == target_idx:
                candidate.append(new_target)
            else:
                candidate.append(route[:])

        return [r for r in candidate if r]

    return None

def random_neighbor(
    inst: Instance,
    routes: List[List[int]],
    rng: random.Random,
    use_tw: bool = False,
    p_relocate: float = 0.5,
    p_exchange: float = 0.2,
    max_attempts: int = 100
) -> Optional[List[List[int]]]:
    """
    Generate one random feasible neighbor for simulated annealing.

    The neighbor is sampled from:
    - inter-route relocate, with probability p_relocate
    - inter-route exchange, with probability p_exchange
    - intra-route 2-opt, with remaining probability

    Args:
        inst: Problem instance.
        routes: Current solution.
        rng: Random generator.
        use_tw: Whether to enforce time windows.
        p_relocate: Probability of choosing relocate.
        p_exchange: Probability of choosing exchange.
        max_attempts: Total number of sampling attempts.

    Returns:
        A feasible neighboring solution, or None if no move is found.
    """
    if p_relocate + p_exchange > 1.0:
        raise ValueError("p_relocate + p_exchange must be <= 1.0")

    for _ in range(max_attempts):
        u = rng.random()

        if u < p_relocate:
            candidate = _random_relocate_neighbor(inst, routes, rng, use_tw=use_tw)
            if candidate is not None:
                return candidate

        elif u < p_relocate + p_exchange:
            candidate = _random_exchange_neighbor(inst, routes, rng, use_tw=use_tw)
            if candidate is not None:
                return candidate

        else:
            candidate = _random_2opt_neighbor(inst, routes, rng, use_tw=use_tw)
            if candidate is not None:
                return candidate

    return None

def _perturb_solution(inst: Instance, routes: List[List[int]], use_tw: bool = False) -> List[List[int]]:
    """Randomly move a client to escape local optimum."""
    import random
    routes = [r[:] for r in routes if r]
    
    if not routes or not any(routes):
        return routes
    
    # Pick random route and random client
    route_idx = random.randint(0, len(routes) - 1)
    if not routes[route_idx]:
        return routes
    
    client_idx = random.randint(0, len(routes[route_idx]) - 1)
    client = routes[route_idx][client_idx]
    
    # Remove from route
    routes[route_idx] = routes[route_idx][:client_idx] + routes[route_idx][client_idx+1:]
    
    # Insert into random position in random route
    target_route = random.randint(0, len(routes) - 1)
    insert_pos = random.randint(0, len(routes[target_route]))
    routes[target_route].insert(insert_pos, client)
    
    return [r for r in routes if r]

def simulated_annealing(
    inst: Instance,
    initial_routes: List[List[int]],
    use_tw: bool = False,
    seed: int = 42,
    initial_temp: float = 1000.0,
    cooling_rate: float = 0.995,
    min_temp: float = 0.1,
    iterations_per_temp: int = 200,
    p_relocate: float = 0.5,
    p_exchange: float = 0.2,
    vehicle_penalty: Optional[float] = None,
    max_stagnation_levels: int = 50,
    return_history: bool = False
):
    """
    Solve the VRP using Simulated Annealing.

    Starting from an initial feasible solution, the method repeatedly samples
    a random neighbor and accepts it if:
    - it improves the scalar energy, or
    - it worsens the energy but passes the annealing acceptance probability.

    The best feasible solution found during the search is returned.

    Args:
        inst: Problem instance.
        initial_routes: Initial feasible solution.
        use_tw: Whether to enforce time windows.
        seed: Random seed for reproducibility.
        initial_temp: Initial temperature.
        cooling_rate: Multiplicative temperature decrease factor.
        min_temp: Stopping temperature threshold.
        iterations_per_temp: Number of iterations performed at each temperature.
        p_relocate: Probability of sampling an inter-route relocate move.
        p_exchange: Probability of sampling an inter-route exchange move.
        vehicle_penalty: Optional penalty M used in the scalar energy.
        max_stagnation_levels: Stop if no best-solution improvement occurs for
            this many temperature levels.
        return_history: If True, also return the evolution history.

    Returns:
        If return_history is False:
            Best solution found.
        If return_history is True:
            A tuple (best_solution, history), where history contains the score
            of the current accepted solution after each accepted move.
    """
    rng = random.Random(seed)

    current = [route[:] for route in initial_routes if route]
    best = [route[:] for route in current]

    current_score = solution_score(inst, current)
    best_score = current_score

    current_energy = solution_energy(inst, current, vehicle_penalty=vehicle_penalty)
    best_energy = current_energy

    history = [current_score]

    temperature = initial_temp
    stagnation_levels = 0

    while temperature > min_temp and stagnation_levels < max_stagnation_levels:
        improved_this_level = False

        for _ in range(iterations_per_temp):
            candidate = random_neighbor(
                inst,
                current,
                rng,
                use_tw=use_tw,
                p_relocate=p_relocate,
                p_exchange=p_exchange
            )

            if candidate is None:
                continue

            candidate_energy = solution_energy(
                inst,
                candidate,
                vehicle_penalty=vehicle_penalty
            )
            delta = candidate_energy - current_energy

            accept = False
            if delta <= 0:
                accept = True
            else:
                probability = math.exp(-delta / temperature)
                if rng.random() < probability:
                    accept = True

            if accept:
                current = candidate
                current_energy = candidate_energy
                current_score = solution_score(inst, current)
                history.append(current_score)

                if current_score < best_score:
                    best = [route[:] for route in current]
                    best_score = current_score
                    best_energy = current_energy
                    improved_this_level = True

        if improved_this_level:
            stagnation_levels = 0
        else:
            stagnation_levels += 1

        temperature *= cooling_rate

    if return_history:
        return best, history
    return best


def _iter_exchange_candidates(
    inst: Instance,
    routes: List[List[int]],
    use_tw: bool = False
):
    """
    Enumerate all feasible inter-route exchange neighbors.

    Each yielded candidate corresponds to swapping one client from a source
    route with one client from a target route.

    Yields:
        A tuple (client_pair, candidate_routes, candidate_score).
            client_pair: a frozenset of two clients being swapped.
    """
    current_routes = [r[:] for r in routes if r]

    if len(current_routes) < 2:
        return

    for source_idx in range(len(current_routes)):
        source_route = current_routes[source_idx]

        for source_pos in range(len(source_route)):
            source_client = source_route[source_pos]

            for target_idx in range(source_idx + 1, len(current_routes)):
                target_route = current_routes[target_idx]

                for target_pos in range(len(target_route)):
                    target_client = target_route[target_pos]

                    # Swap the two clients
                    new_source = source_route[:]
                    new_target = target_route[:]

                    new_source[source_pos] = target_client
                    new_target[target_pos] = source_client

                    # Check feasibility
                    feas_source, _, _, _ = check_route(inst, new_source, use_tw=use_tw)
                    if not feas_source:
                        continue

                    feas_target, _, _, _ = check_route(inst, new_target, use_tw=use_tw)
                    if not feas_target:
                        continue

                    # Build candidate solution
                    candidate = []
                    for idx, route in enumerate(current_routes):
                        if idx == source_idx:
                            candidate.append(new_source)
                        elif idx == target_idx:
                            candidate.append(new_target)
                        else:
                            candidate.append(route[:])

                    candidate_score = solution_score(inst, candidate)

                    # Use frozenset to represent the client pair (order-independent)
                    client_pair = frozenset([source_client, target_client])

                    yield client_pair, candidate, candidate_score


def _iter_relocate_candidates(
    inst: Instance,
    routes: List[List[int]],
    use_tw: bool = False
):
    """
    Enumerate all feasible inter-route relocate neighbors.

    Each yielded candidate corresponds to moving one client from a source
    route to a target route at one insertion position.

    Yields:
        A tuple (moved_client, candidate_routes, candidate_score).
    """
    current_routes = [r[:] for r in routes if r]

    if len(current_routes) < 2:
        return

    for source_idx in range(len(current_routes)):
        source_route = current_routes[source_idx]

        for client_pos in range(len(source_route)):
            moved_client = source_route[client_pos]
            new_source = source_route[:client_pos] + source_route[client_pos + 1:]

            if new_source:
                feas_source, _, _, _ = check_route(inst, new_source, use_tw=use_tw)
                if not feas_source:
                    continue

            for target_idx in range(len(current_routes)):
                if target_idx == source_idx:
                    continue

                target_route = current_routes[target_idx]

                for insert_pos in range(len(target_route) + 1):
                    new_target = (
                        target_route[:insert_pos]
                        + [moved_client]
                        + target_route[insert_pos:]
                    )

                    feas_target, _, _, _ = check_route(inst, new_target, use_tw=use_tw)
                    if not feas_target:
                        continue

                    candidate = []
                    for idx, route in enumerate(current_routes):
                        if idx == source_idx:
                            if new_source:
                                candidate.append(new_source)
                        elif idx == target_idx:
                            candidate.append(new_target)
                        else:
                            candidate.append(route[:])

                    candidate = [r for r in candidate if r]
                    candidate_score = solution_score(inst, candidate)

                    yield moved_client, candidate, candidate_score


def tabu_search(
    inst: Instance,
    initial_routes: List[List[int]],
    use_tw: bool = False,
    tabu_tenure: int = 10,
    max_iterations: int = 200,
    max_no_improve: int = 50,
    tabu_no_2opt: bool = False,
    return_history: bool = False
):
    """
    Solve the VRP using Tabu Search with dual neighborhood exploration.

    The method explores both inter-route relocate and exchange neighborhoods
    deterministically at each iteration, selecting the best admissible move
    from both neighborhoods combined. Recently moved/swapped clients are declared
    tabu for a fixed number of iterations, unless the move satisfies the
    aspiration criterion (improves the global best).

    Optionally, a 2-opt intensification is applied after each accepted move.

    Args:
        inst: Problem instance.
        initial_routes: Initial feasible solution.
        use_tw: Whether to enforce time windows.
        tabu_tenure: Number of iterations during which a moved client stays tabu.
        max_iterations: Maximum number of tabu iterations.
        max_no_improve: Stop if the global best does not improve for this many iterations.
        tabu_no_2opt: If True, disable intra-route 2-opt intensification after each accepted move.
        return_history: If True, also return the evolution history.

    Returns:
        If return_history is False:
            Best solution found.
        If return_history is True:
            A tuple (best_solution, history).
    """
    current = [r[:] for r in initial_routes if r]
    best = [r[:] for r in current]

    current_score = solution_score(inst, current)
    best_score = current_score

    history = [current_score]

    # Tabu memory: can track either single clients (relocate) or client pairs (exchange)
    # Key: client_id for relocate, frozenset([client_a, client_b]) for exchange
    # Value: iteration index until which it is tabu
    tabu_until: Dict = {}

    no_improve_count = 0

    for iteration in range(1, max_iterations + 1):
        best_candidate = None
        best_candidate_score = None
        best_move_key = None
        best_move_is_exchange = False

        # Explore relocate neighborhood
        for moved_client, candidate, candidate_score in _iter_relocate_candidates(
            inst,
            current,
            use_tw=use_tw
        ):
            is_tabu = tabu_until.get(moved_client, 0) > iteration
            aspiration = candidate_score < best_score

            if is_tabu and not aspiration:
                continue

            if best_candidate is None or candidate_score < best_candidate_score:
                best_candidate = candidate
                best_candidate_score = candidate_score
                best_move_key = moved_client
                best_move_is_exchange = False

        # Explore exchange neighborhood
        for client_pair, candidate, candidate_score in _iter_exchange_candidates(
            inst,
            current,
            use_tw=use_tw
        ):
            is_tabu = tabu_until.get(client_pair, 0) > iteration
            aspiration = candidate_score < best_score

            if is_tabu and not aspiration:
                continue

            if best_candidate is None or candidate_score < best_candidate_score:
                best_candidate = candidate
                best_candidate_score = candidate_score
                best_move_key = client_pair
                best_move_is_exchange = True

        if best_candidate is None:
            break

        current = [r[:] for r in best_candidate if r]  # remove empty routes if any

        if not tabu_no_2opt:
            current = [
                intra_2opt(inst, route, use_tw=use_tw)
                for route in current
            ]

        # Diversification: if stuck, perturb solution
        if no_improve_count > max_no_improve // 2:
            current = _perturb_solution(inst, current, use_tw=use_tw)
            no_improve_count = 0

        # Verify feasibility
        is_feasible = all(check_route(inst, r, use_tw=use_tw)[0] for r in current)
        if not is_feasible:
            continue

        current_score = solution_score(inst, current)
        history.append(current_score)

        # Update tabu memory
        if best_move_key is not None:
            tabu_until[best_move_key] = iteration + tabu_tenure

        if current_score < best_score:
            best = [r[:] for r in current]
            best_score = current_score
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= max_no_improve:
            break

    if return_history:
        return best, history
    return best