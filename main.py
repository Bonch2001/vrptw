#!/usr/bin/env python3
"""
Main entry point for the VRP solver.

Usage examples:
    python main.py --instance data/data101.vrp --seed 42 --fill-ratio 0.65
    python main.py --instance data/data101.vrp --generator random --plot
    python main.py --instance data/data101.vrp --sweep-fill-ratios 0.5 0.6 0.7 0.8 0.9
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

from src.metrics import analyze_solution, append_run_summary_csv, build_run_summary, save_run_summary_json
from src.instance import parse_instance
from src.generator import greedy_nearest_neighbor, random_solution
from src.solver import local_search, simulated_annealing, tabu_search
from src.validator import validate_solution
from src.utils import print_solution, setup_logging
from src.visualize import plot_routes, plot_history, plot_fill_ratio_experiment

logger = setup_logging()


def generate_initial_solution(inst, args, use_tw):
    """Generate an initial solution according to the selected construction heuristic."""
    if args.generator == "random":
        return random_solution(
            inst,
            seed=args.seed,
            use_tw=use_tw,
            fill_ratio=args.fill_ratio
        )
    elif args.generator == "greedy":
        return greedy_nearest_neighbor(
            inst,
            use_tw=use_tw,
            fill_ratio=args.fill_ratio
        )
    else:
        raise ValueError(f"Unknown generator: {args.generator}")


# def run_single_instance(inst, args, use_tw):
#     """Run one solve with the current parameters."""
#     logger.info(
#         f"Generating initial solution with '{args.generator}' "
#         f"(fill_ratio={args.fill_ratio}, seed={args.seed})"
#     )

#     initial_routes = generate_initial_solution(inst, args, use_tw)
#     initial_routes = [r[:] for r in initial_routes]

#     # Apply metaheuristic optimization
#     if args.method == "local":
#         logger.info("Applying local search...")
#         final_routes, history = local_search(
#             inst,
#             initial_routes,
#             use_tw=use_tw,
#             return_history=True
#         )

#     elif args.method == "sa":
#         logger.info("Applying simulated annealing...")
#         final_routes, history = simulated_annealing(
#             inst,
#             initial_routes,
#             use_tw=use_tw,
#             seed=args.seed,
#             initial_temp=args.sa_initial_temp,
#             cooling_rate=args.sa_cooling_rate,
#             min_temp=args.sa_min_temp,
#             iterations_per_temp=args.sa_iterations_per_temp,
#             p_relocate=args.sa_p_relocate,
#             p_exchange=args.sa_p_exchange,
#             return_history=True
#         )
    
#     elif args.method == "tabu":
#         logger.info("Applying tabu search...")
#         final_routes, history = tabu_search(
#             inst,
#             initial_routes,
#             use_tw=use_tw,
#             tabu_tenure=args.tabu_tenure,
#             max_iterations=args.tabu_max_iterations,
#             max_no_improve=args.tabu_max_no_improve,
#             intensify_with_2opt=(not args.tabu_no_2opt),
#             return_history=True
#         )

#     logger.info("Validating final solution...")
#     is_valid, total_cost, num_routes = validate_solution(inst, final_routes, use_tw=use_tw)

#     if not is_valid:
#         logger.error("✗ Final solution is infeasible!")
#         sys.exit(1)

#     logger.info(f"✓ Final solution is feasible: {num_routes} routes, total cost = {total_cost:.2f}")
#     return initial_routes, final_routes, history, total_cost, num_routes

def run_single_instance(inst, args, use_tw):
    """Run one solve with the current parameters."""
    logger.info(
        f"Generating initial solution with '{args.generator}' "
        f"(fill_ratio={args.fill_ratio}, seed={args.seed})"
    )

    initial_routes = generate_initial_solution(inst, args, use_tw)
    initial_routes = [r[:] for r in initial_routes]

    initial_metrics = analyze_solution(inst, initial_routes, use_tw=use_tw)

    stats = {}
    t0 = perf_counter()

    if args.method == "local":
        logger.info("Applying local search...")
        final_routes, history = local_search(
            inst,
            initial_routes,
            use_tw=use_tw,
            return_history=True,
            #stats=stats
        )

    elif args.method == "sa":
        logger.info("Applying simulated annealing...")
        final_routes, history = simulated_annealing(
            inst,
            initial_routes,
            use_tw=use_tw,
            seed=args.seed,
            initial_temp=args.sa_initial_temp,
            cooling_rate=args.sa_cooling_rate,
            min_temp=args.sa_min_temp,
            iterations_per_temp=args.sa_iterations_per_temp,
            p_relocate=args.sa_p_relocate,
            return_history=True,
            #stats=stats
        )

    elif args.method == "tabu":
        logger.info("Applying tabu search...")
        final_routes, history = tabu_search(
            inst,
            initial_routes,
            use_tw=use_tw,
            tabu_tenure=args.tabu_tenure,
            max_iterations=args.tabu_max_iterations,
            max_no_improve=args.tabu_max_no_improve,
            intensify_with_2opt=(not args.tabu_no_2opt),
            return_history=True,
            #stats=stats
        )

    else:
        raise ValueError(f"Unknown method: {args.method}")

    runtime_seconds = perf_counter() - t0

    logger.info("Validating final solution...")
    is_valid, total_cost, num_routes = validate_solution(inst, final_routes, use_tw=use_tw)

    if not is_valid:
        logger.error("✗ Final solution is infeasible!")
        sys.exit(1)

    final_metrics = analyze_solution(inst, final_routes, use_tw=use_tw)

    summary = build_run_summary(
        instance_name=Path(args.instance).name,
        method=args.method,
        generator=args.generator,
        use_time_windows=use_tw,
        params={
            "fill_ratio": args.fill_ratio,
            "seed": args.seed,
            "sa_initial_temp": getattr(args, "sa_initial_temp", None),
            "sa_cooling_rate": getattr(args, "sa_cooling_rate", None),
            "sa_min_temp": getattr(args, "sa_min_temp", None),
            "sa_iterations_per_temp": getattr(args, "sa_iterations_per_temp", None),
            "sa_p_relocate": getattr(args, "sa_p_relocate", None),
            "tabu_tenure": getattr(args, "tabu_tenure", None),
            "tabu_max_iterations": getattr(args, "tabu_max_iterations", None),
            "tabu_max_no_improve": getattr(args, "tabu_max_no_improve", None),
        },
        initial_metrics=initial_metrics,
        final_metrics=final_metrics,
        runtime_seconds=runtime_seconds,
        history_length=len(history),
        counters=stats
    )

    logger.info(
        f"✓ Final solution is feasible: "
        f"{summary.final_routes} routes, "
        f"distance = {summary.final_distance:.2f}, "
        f"runtime = {summary.runtime_seconds:.3f}s, "
        f"Δroutes = {summary.delta_routes}, "
        f"Δdist = {summary.delta_distance:.2f} "
        f"({summary.distance_improvement_pct:.2f}%)"
    )

    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.instance).stem
    run_name = f"{stem}_{args.method}_{args.generator}_fr{args.fill_ratio}".replace(".", "_")

    save_run_summary_json(summary, str(reports_dir / f"{run_name}.json"))
    append_run_summary_csv(summary, str(reports_dir / "all_runs.csv"))

    return initial_routes, final_routes, history, total_cost, num_routes, summary

def main():
    parser = argparse.ArgumentParser(description="Solve Vehicle Routing Problem instances")

    parser.add_argument(
        "--instance", "-i",
        type=str,
        required=True,
        help="Path to .vrp instance file"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--fill-ratio", "-f",
        type=float,
        default=0.75,
        help="Fill ratio for route construction (default: 0.75)"
    )
    parser.add_argument(
        "--no-time-windows",
        action="store_true",
        help="Disable time window constraints"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for solution (optional)"
    )
    parser.add_argument(
        "--generator", "-g",
        type=str,
        default="greedy",
        choices=["random", "greedy"],
        help="Solution generator: random or greedy (default: greedy)"
    )

    parser.add_argument(
        "--method", "-m",
        type=str,
        default="local",
        choices=["local", "sa", "tabu"],
        help="Optimization method: local, sa, or tabu (default: local)"
    )

    parser.add_argument("--sa-initial-temp", type=float, default=1000.0)
    parser.add_argument("--sa-cooling-rate", type=float, default=0.55)
    parser.add_argument("--sa-min-temp", type=float, default=0.01)
    parser.add_argument("--sa-iterations-per-temp", type=int, default=200)
    parser.add_argument("--sa-p-relocate", type=float, default=0.5)

    parser.add_argument(
        "--sa-p-exchange",
        type=float,
        default=0.2,
        help="Probability of sampling exchange instead of 2-opt in SA"
    )

    parser.add_argument("--tabu-tenure", type=int, default=10)
    parser.add_argument("--tabu-max-iterations", type=int, default=200)
    parser.add_argument("--tabu-max-no-improve", type=int, default=50)
    parser.add_argument(
        "--tabu-no-2opt",
        action="store_true",
        help="Disable 2-opt intensification inside tabu search"
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show initial/final route plots and search history"
    )
    parser.add_argument(
        "--save-plots-dir",
        type=str,
        default="outputs/plots",
        help="Directory where plots will be saved (default: outputs/plots)"
    )
    parser.add_argument(
        "--sweep-fill-ratios",
        nargs="+",
        type=float,
        default=None,
        help="Run a fill-ratio experiment, e.g. --sweep-fill-ratios 0.5 0.6 0.7 0.8 0.9"
    )

    args = parser.parse_args()

    instance_path = Path(args.instance)
    if not instance_path.exists():
        logger.error(f"Instance file not found: {instance_path}")
        sys.exit(1)

    logger.info(f"Parsing instance from {instance_path}")
    inst = parse_instance(str(instance_path))
    logger.info(f"Loaded instance: n={inst.n}, capacity={inst.capacity}")

    use_tw = not args.no_time_windows

    # Optional directory for saving plots
    save_dir = Path(args.save_plots_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Fill-ratio experiment mode
    if args.sweep_fill_ratios is not None:
        logger.info("Running fill-ratio experiment...")

        final_costs = []
        final_num_routes = []

        original_fill_ratio = args.fill_ratio

        for fr in args.sweep_fill_ratios:
            args.fill_ratio = fr
            logger.info(f"Testing fill_ratio={fr}")

            _, final_routes, _, total_cost, num_routes = run_single_instance(inst, args, use_tw)
            final_costs.append(total_cost)
            final_num_routes.append(num_routes)

        args.fill_ratio = original_fill_ratio

        if args.plot or save_dir is not None:
            save_path = None
            if save_dir is not None:
                save_path = str(save_dir / "fill_ratio_experiment.png")

            plot_fill_ratio_experiment(
                args.sweep_fill_ratios,
                final_costs,
                final_num_routes,
                save_path=save_path
            )

        logger.info("Fill-ratio experiment finished.")
        return

    # Normal single run
    initial_routes, final_routes, history, total_cost, num_routes, stats = run_single_instance(inst, args, use_tw)

    print_solution(inst, final_routes, True, total_cost, use_tw)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for i, route in enumerate(final_routes):
                f.write(f"Route {i+1}: {' '.join(map(str, route))}\n")
        logger.info(f"Solution saved to {output_path}")

    if args.plot or save_dir is not None:
        initial_path = str(save_dir / "initial_solution.png") if save_dir is not None else None
        final_path = str(save_dir / "final_solution.png") if save_dir is not None else None
        history_path = str(save_dir / "search_history.png") if save_dir is not None else None

        plot_routes(
            inst,
            initial_routes,
            title=f"Initial solution ({args.generator}, fill_ratio={args.fill_ratio})",
            save_path=initial_path
        )

        plot_routes(
            inst,
            final_routes,
            title=f"Final solution ({num_routes} routes, cost={total_cost:.2f})",
            save_path=final_path
        )

        plot_history(
            history,
            save_path=history_path
        )


if __name__ == "__main__":
    main()