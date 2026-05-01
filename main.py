#!/usr/bin/env python3
"""
Main entry point for the VRP solver.

Usage examples:
    python main.py --instance data/data101.vrp --seed 42 --fill-ratio 0.65
    python main.py --instance data/data101.vrp --generator random --plot
    python main.py --instance data/data101.vrp --sweep-fill-ratios 0.5 0.6 0.7 0.8 0.9
"""

import argparse
from ast import arg
import sys
from itertools import product
from pathlib import Path
from time import perf_counter
import os
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.metrics import analyze_solution, append_run_summary_csv, build_run_summary, save_run_summary_json
from src.instance import parse_instance
from src.generator import greedy_nearest_neighbor, random_solution
from src.solver import local_search, simulated_annealing, solution_score, tabu_search
from src.validator import validate_solution
from src.utils import print_solution, setup_logging
from src.visualize import plot_routes, plot_history, plot_fill_ratio_experiment
from src.milp_solver import solve_cvrp_milp

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


def parse_sweep_values(values, default):
    return list(values) if values is not None else [default]


def get_instance_csv_path(instance_path: str) -> Path:
    """Generate CSV path for a specific instance."""
    instance_name = Path(instance_path).stem  # Remove .vrp extension
    reports_dir = Path("outputs/reports")
    return reports_dir / f"{instance_name}_runs.csv"


def build_sweep_combinations(args):
    instances = parse_sweep_values(args.sweep_instances, args.instance)
    seeds = parse_sweep_values(args.sweep_seeds, args.seed)
    if args.method == "sa":
        return [
            {
                "instance": instance,
                "seed": seed,
                "sa_initial_temp": sa_initial_temp,
                "sa_cooling_rate": sa_cooling_rate,
                "sa_min_temp": sa_min_temp,
                "sa_iterations_per_temp": sa_iterations_per_temp,
                "sa_p_relocate": sa_p_relocate,
                "sa_p_exchange": sa_p_exchange,
            }
            for instance in instances
            for seed, sa_initial_temp, sa_cooling_rate, sa_min_temp,
                sa_iterations_per_temp, sa_p_relocate, sa_p_exchange in product(
                seeds,
                parse_sweep_values(args.sweep_sa_initial_temps, args.sa_initial_temp),
                parse_sweep_values(args.sweep_sa_cooling_rates, args.sa_cooling_rate),
                parse_sweep_values(args.sweep_sa_min_temps, args.sa_min_temp),
                parse_sweep_values(args.sweep_sa_iterations_per_temp, args.sa_iterations_per_temp),
                parse_sweep_values(args.sweep_sa_p_relocate, args.sa_p_relocate),
                parse_sweep_values(args.sweep_sa_p_exchange, args.sa_p_exchange),
            )
        ]
    elif args.method == "tabu":
        return [
            {
                "instance": instance,
                "seed": seed,
                "tabu_tenure": tabu_tenure,
                "tabu_max_iterations": tabu_max_iterations,
                "tabu_max_no_improve": tabu_max_no_improve,
                "tabu_no_2opt": tabu_no_2opt,
            }
            for instance in instances
            for seed, tabu_tenure, tabu_max_iterations, tabu_max_no_improve, tabu_no_2opt in product(
                seeds,
                parse_sweep_values(args.sweep_tabu_tenures, args.tabu_tenure),
                parse_sweep_values(args.sweep_tabu_max_iterations, args.tabu_max_iterations),
                parse_sweep_values(args.sweep_tabu_max_no_improve, args.tabu_max_no_improve),
                parse_sweep_values(args.sweep_tabu_no_2opt, args.tabu_no_2opt),
            )
        ]
    else:
        return [
            {"instance": instance, "seed": seed}
            for instance in instances
            for seed in seeds
        ]


def run_parameter_sweep(inst, args, use_tw):
    combinations = build_sweep_combinations(args)
    if not combinations:
        logger.warning("No sweep combinations generated. Running a single instance instead.")
        run_single_instance(inst, args, use_tw)
        return

    if args.dry_run:
        logger.info("Dry run: the following parameter combinations will be executed:")
        for combo in combinations:
            logger.info(str(combo))
        logger.info(f"Total planned runs: {len(combinations)}")
        return

    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    instance_csv_path = get_instance_csv_path(args.instance)
    if args.reset_all_runs and instance_csv_path.exists():
        instance_csv_path.unlink()
        logger.info(f"Removed existing run summary file: {instance_csv_path}")

    best_summary = None
    best_distance = float("inf")
    best_routes = float("inf")

    logger.info(f"Running parameter sweep with {len(combinations)} runs...")
    for index, combo in enumerate(combinations, start=1):
        logger.info(f"[{index}/{len(combinations)}] Running combination: {combo}")
        sweep_args = argparse.Namespace(**vars(args))
        for key, value in combo.items():
            setattr(sweep_args, key, value)

        # Parse instance if it varies
        if "instance" in combo:
            inst = parse_instance(combo["instance"])
        else:
            inst = parse_instance(args.instance)

        try:
            _, _, _, total_cost, num_routes, summary = run_single_instance(inst, sweep_args, use_tw)

            # Append each run to CSV
            instance_csv_path = get_instance_csv_path(sweep_args.instance)
            append_run_summary_csv(summary, str(instance_csv_path))

            if num_routes < best_routes or (num_routes == best_routes and total_cost < best_distance):
                best_routes = num_routes
                best_distance = total_cost
                best_summary = summary
        except SystemExit:
            logger.error(f"Run failed for combo: {combo}")
            continue

    if best_summary is not None:
        logger.info("Best sweep result:")
        logger.info(
            f"  method={best_summary.method} generator={best_summary.generator} "
            f"final_routes={best_summary.final_routes} final_distance={best_summary.final_distance:.2f} "
            f"params={best_summary.params} runtime={best_summary.runtime_seconds:.3f}s"
        )
    else:
        logger.warning("Sweep completed but no valid run summaries were recorded.")


def build_run_tag(args, use_tw):
    """
    Build a compact filename suffix describing the run parameters.
    """
    parts = [
        Path(args.instance).stem,
        f"gen-{args.generator}",
        f"method-{args.method}",
        f"fr-{args.fill_ratio}",
        "tw" if use_tw else "notw",
        f"seed-{args.seed}",
    ]

    if args.method == "sa":
        parts.extend([
            f"T0-{args.sa_initial_temp}",
            f"cool-{args.sa_cooling_rate}",
            f"Tmin-{args.sa_min_temp}",
            f"iterT-{args.sa_iterations_per_temp}",
            f"prel-{args.sa_p_relocate}",
            f"pex-{args.sa_p_exchange}"
        ])

    if args.method == "tabu":
        parts.extend([
            f"tenure-{args.tabu_tenure}",
            f"iter-{args.tabu_max_iterations}",
            f"noimp-{args.tabu_max_no_improve}",
            f"twoopt-{int(not args.tabu_no_2opt)}",
        ])

    if args.method == "milp":
        parts.extend([
            f"milpk-{args.milp_max_vehicles}",
            f"milpt-{args.milp_time_limit}",
            f"milpgap-{args.milp_mip_gap}",
        ])

    # make filename safe
    tag = "_".join(str(p).replace(".", "p") for p in parts)
    return tag

def run_one_combination_worker(task):
    """
    Run a single parameter combination in a separate process.
    Does not do plotting or write shared reports to file.
    """
    instance_path, base_args_dict, combo = task

    # reconstruct the args namespace locally in the worker process
    args = argparse.Namespace(**base_args_dict)
    for k, v in combo.items():
        setattr(args, k, v)

    inst = parse_instance(str(instance_path))
    use_tw = not args.no_time_windows

    _, _, _, total_cost, num_routes, summary = run_single_instance(
        inst,
        args,
        use_tw,
        save_reports=False,
        quiet=True
    )

    return {
        "combo": combo,
        "num_routes": num_routes,
        "distance": total_cost,
        "runtime": summary.runtime_seconds,
        "summary": summary,
    }

def run_parallel_grid_search(args, param_grid, max_workers=6):
    """
    runs all combinations of parameters in parallel
    csv writing is done sequentially in the main process
    """
    keys = list(param_grid.keys())
    values_product = list(itertools.product(*(param_grid[k] for k in keys)))

    combos = [dict(zip(keys, vals)) for vals in values_product]

    tasks = [
        (Path(combo.get("instance", args.instance)), vars(args).copy(), combo)
        for combo in combos
    ]

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_combo = {
            executor.submit(run_one_combination_worker, task): task[2]
            for task in tasks
        }

        total = len(future_to_combo)

        for idx, future in enumerate(as_completed(future_to_combo), start=1):
            combo = future_to_combo[future]

            try:
                result = future.result()
                results.append(result)

                summary = result["summary"]

                # ensure process safety by doing it sequentially as results come in
                reports_dir = Path("outputs/reports")
                reports_dir.mkdir(parents=True, exist_ok=True)
                
                # commented out the run_name-based JSON saving to avoid clutter when sweeping many combinations
                # stem = Path(args.instance).stem
                # combo_tag = "_".join(
                #     f"{k}-{str(v).replace('.', 'p')}" for k, v in combo.items()
                # )
                # run_name = f"{stem}_{args.method}_{args.generator}_{combo_tag}"
                # save_run_summary_json(summary, str(reports_dir / f"{run_name}.json"))
                
                instance_csv_path = get_instance_csv_path(summary.instance_name)
                append_run_summary_csv(summary, str(instance_csv_path))

                logger.info(
                    f"[{idx}/{total}] Done {combo} -> "
                    f"routes={result['num_routes']}, "
                    f"dist={result['distance']:.2f}, "
                    f"time={result['runtime']:.2f}s"
                )

            except Exception as e:
                logger.exception(f"[{idx}/{total}] Failed combo {combo}: {e}")

    return results

def pick_best_result(results):
    """
    Lexicographically pick the best result based on (num_routes, distance, runtime).
    """
    return min(results, key=lambda r: (r["num_routes"], r["distance"], r["runtime"]))

def run_single_instance(inst, args, use_tw, save_reports=True, quiet=False):
    if not quiet:
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
        if not quiet:
            logger.info("Applying local search...")
        final_routes, history = local_search(
            inst,
            initial_routes,
            use_tw=use_tw,
            return_history=True,
        )

    elif args.method == "sa":
        if not quiet:
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
            p_exchange=args.sa_p_exchange,
            return_history=True,
        )

    elif args.method == "tabu":
        if not quiet:
            logger.info("Applying tabu search...")
        final_routes, history = tabu_search(
            inst,
            initial_routes,
            use_tw=use_tw,
            tabu_tenure=args.tabu_tenure,
            max_iterations=args.tabu_max_iterations,
            max_no_improve=args.tabu_max_no_improve,
            tabu_no_2opt=args.tabu_no_2opt,
            return_history=True,
        )

    elif args.method == "milp":
        if use_tw:
            raise ValueError(
                "This PLNE/MILP implementation currently supports only the capacity-only CVRP. "
                "Please run it with --no-time-windows."
            )

        if not quiet:
            logger.info("Applying exact MILP (CVRP, no time windows)...")

        final_routes, milp_history = solve_cvrp_milp(
            inst,
            max_vehicles=args.milp_max_vehicles,
            time_limit=args.milp_time_limit,
            mip_gap=args.milp_mip_gap,
            return_history=True
        )

        # For plotting, compare heuristic initial solution vs exact MILP result.
        history = [solution_score(inst, initial_routes)]
        history.extend(milp_history)

    else:
        raise ValueError(f"Unknown method: {args.method}")

    runtime_seconds = perf_counter() - t0

    if not quiet:
        logger.info("Validating final solution...")
    is_valid, total_cost, num_routes = validate_solution(inst, final_routes, use_tw=use_tw)

    if not is_valid:
        if not quiet:
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
            "sa_p_exchange": getattr(args, "sa_p_exchange", None),
            "tabu_tenure": getattr(args, "tabu_tenure", None),
            "tabu_max_iterations": getattr(args, "tabu_max_iterations", None),
            "tabu_max_no_improve": getattr(args, "tabu_max_no_improve", None),
            "tabu_no_2opt": getattr(args, "tabu_no_2opt", None),
            # "milp_max_vehicles": getattr(args, "milp_max_vehicles", None),
            # "milp_time_limit": getattr(args, "milp_time_limit", None),
            # "milp_mip_gap": getattr(args, "milp_mip_gap", None),
        },
        initial_metrics=initial_metrics,
        final_metrics=final_metrics,
        runtime_seconds=runtime_seconds,
        history_length=len(history),
        counters=stats
    )

    if not quiet:
        logger.info(
            f"✓ Final solution is feasible: "
            f"{summary.final_routes} routes, "
            f"distance = {summary.final_distance:.2f}, "
            f"runtime = {summary.runtime_seconds:.3f}s, "
            f"Δroutes = {summary.delta_routes}, "
            f"Δdist = {summary.delta_distance:.2f} "
            f"({summary.distance_improvement_pct:.2f}%)"
        )

    if save_reports:
        reports_dir = Path("outputs/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(args.instance).stem
        run_name = f"{stem}_{args.method}_{args.generator}_fr{args.fill_ratio}_{args.no_time_windows}".replace(".", "_")

        save_run_summary_json(summary, str(reports_dir / f"{run_name}.json"))
        instance_csv_path = get_instance_csv_path(args.instance)
        append_run_summary_csv(summary, str(instance_csv_path))

    return initial_routes, final_routes, history, total_cost, num_routes, summary

def main():
    parser = argparse.ArgumentParser(description="Solve Vehicle Routing Problem instances")

    parser.add_argument(
        "--instance", "-i",
        type=str,
        default="data/data101.vrp",
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
        choices=["local", "sa", "tabu", "milp"],
        help="Optimization method: local, sa, tabu, or milp (default: local)"
    )

    parser.add_argument("--sa-initial-temp", type=float, default=1000.0)
    parser.add_argument("--sa-cooling-rate", type=float, default=0.995)
    parser.add_argument("--sa-min-temp", type=float, default=0.01)
    parser.add_argument("--sa-iterations-per-temp", type=int, default=400)
    parser.add_argument("--sa-p-relocate", type=float, default=0.5)

    parser.add_argument(
        "--sa-p-exchange",
        type=float,
        default=0.2,
        help="Probability of sampling exchange instead of 2-opt in SA"
    )

    parser.add_argument(
        "--sweep-seeds",
        nargs="+",
        type=int,
        default=None,
        help="List of seeds to sweep over for repeated runs"
    )
    parser.add_argument(
        "--sweep-sa-initial-temps",
        nargs="+",
        type=float,
        default=None,
        help="List of SA initial temperatures to sweep over"
    )
    parser.add_argument(
        "--sweep-sa-cooling-rates",
        nargs="+",
        type=float,
        default=None,
        help="List of SA cooling rates to sweep over"
    )
    parser.add_argument(
        "--sweep-sa-min-temps",
        nargs="+",
        type=float,
        default=None,
        help="List of SA minimum temperatures to sweep over"
    )
    parser.add_argument(
        "--sweep-sa-iterations-per-temp",
        nargs="+",
        type=int,
        default=None,
        help="List of SA iterations per temperature to sweep over"
    )
    parser.add_argument(
        "--sweep-sa-p-relocate",
        nargs="+",
        type=float,
        default=None,
        help="List of SA relocate probabilities to sweep over"
    )
    parser.add_argument(
        "--sweep-sa-p-exchange",
        nargs="+",
        type=float,
        default=None,
        help="List of SA exchange probabilities to sweep over"
    )

    parser.add_argument(
        "--parallel-grid-search",
        action="store_true",
        help="Run the parameter grid search in parallel"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(6, os.cpu_count() or 1),
        help="Number of worker processes for parallel grid search"
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
        "--sweep-tabu-tenures",
        nargs="+",
        type=int,
        default=None,
        help="List of tabu tenures to sweep over"
    )
    parser.add_argument(
        "--sweep-tabu-max-iterations",
        nargs="+",
        type=int,
        default=None,
        help="List of tabu max iteration values to sweep over"
    )
    parser.add_argument(
        "--sweep-tabu-max-no-improve",
        nargs="+",
        type=int,
        default=None,
        help="List of tabu max no-improve values to sweep over"
    )
    parser.add_argument(
        "--sweep-tabu-no-2opt",
        nargs="+",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=None,
        help="List of tabu no-2opt values to sweep over"
    )
    
    parser.add_argument(
        "--sweep-instances",
        nargs="+",
        type=str,
        default=None,
        help="List of instance files to sweep over"
    )

    parser.add_argument(
        "--reset-all-runs",
        action="store_true",
        help="Remove outputs/reports/all_runs.csv before running a sweep"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned sweep combinations without executing them"
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

    parser.add_argument(
        "--milp-max-vehicles",
        type=int,
        default=None,
        help="Optional upper bound on the number of vehicles for the MILP search"
    )

    parser.add_argument(
        "--milp-time-limit",
        type=float,
        default=60.0,
        help="Per-MILP solve time limit in seconds (default: 60)"
    )

    parser.add_argument(
        "--milp-mip-gap",
        type=float,
        default=0.0,
        help="Relative MIP gap for the MILP solver (default: 0.0)"
    )

    args = parser.parse_args()

    # Parallel grid search for Simulated Annealing and Tabu Search
    if args.parallel_grid_search:
        if args.method not in ["sa", "tabu"]:
            logger.error("--parallel-grid-search is currently configured for --method sa and tabu only.")
            return

        # Build the parameter grid from CLI sweep arguments.
        # If a sweep argument is missing, fall back to the current single value in args.
        param_grid = {}
        if args.sweep_instances:
            param_grid["instance"] = args.sweep_instances
        else:
            param_grid["instance"] = [args.instance]

        if args.method == "sa":
            param_grid.update({
                "seed": args.sweep_seeds if getattr(args, "sweep_seeds", None) else [args.seed],
                "sa_initial_temp": (
                    args.sweep_sa_initial_temps
                    if getattr(args, "sweep_sa_initial_temps", None)
                    else [args.sa_initial_temp]
                ),
                "sa_cooling_rate": (
                    args.sweep_sa_cooling_rates
                    if getattr(args, "sweep_sa_cooling_rates", None)
                    else [args.sa_cooling_rate]
                ),
                "sa_min_temp": (
                    args.sweep_sa_min_temps
                    if getattr(args, "sweep_sa_min_temps", None)
                    else [args.sa_min_temp]
                ),
                "sa_iterations_per_temp": (
                    args.sweep_sa_iterations_per_temp
                    if getattr(args, "sweep_sa_iterations_per_temp", None)
                    else [args.sa_iterations_per_temp]
                ),
                "sa_p_relocate": (
                    args.sweep_sa_p_relocate
                    if getattr(args, "sweep_sa_p_relocate", None)
                    else [args.sa_p_relocate]
                ),
                "sa_p_exchange": (
                    args.sweep_sa_p_exchange
                    if getattr(args, "sweep_sa_p_exchange", None)
                    else [args.sa_p_exchange]
                ),
            })
        elif args.method == "tabu":
            param_grid.update({
                "seed": args.sweep_seeds if getattr(args, "sweep_seeds", None) else [args.seed],
                "tabu_tenure": (
                    args.sweep_tabu_tenures
                    if getattr(args, "sweep_tabu_tenures", None)
                    else [args.tabu_tenure]
                ),
                "tabu_max_iterations": (
                    args.sweep_tabu_max_iterations
                    if getattr(args, "sweep_tabu_max_iterations", None)
                    else [args.tabu_max_iterations]
                ),
                "tabu_max_no_improve": (
                    args.sweep_tabu_max_no_improve
                    if getattr(args, "sweep_tabu_max_no_improve", None)
                    else [args.tabu_max_no_improve]
                ),
                "tabu_no_2opt": (
                    args.sweep_tabu_no_2opt
                    if getattr(args, "sweep_tabu_no_2opt", None)
                    else [args.tabu_no_2opt]
                )
            })

        # For testing purposes:
        # sa
        # param_grid = {
        #     "seed": [1, 2],
        #     "generator": ["random"],
        #     "instance": ["data/data111.vrp", "data/data112.vrp", "data/data201.vrp", "data/data202.vrp"],
        #     "sa_initial_temp": [350, 400, 450, 500, 600],
        #     "sa_cooling_rate": [0.985, 0.99, 0.995],
        #     "sa_iterations_per_temp": [500, 600, 700, 800, 900, 1000],
        #     "sa_p_relocate": [0.3, 0.5],
        #     "sa_p_exchange": [0.1, 0.2, 0.3],
        # }

        # param_grid.update({
        #     "seed": [1, 2, 3],
        #     "instance": ["data/data111.vrp", "data/data112.vrp"],
        #     "generator": ["random", "greedy"],
        # })

        param_grid = {
            "seed": [2],
            "instance": ["data/data101.vrp", "data/data111.vrp"],
            "generator": ["random"],
            "sa_initial_temp": [400],
            "sa_min_temp": [0.01],
            "sa_cooling_rate": [0.99],
            "sa_iterations_per_temp": [500],
            "sa_p_relocate": [0.4],
            "sa_p_exchange": [0.3], 
        }

        # # tabu
        # param_grid = {
        #     "seed": [1, 2, 3],
        #     "tabu_tenure": [20, 30, 40, 50],
        #     "tabu_max_iterations": [500, 600, 700, 800, 900, 1000],
        #     "tabu_max_no_improve": [100, 150, 200, 250, 300],
        #     "tabu_no_2opt": [True, False],
        # }

        # Count combinations
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)

        logger.info("Parallel grid search requested.")
        logger.info(f"Method: {args.method}")
        logger.info(f"Generator: {args.generator}")
        logger.info(f"Instance: {args.instance}")
        logger.info(f"Total parameter combinations: {total_combinations}")
        logger.info(f"Max workers: {args.max_workers}")

        # Dry run: only print the grid and stop
        if getattr(args, "dry_run", False):
            logger.info("Dry run mode enabled. No optimization runs will be executed.")
            logger.info(f"Parameter grid: {param_grid}")
            return

        # Run all combinations in parallel
        results = run_parallel_grid_search(
            args,
            param_grid,
            max_workers=args.max_workers
        )

        if not results:
            logger.error("No valid results returned by the parallel grid search.")
            return

        # Pick the best result using lexicographic ranking:
        # 1) fewer routes
        # 2) lower distance
        # 3) lower runtime
        best = pick_best_result(results)

        logger.info("Parallel grid search completed.")
        logger.info(f"Best combination: {best['combo']}")
        logger.info(
            f"Best result -> distance={best['distance']:.2f}, "
            f"routes={best['num_routes']}, "
            f"time={best['runtime']:.2f}s"
        )

        # Re-run the best combination normally so that the usual
        # plots / outputs / reports are generated by the main process.
        for k, v in best["combo"].items():
            setattr(args, k, v)

        logger.info("Re-running the best combination with normal output generation...")


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

    sweep_mode = any([
        args.dry_run,
        args.reset_all_runs,
        args.sweep_instances is not None,
        args.sweep_seeds is not None,
        args.sweep_sa_initial_temps is not None,
        args.sweep_sa_cooling_rates is not None,
        args.sweep_sa_min_temps is not None,
        args.sweep_sa_iterations_per_temp is not None,
        args.sweep_sa_p_relocate is not None,
        args.sweep_sa_p_exchange is not None,
        args.sweep_tabu_tenures is not None,
        args.sweep_tabu_max_iterations is not None,
        args.sweep_tabu_max_no_improve is not None,
        args.sweep_tabu_no_2opt is not None,
    ])

    if sweep_mode:
        run_parameter_sweep(inst, args, use_tw)
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
        # initial_path = str(save_dir / "initial_solution.png") if save_dir is not None else None
        # final_path = str(save_dir / "final_solution.png") if save_dir is not None else None
        # history_path = str(save_dir / "search_history.png") if save_dir is not None else None

        run_tag = build_run_tag(args, use_tw)

        initial_path = str(save_dir / f"{run_tag}_initial.png")
        final_path = str(save_dir / f"{run_tag}_final.png")
        history_path = str(save_dir / f"{run_tag}_history.png")

        plot_routes(
            inst,
            initial_routes,
            title=f"Initial solution ({args.generator}, fill_ratio={args.fill_ratio})",
            save_path=initial_path,
            show=True,
            show_route_labels=False
        )

        plot_routes(
            inst,
            final_routes,
            title=f"Final solution ({num_routes} routes, cost={total_cost:.2f})",
            save_path=final_path,
            show=True,
            show_route_labels=False
        )

        plot_history(
            history,
            save_path=history_path
        )


if __name__ == "__main__":
    main()