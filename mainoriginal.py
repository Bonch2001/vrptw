#!/usr/bin/env python3
"""
Main entry point for the VRP solver.

Usage example:
    python main.py --instance data/data101.vrp --seed 42 --fill-ratio 0.65
"""

import argparse
import sys
from pathlib import Path

from src.instance import parse_instance
from src.generator import greedy_nearest_neighbor, random_solution
from src.solver import local_search
from src.validator import validate_solution
from src.utils import print_solution, setup_logging
from src.visualize import plot_routes, plot_history, plot_fill_ratio_experiment

logger = setup_logging()


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
        default=0.65,
        help="Fill ratio for route construction (default: 0.65)"
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

    args = parser.parse_args()

    # Validate input file
    instance_path = Path(args.instance)
    if not instance_path.exists():
        logger.error(f"Instance file not found: {instance_path}")
        sys.exit(1)

    # Parse instance
    logger.info(f"Parsing instance from {instance_path}")
    inst = parse_instance(str(instance_path))
    logger.info(f"Loaded instance: n={inst.n}, capacity={inst.capacity}")

    print(args)

    # Generate solution
    use_tw = not args.no_time_windows
    logger.info(f"Generating random solution (fill_ratio={args.fill_ratio}, seed={args.seed})")
    
    if args.generator == "random":
        routes = random_solution(inst, seed=args.seed, use_tw=use_tw, fill_ratio=args.fill_ratio)
    elif args.generator == "greedy":
        routes = greedy_nearest_neighbor(inst, use_tw=use_tw, fill_ratio=args.fill_ratio)

    # Apply intra-route operations
    routes = local_search(inst, routes, use_tw=use_tw)

    # Validate solution
    logger.info("Validating solution...")
    is_valid, total_cost, num_routes = validate_solution(inst, routes, use_tw=use_tw)

    if is_valid:
        logger.info(f"✓ Solution is feasible: {num_routes} routes, total cost = {total_cost:.2f}")
    else:
        logger.error("✗ Solution is infeasible!")
        sys.exit(1)

    # Print solution
    print_solution(inst, routes, is_valid, total_cost, use_tw)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for i, route in enumerate(routes):
                f.write(f"Route {i+1}: {' '.join(map(str, route))}\n")
        logger.info(f"Solution saved to {output_path}")

    # Visualize solution
    plot_routes(inst, routes, title=f"VRPTW Solution (cost={total_cost:.2f})")
    

if __name__ == "__main__":
    main()
