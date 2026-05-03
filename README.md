# VRPTW Solver

A lightweight Vehicle Routing Problem with Time Windows solver and experiment framework.

The repository includes:

- a construction heuristic (`greedy` or `random`) for building an initial solution
- local search, simulated annealing (`sa`), and tabu search (`tabu`) metaheuristics
- a MILP-based solver for capacity-only CVRP (`milp`)
- support for parameter sweeps and parallel grid search
- per-instance CSV report writing under `outputs/reports`

## Requirements

- Python 3.9
- No external dependencies beyond the standard library

## Project structure

- `main.py` — CLI entry point and orchestration
- `src/instance.py` — instance parsing
- `src/generator.py` — solution constructors
- `src/solver.py` — optimization algorithms
- `src/metrics.py` — solution scoring and report helpers
- `src/validator.py` — route feasibility checks
- `src/visualize.py` — optional plotting utilities
- `data/` — example VRP instances
- `outputs/` — generated reports, plots, and experiment outputs

## How to run

From the repository root:

```bash
python main.py --instance data/data101.vrp --method sa
```

### Basic options

- `--instance`, `-i` — path to a `.vrp` instance file
- `--seed`, `-s` — random seed (default: `42`)
- `--fill-ratio`, `-f` — fill ratio for route construction (default: `0.75`)
- `--no-time-windows` — disable time window feasibility checks
- `--generator`, `-g` — `random` or `greedy`
- `--method`, `-m` — `local`, `sa`, `tabu`, or `milp`
- `--plot` — save route/history plots

### Simulated annealing options

- `--sa-initial-temp` — initial temperature
- `--sa-cooling-rate` — cooling factor per temperature level
- `--sa-min-temp` — minimum stopping temperature
- `--sa-iterations-per-temp` — iterations per temperature
- `--sa-p-relocate` — probability of sampling relocate moves
- `--sa-p-exchange` — probability of sampling exchange moves

### Tabu search options

- `--tabu-tenure` — tabu tenure in iterations
- `--tabu-max-iterations` — total tabu iterations
- `--tabu-max-no-improve` — stop after this many non-improving iterations
- `--tabu-no-2opt` — disable intra-route 2-opt intensification

## Sweeping and grid search

The solver supports parameter sweeps and parallel grid search.

### Sweep individual parameters

Use `--sweep-*` arguments to test multiple values. For example:

```bash
python main.py --instance data/data101.vrp --method sa \
  --sweep-seeds 1 2 3 \
  --sweep-sa-initial-temps 500 1000 1500 \
  --sweep-sa-cooling-rates 0.995 0.99
```

### Parallel grid search

Use `--parallel-grid-search` to evaluate combinations in parallel:

```bash
python main.py --instance data/data101.vrp --method sa \
  --parallel-grid-search --max-workers 8 \
  --sweep-seeds 1 2 \
  --sweep-sa-initial-temps 500 1000 \
  --sweep-sa-cooling-rates 0.995 0.99
```

### Sweep across instances

Use `--sweep-instances` to run experiments on multiple files in one command:

```bash
python main.py --method sa --parallel-grid-search \
  --sweep-instances data/data101.vrp data/data102.vrp \
  --sweep-seeds 1 2
```

## Output files

- JSON summaries are saved under `outputs/reports`
- per-instance CSV reports are written to `outputs/reports/{instance_name}_runs.csv`
- plots are saved to `outputs/plots` when `--plot` is enabled

## Examples

Run a single SA experiment:

```bash
python main.py --instance data/data101.vrp --method sa --generator greedy
```

Run a Tabu search experiment with random init:

```bash
python main.py --instance data/data101.vrp --method tabu --generator random
```

Run a sweep of SA cooling rates:

```bash
python main.py --instance data/data101.vrp --method sa \
  --sweep-sa-cooling-rates 0.99 0.995 0.999
```

Run a parallel sweep across two instances:

```bash
python main.py --method sa --parallel-grid-search \
  --sweep-instances data/data101.vrp data/data102.vrp \
  --sweep-seeds 1 2
```

## Notes

- `milp` mode currently only supports capacity-only CVRP and cannot be used with `--no-time-windows`
- The solver writes run summaries incrementally, so repeated sweeps append new rows to the instance CSV files
- Use `--reset-all-runs` before a sweep to remove previous per-instance CSV results if you want a fresh report
