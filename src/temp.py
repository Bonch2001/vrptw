"""
Visualization utilities for VRPTW solutions.
"""

import matplotlib.pyplot as plt
from src.metrics import best_so_far_history


def plot_routes(inst, routes, title="Routes", save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(8, 8))

    xs = [inst.coords[i][0] for i in range(1, inst.n)]
    ys = [inst.coords[i][1] for i in range(1, inst.n)]
    ax.scatter(xs, ys, s=20, label="Clients")

    depot_x, depot_y = inst.coords[0]
    ax.scatter([depot_x], [depot_y], s=120, marker="s", label="Depot")

    for idx, route in enumerate(routes, start=1):
        if not route:
            continue

        path = [0] + route + [0]
        px = [inst.coords[node][0] for node in path]
        py = [inst.coords[node][1] for node in path]

        ax.plot(px, py, marker="o", linewidth=1.5, markersize=3, label=f"R{idx}")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_history(history, save_path=None, show=True):
    """
    Plot both current-history and best-so-far history.
    """
    if not history:
        return

    best_history = best_so_far_history(history)

    iterations = list(range(len(history)))

    cur_routes = [h[0] for h in history]
    cur_dist = [h[1] for h in history]

    best_routes = [h[0] for h in best_history]
    best_dist = [h[1] for h in best_history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(iterations, cur_dist, label="Current", alpha=0.7)
    ax1.plot(iterations, best_dist, label="Best-so-far", linewidth=2.0)
    ax1.set_ylabel("Total distance")
    ax1.set_title("Search evolution")
    ax1.legend()
    ax1.grid(True)

    ax2.step(iterations, cur_routes, where="post", label="Current", alpha=0.7)
    ax2.step(iterations, best_routes, where="post", label="Best-so-far", linewidth=2.0)
    ax2.set_xlabel("Accepted move")
    ax2.set_ylabel("Number of routes")
    ax2.legend()
    ax2.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_fill_ratio_experiment(fill_ratios, final_costs, final_num_routes, save_path=None, show=True):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax1.plot(fill_ratios, final_costs, marker="o")
    ax1.set_ylabel("Final total distance")
    ax1.set_title("Impact of fill ratio on solution quality")
    ax1.grid(True)

    ax2.plot(fill_ratios, final_num_routes, marker="s")
    ax2.set_xlabel("Fill ratio")
    ax2.set_ylabel("Final number of routes")
    ax2.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()