""""
    Visualization utilities for VRPTW solutions.
"""

import matplotlib.pyplot as plt


def plot_routes(inst, routes, title="Routes", save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot clients
    xs = [inst.coords[i][0] for i in range(1, inst.n)]
    ys = [inst.coords[i][1] for i in range(1, inst.n)]
    ax.scatter(xs, ys, s=20, label="Clients")

    # Plot depot
    depot_x, depot_y = inst.coords[0]
    ax.scatter([depot_x], [depot_y], s=120, marker="s", label="Depot")

    # Plot each route
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
    plt.show()


def plot_history(history, save_path=None, show=True):
    """
    Plot the evolution of the search over accepted improving moves.

    Two stacked plots are used:
    - top: accepted move index vs total distance
    - bottom: accepted move index vs number of routes
    """
    iterations = list(range(len(history)))
    num_routes = [h[0] for h in history]
    total_dist = [h[1] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax1.plot(iterations, total_dist, marker="o")
    ax1.set_ylabel("Total distance")
    ax1.set_title("Search evolution")
    ax1.grid(True)

    ax2.plot(iterations, num_routes, marker="s")
    ax2.set_xlabel("Accepted improving move")
    ax2.set_ylabel("Number of routes")
    ax2.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_fill_ratio_experiment(fill_ratios, final_costs, final_num_routes, save_path=None, show=True):
    """
    Plot the impact of the fill ratio on final distance and final number of routes.

    Two separate stacked plots are used for clarity:
    - top: fill ratio vs final total distance
    - bottom: fill ratio vs final number of routes
    """
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