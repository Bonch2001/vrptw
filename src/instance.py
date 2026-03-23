"""
Instance class and parsing for VRP instances.
"""

from dataclasses import dataclass
from math import hypot
from typing import List, Tuple


@dataclass
class Instance:
    """Vehicle Routing Problem instance."""
    
    n: int                             # number of nodes incl depots (0..n-1)
    capacity: int                      # vehicle capacity
    coords: List[Tuple[float, float]]  # node coordinates
    demand: List[int]                  # node demand
    ready: List[float]                 # time window ready times
    due: List[float]                   # time window due times
    service: List[float]               # service times at nodes
    speed_coeff: float = 1.0           # speed coefficient for travel time

    def dist(self, i: int, j: int) -> float:
        """Euclidean distance between nodes i and j."""
        xi, yi = self.coords[i]
        xj, yj = self.coords[j]
        return hypot(xi - xj, yi - yj)

    def travel_time(self, i: int, j: int) -> float:
        """Travel time between nodes i and j (distance / speed)."""
        return self.dist(i, j) * self.speed_coeff


def parse_instance(filename: str) -> Instance:
    """Parse VRPTW instance file (Solomon format with metadata headers)."""
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Skip metadata lines (NAME, COMMENT, TYPE, COORDINATES etc.)
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("NB_DEPOTS:"):
            n_depots = int(line.split(":")[1].strip())
        elif line.startswith("NB_CLIENTS:"):
            n_clients = int(line.split(":")[1].strip())
        elif line.startswith("MAX_QUANTITY:"):
            capacity = int(line.split(":")[1].strip())
        elif line.startswith("DATA_DEPOTS"):
            data_start = i + 1
            break

    n = n_clients + n_depots

    # Parse depot
    depot_line = lines[data_start].split()
    depot_x, depot_y = float(depot_line[1]), float(depot_line[2])
    depot_ready, depot_due = float(depot_line[3]), float(depot_line[4])

    # Find clients section
    clients_start = data_start + 1
    for i in range(data_start + 1, len(lines)):
        if lines[i].startswith("DATA_CLIENTS"):
            clients_start = i + 1
            break

    # Parse clients
    coords = [(depot_x, depot_y)]
    demand = [0]
    ready = [depot_ready]
    due = [depot_due]
    service = [0.0]

    for line in lines[clients_start:]:
        parts = line.split()
        if len(parts) < 6:
            continue
        coords.append((float(parts[1]), float(parts[2])))
        demand.append(int(parts[5]))
        ready.append(float(parts[3]))
        due.append(float(parts[4]))
        service.append(float(parts[6]))

    return Instance(
        n=n,
        capacity=capacity,
        coords=coords,
        demand=demand,
        ready=ready,
        due=due,
        service=service
    )
