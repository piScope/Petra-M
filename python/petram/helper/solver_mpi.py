"""
Helpers for running linear solvers on a reduced number of MPI processes,
including data layouts, rank grouping, and subcommunicator management.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class SolverLayout:
    """
    Zero-based local equation IDs and requested solution IDs, in caller order.
    """
    equation_ids: np.ndarray
    solution_ids: np.ndarray


def find_solver_group_leader(node_ids, solver_size):
    """
    Return the leader's global MPI rank (rank given in MPI.COMM_WORLD) for 
    each process. The returned array contains exactly solver_size distinct 
    leader ranks.
    (example)
       node_ids   = [0, 0, 0, 0, 4, 4, 4, 4]
       solver_size = 4 

       leaders = [0, 0, 2, 2, 4, 4, 6, 6]
    """
    if not 1 <= solver_size <= len(node_ids):
        raise ValueError('Invalid solver MPI size')
    nodes = sorted(set(node_ids))
    members = {node: [i for i, n in enumerate(node_ids) if n == node]
               for node in nodes}
    slots = dict.fromkeys(nodes, 0)
    remaining = solver_size
    while remaining:
        for node in nodes:
            if remaining and slots[node] < len(members[node]):
                slots[node] += 1
                remaining -= 1
    leaders = np.full(len(node_ids), -1, dtype=np.int64)
    counts = {}
    for node in nodes:
        ranks, k = members[node], slots[node]
        for j in range(k):
            group = ranks[j * len(ranks) // k:(j + 1) * len(ranks) // k]
            leader = group[0]
            leaders[group] = leader
            counts[leader] = len(group)
    for rank in np.flatnonzero(leaders < 0):
        leader = min(counts, key=lambda r: (counts[r], r))
        leaders[rank] = leader
        counts[leader] += 1
    return leaders


class SolverRankLayout:
    """
    Own aggregation/solver communicators; borrow the application communicator.
    """
    def __init__(self, app_comm, size):
        from mpi4py import MPI
        self.app_comm = app_comm
        self.reduced = size != app_comm.size
        if not self.reduced:
            self.group_comm = MPI.COMM_SELF
            self.solver_comm = app_comm
            self.is_leader = True
            return
        node = app_comm.Split_type(MPI.COMM_TYPE_SHARED, key=app_comm.rank)
        node_id = node.bcast(app_comm.rank if node.rank == 0 else None, root=0)
        node_ids = app_comm.allgather(node_id)
        node.Free()
        leaders = find_solver_group_leader(node_ids, size)
        leader = int(leaders[app_comm.rank])
        self.group_comm = app_comm.Split(leader, key=0 if app_comm.rank == leader else app_comm.rank + 1)
        self.is_leader = app_comm.rank == leader
        self.solver_comm = app_comm.Split(0 if self.is_leader else MPI.UNDEFINED,
                                          key=app_comm.rank)

    def close(self):
        if self.reduced:
            if self.is_leader:
                self.solver_comm.Free()
            self.group_comm.Free()
            self.reduced = False
