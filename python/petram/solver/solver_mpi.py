"""
Communicator reduction and indexed transfers for distributed linear solvers.
"""
from dataclasses import dataclass
import os
import numpy as np


@dataclass
class SolverLayout:
    """
    Zero-based local equation IDs and requested solution IDs, in caller order.
    """
    equation_ids: np.ndarray
    solution_ids: np.ndarray


def collective_call(comm, operation):
    """
    Execute local work and report failures coherently before the next collective.
    """
    result = None
    error = None
    try:
        result = operation()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    errors = comm.allgather(error)
    for rank, message in enumerate(errors):
        if message is not None:
            raise RuntimeError(f'Linear solver rank {rank}: {message}')
    return result


def resolve_solver_size(comm, value, environ=None):
    """
    Resolve the model/environment setting collectively without changing the model.
    """
    environ = os.environ if environ is None else environ

    def resolve():
        text = str(value).strip()
        n = int(text)
        source = 'model'
        env = None
        if n == -1:
            env = environ.get('PETRAM_MUMPS_MPI_SIZE')
            source = 'default' if env is None else 'environment'
            n = comm.size if env is None else int(env.strip())
        if not 1 <= n <= comm.size:
            raise ValueError(f'Solver MPI ranks must be in 1..{comm.size}, got {n}')
        return n, source, text, env

    resolved = collective_call(comm, resolve)
    settings = comm.allgather(resolved)
    if any(item != settings[0] for item in settings):
        raise ValueError('MUMPS MPI settings differ between application ranks')
    return resolved[:2]


def select_groups(node_ids, solver_size):
    """
    Return each application's leader, balancing solver counts across nodes.
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


def _counts(counts):
    """
    Check portable MPI counts and compute displacements without narrowing.
    """
    counts = np.asarray(counts, dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts)]
    if np.any(counts < 0) or offsets[-1] > np.iinfo(np.int32).max:
        raise ValueError('Transfer exceeds supported MPI count range')
    return counts, offsets


def gather_array(comm, values):
    """
    Gather a contiguous one-dimensional NumPy buffer onto local rank zero.
    """
    from mpi4py import MPI
    values = collective_call(comm, lambda: np.ascontiguousarray(values).reshape(-1))
    meta = comm.allgather((values.size, values.dtype.str))
    if len({dtype for _, dtype in meta}) != 1:
        raise ValueError('Inconsistent transfer dtypes')
    counts, offsets = _counts([n for n, _ in meta])
    result = collective_call(comm, lambda: np.empty(offsets[-1], dtype=values.dtype)
                             if comm.rank == 0 else None)
    recv = [result, counts, offsets[:-1], MPI._typedict[values.dtype.char]] if comm.rank == 0 else None
    comm.Gatherv(values, recv, root=0)
    return result


def exchange_arrays(comm, values):
    """
    Exchange one NumPy buffer per destination and return buffers by source rank.
    """
    from mpi4py import MPI
    packed = collective_call(comm, lambda: np.ascontiguousarray(np.concatenate(values)))
    if len(values) != comm.size:
        raise ValueError('One buffer per destination is required')
    dtypes = comm.allgather(packed.dtype.str)
    if len(set(dtypes)) != 1:
        raise ValueError('Inconsistent exchange dtypes')
    send_counts = np.array([v.size for v in values], dtype=np.int64)
    recv_counts = np.empty(comm.size, dtype=np.int64)
    comm.Alltoall(send_counts, recv_counts)
    (sc, so), (rc, ro) = collective_call(
        comm, lambda: (_counts(send_counts), _counts(recv_counts)))
    result = collective_call(comm, lambda: np.empty(ro[-1], dtype=packed.dtype))
    dtype = MPI._typedict[packed.dtype.char]
    comm.Alltoallv([packed, sc, so[:-1], dtype], [result, rc, ro[:-1], dtype])
    return [result[ro[i]:ro[i + 1]] for i in range(comm.size)]


class IndexedTransfer:
    """
    Cache a transfer from uniquely owned global IDs to requested local IDs.
    """
    def __init__(self, comm, owned_ids, requested_ids):
        from mpi4py import MPI
        self.comm = comm
        owned = np.asarray(owned_ids, dtype=np.int64)
        requested = np.asarray(requested_ids, dtype=np.int64)
        counts, offsets = _counts(comm.allgather(owned.size))
        ids = collective_call(comm, lambda: np.empty(offsets[-1], dtype=np.int64))
        comm.Allgatherv(owned, [ids, counts, offsets[:-1], MPI.INT64_T])

        def make_requests():
            order = np.argsort(ids)
            sorted_ids = ids[order]
            if np.any(np.diff(sorted_ids) == 0):
                raise ValueError('Duplicate owned solution indices')
            pos = np.searchsorted(sorted_ids, requested)
            if np.any(pos >= ids.size) or not np.array_equal(sorted_ids[pos], requested):
                raise ValueError('Missing solution indices')
            source = order[pos]
            owners = np.searchsorted(offsets[1:], source, side='right')
            positions = [np.flatnonzero(owners == r) for r in range(comm.size)]
            requests = [source[p] - offsets[r] for r, p in enumerate(positions)]
            return positions, requests

        self.positions, requests = collective_call(comm, make_requests)
        self.send_indices = exchange_arrays(comm, requests)
        self.size = requested.size

    def apply(self, values):
        values = np.asarray(values)
        nrhs = values.shape[1]
        sends = collective_call(self.comm, lambda: [
            np.ascontiguousarray(values[idx]).reshape(-1) for idx in self.send_indices])
        received = exchange_arrays(self.comm, sends)
        result = np.empty((self.size, nrhs), dtype=values.dtype)
        for positions, data in zip(self.positions, received):
            result[positions] = data.reshape(-1, nrhs)
        return result


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
        leaders = select_groups(node_ids, size)
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


class ReducedLinearSolver:
    """
    Aggregate application data, call a communicator-aware backend, and return it.
    """
    is_iterative = False

    def __init__(self, gui, engine, app_comm, size, backend_factory):
        self.gui, self.engine = gui, engine
        self.comm = app_comm
        self.ranks = SolverRankLayout(app_comm, size)
        self.backend_factory = backend_factory
        self.backend = None
        self.skip_solve = False
        self.closed = False

    def _call_backend(self, operation):
        return collective_call(self.comm, lambda: operation() if self.ranks.is_leader else None)

    def SetOperator(self, A, dist=False, name=None, ifactor=0):
        def prepare():
            coo = A.tocoo(copy=True)
            if coo.shape[0] != coo.shape[1]:
                raise ValueError('MUMPS requires a square matrix')
            layout = getattr(A, 'solver_layout', None)
            if layout is None:
                if dist:
                    raise ValueError('Distributed COO input requires SolverLayout metadata')
                ids = np.arange(coo.shape[0], dtype=np.int64) if self.comm.rank == 0 else np.empty(0, dtype=np.int64)
                layout = SolverLayout(ids, ids)
            for ids in (layout.equation_ids, layout.solution_ids):
                if np.any(np.asarray(ids) < 0) or np.any(np.asarray(ids) >= coo.shape[0]):
                    raise ValueError('Solver layout index out of range')
            return coo, layout

        coo, self.layout = collective_call(self.comm, prepare)
        meta = self.comm.allgather((coo.shape, coo.dtype.str))
        if any(item != meta[0] for item in meta):
            raise ValueError('Inconsistent matrix dimensions or dtype')
        self.shape = coo.shape
        group = self.ranks.group_comm
        # Centralized input is supplied once, even if callers replicate it.
        take = dist or self.comm.rank == 0
        row = gather_array(group, np.asarray(coo.row if take else [], dtype=np.int64))
        col = gather_array(group, np.asarray(coo.col if take else [], dtype=np.int64))
        data = gather_array(group, coo.data if take else np.empty(0, dtype=coo.dtype))
        equations = gather_array(group, np.asarray(self.layout.equation_ids, dtype=np.int64))
        solutions = gather_array(group, np.asarray(self.layout.solution_ids, dtype=np.int64))
        self.solution_counts = group.allgather(len(self.layout.solution_ids))

        def setup():
            from scipy.sparse import coo_matrix
            matrix = coo_matrix((data, (row, col)), shape=self.shape)
            matrix.solver_layout = SolverLayout(equations, solutions)
            if self.backend is None:
                self.backend = self.backend_factory(self.ranks.solver_comm)
            self.backend.skip_solve = self.skip_solve
            self.backend.SetOperator(matrix, dist=True, name=name, ifactor=ifactor)

        self._call_backend(setup)

    def Mult(self, b, x=None, case_base=0):
        from mpi4py import MPI
        distributed = self.gui.use_dist_rhs
        group = self.ranks.group_comm

        def prepare():
            if distributed or self.comm.rank == 0:
                values = np.asarray(b)
                rows = len(self.layout.equation_ids) if distributed else self.shape[0]
                if values.ndim != 2 or values.shape[0] != rows or values.shape[1] < 1:
                    raise ValueError('RHS dimensions do not match the solver layout')
                return np.ascontiguousarray(values)
            return None

        values = collective_call(self.comm, prepare)
        nrhs = self.comm.bcast(values.shape[1] if self.comm.rank == 0 else None, root=0)
        if distributed:
            meta = self.comm.allgather((values.shape[1], values.dtype.str))
            if len(set(meta)) != 1:
                raise ValueError('Inconsistent RHS count or dtype')
            packed = gather_array(group, values.reshape(-1))
            bb = packed.reshape(-1, nrhs) if self.ranks.is_leader else None
        else:
            bb = values

        def solve():
            self.backend.skip_solve = self.skip_solve
            return self.backend.Mult(bb, x=x, case_base=case_base)

        sol = self._call_backend(solve)
        if not self.gui.use_dist_sol:
            return sol if self.comm.rank == 0 else None
        dtype = group.bcast(sol.dtype.str if self.ranks.is_leader else None, root=0)
        counts, offsets = _counts(np.asarray(self.solution_counts) * nrhs)
        result = collective_call(self.comm, lambda: np.empty(
            (len(self.layout.solution_ids), nrhs), dtype=dtype))
        send = [np.ascontiguousarray(sol).reshape(-1), counts, offsets[:-1],
                MPI._typedict[result.dtype.char]] if self.ranks.is_leader else None
        group.Scatterv(send, result.reshape(-1), root=0)
        return result

    def close(self):
        if not self.closed:
            self._call_backend(lambda: self.backend.close() if self.backend is not None else None)
            self.ranks.close()
            self.closed = True
