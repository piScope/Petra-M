"""
Check MPI vector recipes with default, split, and single-rank communicators.

PYTHONPATH=python mpiexec -n 4 python test/dev_experiment/mpi_recipes_subcomm.py
"""

from contextlib import redirect_stderr
from io import StringIO

from mpi4py import MPI
import numpy as np

import petram.mfem_config
petram.mfem_config.use_parallel = True

from petram.helper.mpi_recipes import (
    allgather_vector, alltoall_vector, collective_call, gather_vector, scatter_vector,
)


def check_collective_call(comm):
    """
    Return local results and propagate one rank's failure to every member.
    """
    assert collective_call(comm, lambda: comm.rank + 10) == comm.rank + 10

    def fail():
        if comm.rank == comm.size - 1:
            raise ValueError('local failure')

    captured = StringIO()
    with redirect_stderr(captured):
        try:
            collective_call(comm, fail)
        except RuntimeError as exc:
            if comm.rank == comm.size - 1:
                assert str(exc).startswith(f'MPI rank {comm.rank}:\nTraceback')
                assert 'ValueError: local failure' in str(exc)
            else:
                assert str(exc) == f'MPI rank {comm.size - 1}: ValueError: local failure'
        else:
            raise AssertionError('Local failure was not propagated')
    assert captured.getvalue() == ''


def check_recipes(comm, dtype, empty, use_default=False):
    """
    Compare transfers against values constructed from source and destination IDs.
    """
    kwargs = {} if use_default else {'comm': comm}
    gather_kwargs = {} if use_default else {'world': comm}
    root = comm.size - 1

    def rows(rank):
        return 0 if empty else (rank + 1) % 3

    def block(rank):
        values = np.arange(rows(rank) * 2).reshape(-1, 2) + 100 * rank
        if np.issubdtype(dtype, np.complexfloating):
            values = values + 0.25j * values
        return values.astype(dtype)

    local = block(comm.rank)
    expected = np.concatenate([block(rank) for rank in range(comm.size)])
    np.testing.assert_array_equal(allgather_vector(local, **kwargs), expected)
    counts = np.array([rows(rank) for rank in range(comm.size)])
    np.testing.assert_array_equal(
        allgather_vector(local, rcounts=counts, **kwargs), expected)
    np.testing.assert_array_equal(counts, [rows(rank) for rank in range(comm.size)])

    gathered = gather_vector(local.reshape(-1), root=root, **gather_kwargs)
    if comm.rank == root:
        np.testing.assert_array_equal(gathered, expected.reshape(-1))
    else:
        assert gathered is None

    sent = expected.reshape(-1) if comm.rank == root else None
    received = scatter_vector(sent, rcounts=local.size, root=root, **kwargs)
    np.testing.assert_array_equal(received, local.reshape(-1))

    def message(source, destination):
        count = 0 if empty else (source + 2 * destination) % 4
        values = np.arange(count) + 100 * source + 10 * destination
        if np.issubdtype(dtype, np.complexfloating):
            values = values + 0.5j * values
        return values.astype(dtype)

    messages = [message(comm.rank, destination) for destination in range(comm.size)]
    received = alltoall_vector(messages, dtype, **kwargs)
    assert len(received) == comm.size
    for source, values in enumerate(received):
        np.testing.assert_array_equal(values, message(source, comm.rank))


def main():
    """
    Include uneven groups, nonzero local roots, empty ranks, and empty transfers.
    """
    world = MPI.COMM_WORLD
    group = world.Split(int(world.rank == 0), key=world.rank)
    cases = 0
    try:
        for comm, use_default in ((world, True), (group, False), (MPI.COMM_SELF, False)):
            check_collective_call(comm)
            for dtype in (np.int32, np.int64, np.float32, np.float64,
                          np.complex64, np.complex128):
                for empty in (False, True):
                    check_recipes(comm, dtype, empty, use_default)
                    cases += 1
    finally:
        group.Free()
    world.Barrier()
    if world.rank == 0:
        print(f'PASS: {cases} MPI recipe cases on {world.size} application ranks')


if __name__ == '__main__':
    main()
