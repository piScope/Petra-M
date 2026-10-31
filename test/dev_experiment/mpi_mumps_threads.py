"""
Single-node native thread/restoration checks; run with eight ranks and OMP=2.

PYTHONPATH=python OMP_NUM_THREADS=2 mpiexec -n 8 python test/dev_experiment/mpi_mumps_threads.py
"""
from contextlib import ExitStack
import itertools
from unittest.mock import patch

from mpi4py import MPI

from petram.ext.mumps import mumps_solve
from mpi_mumps_reduction import run_case, run_engine_case


comm = MPI.COMM_WORLD
node = comm.Split_type(MPI.COMM_TYPE_SHARED)
assert node.size == comm.size, 'This test expects one physical node'
node.Free()
initial = mumps_solve.get_omp_max_threads()
assert initial == 2, 'Run with OMP_NUM_THREADS=2'
events = []
expected = None


def observe(kind):
    """
    Observe the actual setting at each native MUMPS entry point.
    """
    count = mumps_solve.get_omp_max_threads()
    assert count == expected, (comm.rank, kind, count, expected)
    events.append(kind)


def observed_constructor(native):
    """
    Check JOB -1 before constructing a native arithmetic variant.
    """
    def construct(*args, **kwargs):
        observe('initialize')
        return native(*args, **kwargs)
    return construct


def observed_method(method, kind):
    """
    Check run/finish without replacing the underlying numerical operation.
    """
    def call(self, *args, **kwargs):
        observe(kind)
        return method(self, *args, **kwargs)
    return call


with ExitStack() as stack:
    for name in ('DMUMPS', 'ZMUMPS', 'SMUMPS', 'CMUMPS'):
        native = getattr(mumps_solve, name)
        stack.enter_context(patch.object(native, 'run', observed_method(native.run, 'job')))
        stack.enter_context(patch.object(native, 'finish', observed_method(native.finish, 'finish')))
        stack.enter_context(patch.object(mumps_solve, name, observed_constructor(native)))
    cases = 0
    for active, size, complex_values, single, rhs in itertools.product(
            (False, True), sorted({1, min(2, comm.size), comm.size}),
            (False, True), (False, True), (False, True)):
        expected = initial * comm.size // size if active else initial
        events.clear()
        with patch.object(mumps_solve, 'set_omp_num_threads',
                          wraps=mumps_solve.set_omp_num_threads) as set_threads:
            run_case(size, rhs, True, complex_values, single,
                     maximize_omp_threads=active)
            if not active or size == comm.size:
                set_threads.assert_not_called()
            elif events:
                assert set_threads.called
        assert mumps_solve.get_omp_max_threads() == initial
        if events:
            assert {'initialize', 'job', 'finish'} <= set(events)
        cases += 1
    for size, convert_real in itertools.product(sorted({1, comm.size}), (False, True)):
        expected = initial * comm.size // size
        run_engine_case(size, convert_real, True, True, maximize_omp_threads=True)
        assert mumps_solve.get_omp_max_threads() == initial
        cases += 1
if comm.rank == 0:
    print(f'PASS: {cases} thread-control cases on {comm.size} application ranks; '
          f'initial={initial}, reduced-to-one={initial * comm.size}', flush=True)
