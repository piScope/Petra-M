"""
Numerical standalone MUMPS checks; run in a fresh MPI Python process.

PYTHONPATH=python OMP_NUM_THREADS=1 mpiexec -n 4 python test/dev_experiment/mpi_mumps_reduction.py
"""

from types import SimpleNamespace
from unittest.mock import patch
import itertools
from pathlib import Path
import os
import shutil
import tempfile

from mpi4py import MPI
import numpy as np
from scipy.sparse import coo_matrix

import petram.mfem_config
petram.mfem_config.use_parallel = True

from petram.solver.mumps_model import MUMPS, MUMPSSolver, ReducedMUMPSSolver
from petram.solver.solver_model import LinearSolver
from petram.helper.solver_mpi import SolverLayout
from petram.helper.mpi_recipes import collective_call
import petram.debug
petram.debug.set_debug_level(0)


comm = MPI.COMM_WORLD


def run_case(size, distributed_rhs, distributed_sol, complex_values=False,
             single=False, skip=False, ordering='auto', out_of_core=False,
             blr=False, write_output=False, save_restore=False,
             maximize_omp_threads=False):
    """
    Check two RHS solves and refactorization against an independent dense solve.
    """
    n = 11
    matrix = np.diag(np.arange(n, dtype=float) + 5)
    matrix += np.diag(np.full(n - 1, -0.5), 1)
    matrix += np.diag(np.full(n - 1, -0.25), -1)
    if complex_values:
        matrix = matrix + 0.3j * np.eye(n)
    exact = np.column_stack((np.arange(n) + 1., np.cos(np.arange(n))))
    if complex_values:
        exact = exact + 0.2j * exact[::-1]

    # Leave the last application rank empty, use noncontiguous equation IDs,
    # and request a replicated auxiliary unknown on every rank.
    owners = max(1, comm.size - 1)
    ids = np.arange(n, dtype=np.int64)[comm.rank::owners] if comm.rank < owners else np.empty(0, dtype=np.int64)
    requested = np.r_[ids[::-1][ids[::-1] != n - 1], n - 1]
    gui = MUMPS()
    gui.solver_mpi_ranks = size
    gui.maximize_omp_threads = maximize_omp_threads
    gui.use_dist_rhs = distributed_rhs
    gui.use_dist_sol = distributed_sol
    gui.use_single_precision = single
    gui.ordering = ordering
    gui.out_of_core = out_of_core
    gui.use_blr = blr
    gui.write_mat = write_output
    gui.write_inv = write_output
    gui.get_solve_root = lambda: SimpleNamespace(
        get_linearsystem_type_from_modeltree=lambda: 'coo')
    solver = gui.allocate_solver(complex_values)
    reduced = size < comm.size
    solver.skip_solve = skip
    try:
        assert isinstance(solver, ReducedMUMPSSolver if reduced else MUMPSSolver)
        assert isinstance(solver, LinearSolver)
        assert not solver.is_iterative
        for factor in (1., 1.5):
            gui.write_fac = save_restore
            gui.restore_fac = False
            gui.factor_path = str(Path.cwd() / 'factors' / f'saved_{factor}*')
            operator = matrix * factor
            rows = np.repeat(ids, n)
            cols = np.tile(np.arange(n), ids.size)
            local = coo_matrix((operator[ids].reshape(-1), (rows, cols)), shape=(n, n))
            local.solver_layout = SolverLayout(ids, requested)
            solver.SetOperator(local, dist=True)
            first_backend = solver.backend if reduced else solver
            rhs_transfer = None
            for scale in (1., 2.):
                rhs = operator @ (exact * scale)
                values = rhs[ids]
                result = solver.Mult(values)
                backend = solver.backend if reduced else solver
                current_transfer = backend._rhs_transfer if backend is not None else None
                if scale == 1.:
                    rhs_transfer = current_transfer

                def check():
                    assert backend is first_backend
                    if backend is not None:
                        assert backend.comm.size == size
                        assert backend._rhs_transfer is rhs_transfer
                        assert (rhs_transfer is None) == distributed_rhs
                    else:
                        assert reduced and not solver.ranks.is_leader
                    expected = np.zeros_like(exact) if skip else np.linalg.solve(operator, rhs)
                    if distributed_sol:
                        expected = expected[requested]
                    elif comm.rank != 0:
                        assert result is None
                        return
                    np.testing.assert_allclose(result, expected,
                                               rtol=5e-5 if single else 1e-11,
                                               atol=5e-5 if single else 1e-11)

                collective_call(comm, check)
            if save_restore:
                solver.close()
                gui.write_fac = False
                gui.restore_fac = True
                solver = gui.allocate_solver(complex_values)
                solver.SetOperator(local, dist=True)
                restored = solver.Mult(values)

                def check_restore():
                    if distributed_sol:
                        expected = np.linalg.solve(operator, rhs)[requested]
                    elif comm.rank == 0:
                        expected = np.linalg.solve(operator, rhs)
                    else:
                        assert restored is None
                        return
                    np.testing.assert_allclose(restored, expected, rtol=1e-11, atol=1e-11)

                collective_call(comm, check_restore)
        if write_output:
            comm.Barrier()

            def check_files():
                if comm.rank != 0:
                    return
                files = sorted(Path.cwd().glob('matrix.??????'))
                assert len(files) == size
                saved_matrix = np.zeros_like(operator)
                for file in files:
                    if file.stat().st_size:
                        entries = np.loadtxt(file, ndmin=2)
                        entries_values = entries[:, 2] + 1j * entries[:, 3]
                        np.add.at(saved_matrix, (entries[:, 0].astype(int), entries[:, 1].astype(int)),
                                  entries_values if complex_values else entries_values.real)
                np.testing.assert_allclose(saved_matrix, operator)
                for column in range(2):
                    saved_rhs = np.zeros(n, dtype=operator.dtype)
                    files = sorted(Path.cwd().glob(f'rhs_{column}.??????'))
                    assert len(files) == (size if distributed_rhs else 1)
                    for file in files:
                        if not file.stat().st_size:
                            continue
                        entries = np.loadtxt(file, ndmin=2)
                        realcol = 2 if distributed_rhs else 1
                        entries_values = entries[:, realcol].copy()
                        if complex_values:
                            entries_values = entries_values + 1j * entries[:, realcol + 1]
                        np.add.at(saved_rhs, entries[:, 0].astype(int), entries_values)
                    np.testing.assert_allclose(saved_rhs, rhs[:, column], rtol=1e-7, atol=1e-7)
                inverse = np.empty_like(operator)
                indices = []
                files = sorted(Path.cwd().glob('matrix_inv_idx.*.npz'))
                assert len(files) == size
                for file in files:
                    ids = np.load(file)['A_inv_idx'] - 1
                    datafile = file.with_name(file.name.replace('matrix_inv_idx.', 'matrix_inv.'))
                    inverse[ids] = np.load(datafile)['A_inv']
                    indices.extend(ids)
                np.testing.assert_array_equal(sorted(indices), np.arange(n))
                np.testing.assert_allclose(inverse, np.linalg.inv(operator), rtol=1e-11, atol=1e-11)

            collective_call(comm, check_files)
    finally:
        solver.close()
        solver.close()


def main():
    """
    Exercise reduced and unreduced ranks with every RHS/output combination.
    """
    cases = 0
    for size, rhs, sol, complex_values in itertools.product(
            sorted({1, max(1, comm.size - 1), comm.size}),
            (False, True), (False, True), (False, True)):
        run_case(size, rhs, sol, complex_values)
        cases += 1
    for rhs, sol, single, skip in itertools.product(
            (False, True), (False, True), (False, True), (False, True)):
        run_case(min(2, comm.size), rhs, sol, single=single, skip=skip)
        cases += 1
    for options in ({'out_of_core': True}, {'blr': True},
                    {'ordering': 'ParMetis'}, {'ordering': 'PT-Scotch'}):
        run_case(min(2, comm.size), True, True, **options)
        cases += 1
    for size, convert_real, rhs, sol in itertools.product(
            sorted({min(2, comm.size), comm.size}),
            (False, True), (False, True), (False, True)):
        run_engine_case(size, convert_real, rhs, sol)
        cases += 1
    for rhs, complex_values in itertools.product((False, True), repeat=2):
        run_native_case(rhs, complex_values)
        cases += 1
    for rhs, sol, complex_values in itertools.product((False, True), repeat=3):
        directory = comm.bcast(tempfile.mkdtemp(prefix='base08-mumps-') if comm.rank == 0 else None)
        previous = Path.cwd()
        os.chdir(directory)
        try:
            run_case(min(2, comm.size), rhs, sol, complex_values,
                     write_output=True, save_restore=True)
            cases += 1
        finally:
            os.chdir(previous)
            comm.Barrier()
            if comm.rank == 0:
                shutil.rmtree(directory)
    if comm.rank == 0:
        print(f'PASS: {cases} MUMPS cases on {comm.size} application ranks', flush=True)


def run_engine_case(size, convert_real, distributed_rhs, distributed_sol,
                    maximize_omp_threads=False):
    """
    Finalize and restore real Hypre field blocks plus a replicated auxiliary DOF.
    """
    from scipy.sparse import csr_matrix
    from mfem.common.parcsr_extra import ToHypreParCSR, ToHypreParVec
    from mfem.common.chypre import CHypreMat, CHypreVec
    from petram.engine import Engine
    from petram.helper.block_matrix import BlockMatrix, ScipyCoo

    n = 9
    partitions = np.linspace(0, n, comm.size + 1, dtype=int)
    start, stop = partitions[comm.rank:comm.rank + 2]
    matrix = np.diag(np.arange(n + 1) + 5.)
    matrix[:n, n] = 0.2
    matrix[n, :n] = 0.3
    matrix = matrix + 0.1j * np.eye(n + 1)
    exact = np.arange(n + 1) + 1. + 0.3j
    rhs = matrix @ exact
    blocks = BlockMatrix((2, 2), kind='hypre', complex=True)
    real = ToHypreParCSR(csr_matrix(matrix[start:stop, :n].real))
    imag = ToHypreParCSR(csr_matrix(matrix[start:stop, :n].imag))
    blocks[0, 0] = CHypreMat(real, imag)
    blocks[0, 1] = ScipyCoo(matrix[:n, n:])
    blocks[1, 0] = ScipyCoo(matrix[n:, :n])
    blocks[1, 1] = ScipyCoo(matrix[n:, n:])
    load = BlockMatrix((2, 1), kind='hypre', complex=True)
    load[0] = CHypreVec(ToHypreParVec(rhs[start:stop].real),
                       ToHypreParVec(rhs[start:stop].imag))
    load[1] = rhs[n:].reshape(-1, 1)
    initial = load * 0.
    engine = Engine.__new__(Engine)
    mask = ([True, True], [True, True])
    fmt = 'coo_real' if convert_real else 'coo'
    operator = engine.finalize_matrix(blocks, mask, True, format=fmt, verbose=False)
    with patch.object(CHypreVec, 'GlobalVector', side_effect=AssertionError('RHS all-gather')):
        values = engine.finalize_rhs([load], blocks, initial, mask, True,
                                     format=fmt, verbose=False)
    assert values.shape == (len(operator.solver_layout.equation_ids), 1)
    gui = MUMPS()
    gui.solver_mpi_ranks = size
    gui.maximize_omp_threads = maximize_omp_threads
    gui.use_dist_rhs = distributed_rhs
    gui.use_dist_sol = distributed_sol
    gui.get_solve_root = lambda: SimpleNamespace(
        get_linearsystem_type_from_modeltree=lambda: fmt)
    solver = gui.allocate_solver(not convert_real)
    try:
        solver.SetOperator(operator, dist=True)
        result = solver.Mult(values)
        if convert_real:
            result = gui.real_to_complex(result)
        if distributed_sol:
            restored = BlockMatrix((2, 1), kind='hypre', complex=True)
            blocks.reformat_distributed_mat(result, 0, restored, mask)

            def check():
                np.testing.assert_allclose(restored[0].toarray().reshape(-1), exact[start:stop])
                np.testing.assert_allclose(restored[1].toarray().reshape(-1), exact[n:])
        else:
            def check():
                if comm.rank == 0:
                    np.testing.assert_allclose(result[:, 0], exact)
                else:
                    assert result is None
        collective_call(comm, check)
    finally:
        solver.close()


def run_native_case(distributed_rhs, complex_values):
    """
    Preserve the direct backend input contract used by preconditioners.
    """
    n = 9
    ids = np.array_split(np.arange(n, dtype=np.int64), comm.size)[comm.rank]
    diagonal = np.arange(n) + 3.
    exact = np.arange(n) + 1.
    if complex_values:
        diagonal = diagonal + 0.2j
        exact = exact + 0.3j
    rhs = (diagonal * exact).reshape(-1, 1)
    matrix = coo_matrix((diagonal[ids], (ids, ids)), shape=(n, n))
    gui = MUMPS()
    gui.use_dist_rhs = distributed_rhs
    solver = MUMPSSolver(gui, None, comm=comm)
    try:
        solver.AllocSolver(complex_values, False)
        solver.SetOperator(matrix, dist=True)
        values = rhs[ids] if distributed_rhs else (rhs if comm.rank == 0 else None)
        result = solver.Mult(values)
        collective_call(comm, lambda: np.testing.assert_allclose(result[:, 0], exact[ids]))
        assert solver._rhs_transfer is None
    finally:
        solver.close()


if __name__ == '__main__':
    main()
