"""
Standalone MUMPS thread policy and restoration checks; requires the DS API.
"""
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mpi4py import MPI

from petram.ext.mumps import mumps_solve
from petram.solver.mumps_model import (MUMPS, MUMPSPreconditionerModel,
                                       MUMPSSolver, MUMPSBlockSolver, ReducedMUMPSSolver)


class MUMPSThreadTests(unittest.TestCase):
    def setUp(self):
        self.model = MUMPS()
        self.initial = mumps_solve.get_omp_max_threads()
        self.addCleanup(mumps_solve.set_omp_num_threads, self.initial)
        mumps_solve.set_omp_num_threads(2)

    def test_default_does_not_query_or_change_runtime(self):
        self.assertFalse(self.model.maximize_omp_threads)
        with patch.object(mumps_solve, 'get_omp_max_threads', side_effect=AssertionError):
            self.model.get_solve_root = lambda: SimpleNamespace(
                get_linearsystem_type_from_modeltree=lambda: 'coo')
            solver = self.model.allocate_solver()
            self.assertIsNone(solver.omp_threads)
            solver.close()
            with MUMPSSolver(self.model, None)._omp_scope():
                pass
            solver = MUMPSSolver(
                self.model, None,
                rank_layout=SimpleNamespace(node_app_size=8, node_solver_size=1))
            self.assertIsNone(solver.omp_threads)

    def test_boolean_panel_and_persistence(self):
        from petram.model import load_model_json
        self.model.maximize_omp_threads = True
        self.model.solver_mpi_ranks = 3
        other = MUMPS()
        other.import_panel1_value(self.model.get_panel1_value())
        self.assertTrue(other.maximize_omp_threads)
        self.assertEqual(other.solver_mpi_ranks, 3)
        self.assertEqual(len(other.panel1_param()), len(other.get_panel1_value()))
        self.assertEqual(len(other.panel1_param()), len(other.panel1_tip()))
        self.assertTrue(pickle.loads(pickle.dumps(self.model)).maximize_omp_threads)
        with tempfile.TemporaryDirectory() as directory:
            self.model._generate_script_json(dir=directory)
            restored = load_model_json(Path(directory) / 'model.json')
            self.assertTrue(restored.maximize_omp_threads)
        del self.model.maximize_omp_threads
        self.assertFalse(pickle.loads(pickle.dumps(self.model)).maximize_omp_threads)

    def test_node_budget_and_uneven_division(self):
        self.model.maximize_omp_threads = True
        for app_size, solver_size, active, expected in (
                (8, 1, True, 16), (8, 2, True, 8),
                (8, 8, True, 2), (5, 2, True, 5),
                (8, 2, False, None), (8, 0, False, None)):
            ranks = SimpleNamespace(node_app_size=app_size,
                                    node_solver_size=solver_size,
                                    is_leader=active)
            with self.subTest(app_size=app_size, solver_size=solver_size, active=active):
                if active:
                    solver = MUMPSSolver(self.model, None,
                                         rank_layout=ranks)
                    self.assertEqual(solver.omp_threads, expected)
                else:
                    with patch('petram.solver.mumps_model.SolverRankLayout', return_value=ranks), \
                         patch.object(mumps_solve, 'get_omp_max_threads', side_effect=AssertionError):
                        solver = ReducedMUMPSSolver(self.model, None, MPI.COMM_SELF, 1, False)
                    self.assertIsNone(solver.backend)

    def test_scope_restores_after_success_and_exception(self):
        self.model.maximize_omp_threads = True
        solver = MUMPSSolver(self.model, None,
                             rank_layout=SimpleNamespace(node_app_size=2, node_solver_size=1))
        with solver._omp_scope():
            self.assertEqual(mumps_solve.get_omp_max_threads(), 4)
        self.assertEqual(mumps_solve.get_omp_max_threads(), 2)
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with solver._omp_scope():
                self.assertEqual(mumps_solve.get_omp_max_threads(), 4)
                raise RuntimeError('injected')
        self.assertEqual(mumps_solve.get_omp_max_threads(), 2)

    def test_initialization_failure_restores(self):
        self.model.maximize_omp_threads = True
        solver = MUMPSSolver(self.model, None,
                             rank_layout=SimpleNamespace(node_app_size=2, node_solver_size=1))

        def fail(*args):
            self.assertEqual(mumps_solve.get_omp_max_threads(), 4)
            raise RuntimeError('injected initialization failure')

        with patch.object(mumps_solve, 'DMUMPS', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'initialization failure'):
                solver.AllocSolver(False, False)
        self.assertEqual(mumps_solve.get_omp_max_threads(), 2)

    def test_native_job_and_cleanup_failures_restore(self):
        self.model.maximize_omp_threads = True
        solver = MUMPSSolver(self.model, None,
                             rank_layout=SimpleNamespace(node_app_size=2, node_solver_size=1))
        solver.s = Mock()

        def fail():
            self.assertEqual(mumps_solve.get_omp_max_threads(), 4)
            raise RuntimeError('injected native failure')

        solver.s.run.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, 'native failure'):
            solver._run_job(2)
        self.assertEqual(mumps_solve.get_omp_max_threads(), 2)
        solver.s.run.side_effect = None
        solver.s.get_infog.return_value = -10
        with self.assertRaisesRegex(RuntimeError, 'INFOG'):
            solver._run_job(2)
        self.assertEqual(mumps_solve.get_omp_max_threads(), 2)
        solver.s.finish.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, 'native failure'):
            solver.close()
        self.assertEqual(mumps_solve.get_omp_max_threads(), 2)
        solver.s = None

    def test_rhs_distribution_checks_status_and_restores(self):
        self.model.maximize_omp_threads = True
        solver = MUMPSSolver(self.model, None,
                             rank_layout=SimpleNamespace(node_app_size=2, node_solver_size=1))
        solver.silent = True
        solver.s = Mock()
        solver.s.get_info.return_value = 2
        solver.s.get_infog.side_effect = lambda index: {1: -10, 2: 4}[index]
        solver.s.run.side_effect = lambda: self.assertEqual(
            mumps_solve.get_omp_max_threads(), 4)

        with self.assertRaisesRegex(RuntimeError, r'MUMPS job 9: INFOG\(1\)=-10'):
            solver.read_rhs_distribution(solver.s)
        solver.s.set_job.assert_called_once_with(9)
        solver.s.run.assert_called_once_with()
        self.assertEqual(mumps_solve.get_omp_max_threads(), 2)

    def test_full_rank_allocation_leaves_threads_alone(self):
        self.model.maximize_omp_threads = True
        self.model.get_solve_root = lambda: SimpleNamespace(
            get_linearsystem_type_from_modeltree=lambda: 'coo')
        for parallel in (False, True):
            with self.subTest(parallel=parallel), \
                 patch('petram.solver.mumps_model.use_parallel', parallel), \
                 patch.object(mumps_solve, 'get_omp_max_threads', side_effect=AssertionError), \
                 patch.object(mumps_solve, 'set_omp_num_threads', side_effect=AssertionError):
                solver = self.model.allocate_solver()
                try:
                    self.assertIsNone(solver.omp_threads)
                    with solver._omp_scope():
                        pass
                finally:
                    solver.close()

    def test_block_and_preconditioner_paths_remain_unchanged(self):
        self.model.maximize_omp_threads = True
        self.model.get_solve_root = lambda: SimpleNamespace(
            get_linearsystem_type_from_modeltree=lambda: 'blk_interleave')
        with patch.object(mumps_solve, 'get_omp_max_threads', side_effect=AssertionError):
            self.assertIsInstance(self.model.allocate_solver(), MUMPSBlockSolver)
        preconditioner = MUMPSPreconditionerModel()
        self.assertNotIn('maximize_omp_threads', preconditioner.attribute_set({}))
        self.assertIsNone(MUMPSSolver(self.model, None).omp_threads)


if __name__ == '__main__':
    unittest.main()
