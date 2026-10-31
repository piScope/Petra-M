"""Headless checks for the MUMPS GUI control values and solver gate."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.sparse import coo_matrix

from petram.solver.mumps_model import MUMPS, MUMPSSolver


class MUMPSGUIControlTests(unittest.TestCase):
    def _configured_backend(self, *, gpu_enabled, numerical_pivoting,
                            static_pivoting):
        model = MUMPS()
        model.use_gpu_offlading = gpu_enabled
        model.gpus_per_rank = 2
        model.custom_n_pivoting_thrs = numerical_pivoting
        model.cntl1 = "0.125"
        model.use_static_pivoting = static_pivoting
        model.cntl4 = "0.25"

        comm = SimpleNamespace(rank=0, size=1, Barrier=Mock())
        solver = MUMPSSolver(model, None, comm=comm)
        solver.s = Mock()
        solver.is_complex = False
        solver.data_array = lambda values: values
        solver.silent = True
        with patch.object(solver, "_run_job"):
            solver.SetOperator(coo_matrix(np.eye(2)), dist=False)
        return solver.s

    def test_panel_round_trip_preserves_options_and_checkbox_values(self):
        model = MUMPS()
        model.use_gpu_offlading = False
        model.gpus_per_rank = 3
        model.reduce_mpi_proc = False
        model.solver_mpi_ranks = 2
        model.maximize_omp_threads = True
        model.custom_n_pivoting_thrs = True
        model.cntl1 = "-1"
        model.use_static_pivoting = True
        model.cntl4 = "0.01"
        model.icntl10 = "default"
        model.cntl2 = "default"
        model.icntl6 = "2"
        model.icntl8 = "4"

        restored = MUMPS()
        restored.import_panel1_value(model.get_panel1_value())
        restored.import_panel2_value(model.get_panel2_value())

        # Turning a checkbox off keeps its dependent GUI values for later use.
        self.assertFalse(restored.use_gpu_offlading)
        self.assertEqual(restored.gpus_per_rank, 3)
        self.assertFalse(restored.reduce_mpi_proc)
        self.assertEqual(restored.solver_mpi_ranks, 2)
        self.assertTrue(restored.maximize_omp_threads)

        # A checked custom option remains checked even when its value is -1.
        self.assertTrue(restored.custom_n_pivoting_thrs)
        self.assertEqual(restored.cntl1, "-1")
        self.assertTrue(restored.use_static_pivoting)
        self.assertEqual(restored.cntl4, "0.01")
        self.assertEqual(restored.icntl10, "default")
        self.assertEqual(restored.cntl2, "default")
        self.assertEqual(restored.icntl6, "2")
        self.assertEqual(restored.icntl8, "4")

    def test_solver_applies_checkbox_gates_to_effective_controls(self):
        backend = self._configured_backend(
            gpu_enabled=False, numerical_pivoting=False, static_pivoting=False)
        backend.set_icntl.assert_any_call(51, -1)
        backend.set_cntl.assert_any_call(1, -1.0)
        backend.set_cntl.assert_any_call(4, -1.0)

        backend = self._configured_backend(
            gpu_enabled=True, numerical_pivoting=True, static_pivoting=True)
        backend.set_icntl.assert_any_call(51, 2)
        backend.set_cntl.assert_any_call(1, 0.125)
        backend.set_cntl.assert_any_call(4, 0.25)

    def test_rank_reduction_checkbox_gates_reduced_solver_creation(self):
        model = MUMPS()
        model.solver_mpi_ranks = 2
        model.get_solve_root = lambda: SimpleNamespace(
            get_linearsystem_type_from_modeltree=lambda: "coo")
        app_comm = SimpleNamespace(size=4, rank=0)

        with patch("petram.solver.mumps_model.use_parallel", True), \
             patch("mpi4py.MPI.COMM_WORLD", app_comm), \
             patch.object(MUMPSSolver, "AllocSolver") as alloc_solver, \
             patch("petram.solver.mumps_model.ReducedMUMPSSolver") as reduced_solver:
            # A remembered rank count is ignored while the checkbox is off.
            model.reduce_mpi_proc = False
            solver = model.allocate_solver()
            self.assertIsInstance(solver, MUMPSSolver)
            self.assertIs(solver.comm, app_comm)
            alloc_solver.assert_called_once_with(False, model.use_single_precision)
            reduced_solver.assert_not_called()

            # Enabling it uses the remembered count and creates the reduced path.
            model.reduce_mpi_proc = True
            reduced_solver.return_value = Mock(name="reduced_solver")
            solver = model.allocate_solver()
            self.assertIs(solver, reduced_solver.return_value)
            reduced_solver.assert_called_once_with(
                model, None, app_comm, 2, False)


if __name__ == "__main__":
    unittest.main()
