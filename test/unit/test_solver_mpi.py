"""
Configuration, placement, and indexed-transfer checks for solver MPI reduction.
"""

from pathlib import Path
from contextlib import nullcontext
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mpi4py import MPI
import numpy as np

from petram.helper.mpi_recipes import IndexedTransfer
from petram.helper.solver_mpi import find_solver_group_leader, SolverRankLayout
from petram.solver.mumps_model import MUMPS, MUMPSSolver


class SolverSizeTests(unittest.TestCase):
    def setUp(self):
        self.model = MUMPS()

    def test_model_overrides_environment_without_validating_it(self):
        self.model.solver_mpi_ranks = 1
        self.assertEqual(
            self.model.resolve_solver_size(
                MPI.COMM_SELF, {'PETRAM_MUMPS_MPI_SIZE': 'invalid'}),
            (1, 'model'))

    def test_default_and_environment(self):
        self.assertEqual(
            self.model.resolve_solver_size(MPI.COMM_SELF, {}),
            (1, 'default'))
        self.assertEqual(
            self.model.resolve_solver_size(
                MPI.COMM_SELF, {'PETRAM_MUMPS_MPI_SIZE': '1'}),
            (1, 'environment'))
        self.assertEqual(self.model.solver_mpi_ranks, -1)

    def test_invalid_values_fail(self):
        for value in (0, -2, 2, 1.5, 'invalid'):
            self.model.solver_mpi_ranks = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.model.resolve_solver_size(MPI.COMM_SELF, {})
        self.model.solver_mpi_ranks = -1
        for value in ('', 'invalid', '0', '2'):
            with self.subTest(env=value), self.assertRaises(ValueError):
                self.model.resolve_solver_size(
                    MPI.COMM_SELF, {'PETRAM_MUMPS_MPI_SIZE': value})

    def test_gui_and_model_persistence(self):
        from petram.model import load_model_json
        model = MUMPS()
        self.assertEqual(model.solver_mpi_ranks, -1)
        model.solver_mpi_ranks = 3
        other = MUMPS()
        other.import_panel1_value(model.get_panel1_value())
        self.assertEqual(other.solver_mpi_ranks, 3)
        with tempfile.TemporaryDirectory() as directory:
            model._generate_script_json(dir=directory)
            restored = load_model_json(Path(directory) / 'model.json')
            self.assertEqual(restored.solver_mpi_ranks, 3)
        self.assertEqual(model.solver_mpi_ranks, 3)


class MUMPSJobTests(unittest.TestCase):
    def test_success_returns_without_warning(self):
        backend = Mock()
        backend.get_infog.return_value = 0
        with patch('petram.solver.mumps_model.dprint1') as log:
            self.assertEqual(MUMPSSolver._run_job(SimpleNamespace(s=backend, _omp_scope=nullcontext), 3), 0)
        backend.set_job.assert_called_once_with(3)
        backend.run.assert_called_once_with()
        log.assert_not_called()

    def test_negative_global_status_raises_with_details(self):
        backend = Mock()
        backend.get_infog.side_effect = lambda index: {1: -10, 2: 4}[index]
        with self.assertRaisesRegex(
                RuntimeError, r'MUMPS job 2: INFOG\(1\)=-10, INFOG\(2\)=4'):
            MUMPSSolver._run_job(SimpleNamespace(s=backend, _omp_scope=nullcontext), 2)

    def test_positive_status_warns_without_aborting(self):
        backend = Mock()
        backend.get_infog.side_effect = lambda index: {1: 1, 2: 4}[index]
        for job in (1, 2, 3, 7, 8):
            with self.subTest(job=job), patch('petram.solver.mumps_model.dprint1') as log:
                self.assertEqual(MUMPSSolver._run_job(SimpleNamespace(s=backend, _omp_scope=nullcontext), job), 1)
                log.assert_called_once_with(
                    f'MUMPS job {job} warning: INFOG(1)=1, INFOG(2)=4')


class SolverPlacementTests(unittest.TestCase):
    def test_equal_nodes(self):
        np.testing.assert_array_equal(find_solver_group_leader([0] * 4 + [4] * 4, 4),
                                      [0, 0, 2, 2, 4, 4, 6, 6])

    def test_fewer_solvers_than_nodes(self):
        np.testing.assert_array_equal(find_solver_group_leader([0, 0, 2, 2, 4, 4], 2),
                                      [0, 0, 2, 2, 0, 2])

    def test_uneven_nodes_and_groups(self):
        np.testing.assert_array_equal(find_solver_group_leader([0] * 5 + [5] * 2, 3),
                                      [0, 0, 2, 2, 2, 5, 5])

    def test_node_counts_exclude_remote_aggregation_members(self):
        for nodes, size in (([0] * 4 + [4] * 4, 4),
                            ([0, 0, 2, 2, 4, 4], 2),
                            ([0] * 5 + [5] * 2, 3)):
            leaders = find_solver_group_leader(nodes, size)
            for rank, node_id in enumerate(nodes):
                node = Mock()
                node.rank = 0
                node.bcast.return_value = node_id
                comm = Mock(size=len(nodes), rank=rank)
                comm.Split_type.return_value = node
                comm.allgather.return_value = nodes
                layout = SolverRankLayout(comm, size)
                self.assertEqual(layout.node_app_size, nodes.count(node_id))
                self.assertEqual(layout.node_solver_size,
                                 sum(nodes[r] == node_id for r in set(leaders)))
                comm.Split_type.assert_called_once()
                comm.allgather.assert_called_once()
                layout.close()

    def test_all_counts_keep_exact_leader_count_and_root(self):
        nodes = [0] * 5 + [5] * 2 + [7]
        for size in range(1, len(nodes) + 1):
            groups = find_solver_group_leader(nodes, size)
            self.assertEqual(len(set(groups)), size)
            self.assertEqual(groups[0], 0)
            for leader in set(groups):
                self.assertEqual(groups[leader], leader)


class IndexedTransferTests(unittest.TestCase):
    def test_repeated_requests_and_multiple_rhs(self):
        transfer = IndexedTransfer(MPI.COMM_SELF, [9, 3, 7], [7, 9, 7])
        result = transfer.apply(np.array([[90., 91.], [30., 31.], [70., 71.]]))
        np.testing.assert_array_equal(result, [[70., 71.], [90., 91.], [70., 71.]])

    def test_missing_or_duplicate_owners_fail(self):
        for owned, requested in (([1, 1], [1]), ([1], [2])):
            with self.subTest(owned=owned), self.assertRaises(RuntimeError):
                IndexedTransfer(MPI.COMM_SELF, owned, requested)

    def test_empty_transfer(self):
        transfer = IndexedTransfer(MPI.COMM_SELF, [], [])
        self.assertEqual(transfer.apply(np.empty((0, 2))).shape, (0, 2))


if __name__ == '__main__':
    unittest.main()
