"""
Collective CSR factorization and repeated solves with SuperLU_DIST.
"""
import operator
import warnings

import numpy as np
from mpi4py import MPI

from .superlu_solve import libsuperlu_solve

_ORDERINGS = {"natural": 0, "mmd_ata": 1, "mmd_at_plus_a": 2, "metis": 4}
_INT_MAX = np.iinfo(np.int32).max


def _checked(comm, operation):
    """
    Raise the first local input error on every participating rank.
    """
    result = None
    error = None
    try:
        result = operation()
    except Exception as exc:
        error = str(exc) or type(exc).__name__
    errors = comm.allgather(error)
    for rank, message in enumerate(errors):
        if message is not None:
            raise ValueError(f"SuperLU_DIST rank {rank}: {message}")
    return result


class Factor:
    """
    Own factors for canonical local CSR rows. Construction, solve, and close
    are collective. Only float64 and complex128 values are accepted.

    Row ranges must cover the square global matrix in communicator-rank order.
    Input arrays are copied. Use a context manager or explicit close() before
    MPI finalization; garbage collection does not perform collective cleanup.
    """

    def __init__(self, indptr, indices, data, *, global_size, first_row=0,
                 comm=None, ordering="metis", equil=True,
                 replace_tiny_pivot=False, refinement=False, log_statistics=False):
        self._closed = True
        self._native = None
        self.backward_error = None
        comm = MPI.COMM_SELF if comm is None else comm
        if MPI.Is_finalized() or not MPI.Is_initialized():
            raise RuntimeError("An active MPI runtime is required")
        if not isinstance(comm, MPI.Comm) or comm == MPI.COMM_NULL or comm.Is_inter():
            raise ValueError("Expected an active mpi4py intracommunicator")
        self._comm = comm.Dup()
        self._closed = False
        try:
            def prepare():
                n, first = operator.index(global_size), operator.index(first_row)
                if n <= 0 or n > _INT_MAX:
                    raise ValueError("Global size must be positive and fit native int32 indices")
                ptr, col, values = np.asarray(indptr), np.asarray(indices), np.asarray(data)
                if ptr.ndim != 1 or col.ndim != 1 or values.ndim != 1 or ptr.size < 1:
                    raise ValueError("CSR arrays must be one-dimensional with a nonempty row pointer")
                if ptr.dtype.kind not in "iu" or col.dtype.kind not in "iu":
                    raise ValueError("CSR indices must be integers")
                if values.dtype not in (np.dtype("float64"), np.dtype("complex128")):
                    raise ValueError("SuperLU_DIST supports only float64 and complex128")
                rows = ptr.size-1
                if first < 0 or first+rows > n or col.size != values.size:
                    raise ValueError("Invalid local CSR dimensions")
                if col.size > _INT_MAX:
                    raise ValueError("Local CSR nonzero count must fit native int32 indices")
                if ptr[0] != 0 or ptr[-1] != col.size or np.any(ptr[1:] < ptr[:-1]):
                    raise ValueError("Invalid CSR row pointers")
                if np.any(col < 0) or np.any(col >= n) or not np.isfinite(values).all():
                    raise ValueError("Invalid CSR column or nonfinite matrix value")
                if ordering not in _ORDERINGS:
                    raise ValueError("Unsupported ordering: " + str(ordering))
                for option in (equil, replace_tiny_pivot, refinement, log_statistics):
                    if not isinstance(option, (bool, np.bool_)):
                        raise ValueError("Numerical switches must be boolean")
                return (n, first, rows, values.dtype,
                        np.require(ptr, dtype=np.int32, requirements=['C', 'A']),
                        np.require(col, dtype=np.int32, requirements=['C', 'A']),
                        np.require(values, requirements=['C', 'A']))

            n, first, rows, dtype, ptr, col, values = _checked(self._comm, prepare)
            settings = (n, dtype.str, ordering, bool(equil),
                        bool(replace_tiny_pivot), bool(refinement), bool(log_statistics))
            layouts = self._comm.allgather((settings, first, rows, col.size))
            offset = 0
            total_nnz = 0
            for rank_settings, rank_first, rank_rows, rank_nnz in layouts:
                if rank_settings != settings or rank_first != offset:
                    raise ValueError("Ranks must agree on options and own contiguous CSR row ranges")
                offset += rank_rows
                total_nnz += rank_nnz
            if offset != n:
                raise ValueError("Local row ranges must cover the global matrix")
            if total_nnz > _INT_MAX:
                raise ValueError("Global CSR nonzero count must fit native int32 indices")
            self.global_size, self.first_row, self.local_rows = n, first, rows
            self.dtype = dtype
            self._complex = dtype == np.dtype("complex128")
            size = self._comm.size
            nprow = max(p for p in range(1, int(size**0.5)+1) if size % p == 0)
            self.process_grid = (nprow, size//nprow)
            self._native = _checked(self._comm, libsuperlu_solve)
            self._native.initialize(self._comm, self._complex, _ORDERINGS[ordering],
                                    equil, replace_tiny_pivot, refinement, log_statistics)
            self._native.factor_csr(n, first, ptr, col, values)
        except Exception:
            self.close()
            raise

    def solve(self, rhs):
        """
        Return the local solution, preserving rhs. Accept a vector or a matrix
        of RHS columns; repeated calls may use different numbers of columns.
        """
        if self._closed:
            raise RuntimeError("SuperLU_DIST owner is closed")

        def prepare():
            data = np.asarray(rhs)
            if data.dtype != self.dtype:
                raise ValueError("RHS dtype must match the factors exactly")
            if data.ndim not in (1, 2) or data.shape[0] != self.local_rows:
                raise ValueError("RHS must contain the owned local rows")
            nrhs = 1 if data.ndim == 1 else data.shape[1]
            if nrhs < 1 or nrhs > _INT_MAX or not np.isfinite(data).all():
                raise ValueError("Invalid RHS count or nonfinite RHS value")
            work = np.empty((max(self.local_rows, 1), nrhs), dtype=self.dtype, order="F")
            work[:self.local_rows] = data.reshape(self.local_rows, nrhs)
            return work, np.full(nrhs, np.nan, dtype=np.float64), data.ndim

        work, berr, ndim = _checked(self._comm, prepare)
        columns = self._comm.allgather(work.shape[1])
        if any(count != columns[0] for count in columns):
            raise ValueError("All ranks must supply the same number of RHS columns")
        self._native.solve_dense(work, berr)
        self.backward_error = berr
        result = work[:self.local_rows]
        return result[:, 0] if ndim == 1 else result

    def close(self):
        """
        Release native factors and duplicated communicators collectively.
        Repeated close() calls are harmless after a successful collective close.
        """
        if self._closed:
            return
        if MPI.Is_finalized():
            raise RuntimeError("close() must precede MPI finalization")
        if self._native is not None:
            self._native.close()
        self._comm.Free()
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("SuperLU_DIST owner is closed")
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        if not getattr(self, "_closed", True):
            warnings.warn("Unclosed SuperLU_DIST factors: use collective close() before "
                          "MPI finalization; native resources are retained", ResourceWarning)


def factor(matrix, *, global_size=None, first_row=0, comm=None, **options):
    """
    Factor a canonical local SciPy CSR matrix without modifying its arrays.
    global_size defaults to the matrix column count.
    """
    comm = MPI.COMM_SELF if comm is None else comm

    def arrays():
        if getattr(matrix, "format", None) != "csr":
            raise ValueError("Expected a canonical CSR matrix")
        n = matrix.shape[1] if global_size is None else global_size
        if matrix.shape[1] != n:
            raise ValueError("Local CSR columns must match global_size")
        return matrix.indptr, matrix.indices, matrix.data, n

    ptr, col, data, n = _checked(comm, arrays)
    return Factor(ptr, col, data, global_size=n, first_row=first_row, comm=comm, **options)
