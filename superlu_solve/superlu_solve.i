%module(package="petram.ext.superlu") superlu_solve

%{
#include "superlu_solve.hpp"
#include <mpi4py/mpi4py.h>
%}

%include "numpy_helpers.i"
%include "std_string.i"

%init %{
    if (import_mpi4py() < 0)
        return NULL;
%}

%ignore libsuperlu_solve::initialize_raw;
%ignore libsuperlu_solve::factor_raw;
%ignore libsuperlu_solve::solve_raw;
%ignore libsuperlu_solve::libsuperlu_solve(const libsuperlu_solve&);

%include "superlu_solve.hpp"

%extend libsuperlu_solve {
    void initialize(PyObject* communicator, bool complex_values, int ordering,
                    bool equil, bool tiny_pivot, bool refine, bool log_statistics) {
        MPI_Comm* handle = PyMPIComm_Get(communicator);
        if (!handle)
            throw std::invalid_argument("Expected an mpi4py communicator");

        AllowThreads release;
        $self->initialize_raw(handle, complex_values, ordering, equil,
                              tiny_pivot, refine, log_statistics);
    }

    void factor_csr(long long n, long long first, PyObject* indptr,
                    PyObject* indices, PyObject* values) {
        auto* ptr = checked_array(indptr, NPY_INT32, 1);
        auto* col = checked_array(indices, NPY_INT32, 1);
        auto* data = checked_array(
            values, $self->is_complex() ? NPY_COMPLEX128 : NPY_FLOAT64, 1);

        if (PyArray_SIZE(ptr) < 1 || PyArray_SIZE(col) != PyArray_SIZE(data) ||
            !PyArray_IS_C_CONTIGUOUS(ptr) || !PyArray_IS_C_CONTIGUOUS(col) ||
            !PyArray_IS_C_CONTIGUOUS(data))
            throw std::invalid_argument(
                "Expected contiguous CSR arrays with matching sizes");

        AllowThreads release;
        $self->factor_raw(n, first, PyArray_SIZE(ptr) - 1, PyArray_SIZE(col),
                         static_cast<int32_t*>(PyArray_DATA(ptr)),
                         static_cast<int32_t*>(PyArray_DATA(col)),
                         PyArray_DATA(data));
    }

    void solve_dense(PyObject* rhs, PyObject* backward_error) {
        auto* b = checked_array(
            rhs, $self->is_complex() ? NPY_COMPLEX128 : NPY_FLOAT64, 2);
        auto* berr = checked_array(backward_error, NPY_FLOAT64, 1);

        if (!PyArray_IS_F_CONTIGUOUS(b) || !PyArray_ISWRITEABLE(b) ||
            !PyArray_IS_C_CONTIGUOUS(berr) || !PyArray_ISWRITEABLE(berr) ||
            PyArray_DIM(b, 0) < 1 || PyArray_DIM(b, 0) > INT_MAX ||
            PyArray_DIM(b, 1) < 1 || PyArray_DIM(b, 1) > INT_MAX ||
            PyArray_SIZE(berr) != PyArray_DIM(b, 1))
            throw std::invalid_argument(
                "Invalid writable RHS or backward-error array");

        AllowThreads release;
        $self->solve_raw(PyArray_DATA(b),
                         static_cast<int>(PyArray_DIM(b, 0)),
                         static_cast<int>(PyArray_DIM(b, 1)),
                         static_cast<double*>(PyArray_DATA(berr)));
    }
}
