%module(package="petram.ext.superlu_mt") superlumt_solve
%{
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "superlumt_solve.hpp"
#include <climits>
#include <stdexcept>

static PyArrayObject* checked_array(PyObject* obj, int type, int ndim) {
    if (!PyArray_Check(obj))
        throw std::invalid_argument("Expected a NumPy array");
    auto* array = reinterpret_cast<PyArrayObject*>(obj);
    if (PyArray_TYPE(array) != type || PyArray_NDIM(array) != ndim ||
        !PyArray_ISALIGNED(array) || !PyArray_ISNOTSWAPPED(array))
        throw std::invalid_argument("Unexpected NumPy dtype, shape, or alignment");
    return array;
}

class AllowThreads {
    PyThreadState* state;
public:
    AllowThreads() : state(PyEval_SaveThread()) {}
    ~AllowThreads() { PyEval_RestoreThread(state); }
};
%}

%init %{
import_array();
%}

%include <exception.i>
%exception {
    try {
        $action
    } catch (const std::overflow_error& error) {
        SWIG_exception(SWIG_OverflowError, error.what());
    } catch (const std::invalid_argument& error) {
        SWIG_exception(SWIG_ValueError, error.what());
    } catch (const std::bad_alloc&) {
        SWIG_exception(SWIG_MemoryError, "SuperLU_MT allocation failed");
    } catch (const std::exception& error) {
        SWIG_exception(SWIG_RuntimeError, error.what());
    }
}

%ignore libsuperlumt_solve::factor_raw;
%ignore libsuperlumt_solve::solve_raw;
%include "superlumt_solve.hpp"

%extend libsuperlumt_solve {
    void factor_csc(PyObject* indptr, PyObject* indices, PyObject* values,
                    int threads, int ordering) {
        auto* ptr = checked_array(indptr, NPY_INT32, 1);
        auto* rows = checked_array(indices, NPY_INT32, 1);
        if (!PyArray_Check(values))
            throw std::invalid_argument("Expected NumPy matrix values");
        bool complex_values = PyArray_TYPE(reinterpret_cast<PyArrayObject*>(values))
                              == NPY_COMPLEX128;
        auto* data = checked_array(values, complex_values ? NPY_COMPLEX128 : NPY_FLOAT64, 1);
        npy_intp n = PyArray_SIZE(ptr) - 1, nnz = PyArray_SIZE(data);
        if (n <= 0 || n > INT_MAX || nnz > INT_MAX || PyArray_SIZE(rows) != nnz)
            throw std::invalid_argument("Invalid CSC dimensions or int32 capacity");
        if (!PyArray_IS_C_CONTIGUOUS(ptr) || !PyArray_IS_C_CONTIGUOUS(rows) ||
            !PyArray_IS_C_CONTIGUOUS(data))
            throw std::invalid_argument("CSC arrays must be contiguous");
        AllowThreads release;
        $self->factor_raw(static_cast<int>(n), static_cast<int>(nnz),
            static_cast<int*>(PyArray_DATA(ptr)), static_cast<int*>(PyArray_DATA(rows)),
            static_cast<double*>(PyArray_DATA(data)), complex_values, threads, ordering);
    }

    void solve_dense(PyObject* rhs) {
        bool complex_values = $self->is_complex();
        auto* data = checked_array(rhs, complex_values ? NPY_COMPLEX128 : NPY_FLOAT64, 2);
        if (!PyArray_IS_F_CONTIGUOUS(data) || !PyArray_ISWRITEABLE(data) ||
            PyArray_DIM(data, 0) != $self->size() || PyArray_DIM(data, 1) <= 0 ||
            PyArray_DIM(data, 1) > INT_MAX)
            throw std::invalid_argument("RHS must be writable column-major data matching factors");
        AllowThreads release;
        $self->solve_raw(static_cast<int>(PyArray_DIM(data, 1)),
                        static_cast<double*>(PyArray_DATA(data)), complex_values);
    }
}
