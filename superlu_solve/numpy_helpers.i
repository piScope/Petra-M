%{
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
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
        SWIG_exception(SWIG_MemoryError, "SuperLU_DIST allocation failed");
    } catch (const std::exception& error) {
        SWIG_exception(SWIG_RuntimeError, error.what());
    }
}

