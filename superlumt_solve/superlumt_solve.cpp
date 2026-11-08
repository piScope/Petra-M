#include "superlumt_solve.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include "slu_mt_ddefs.h"
#include "slu_dcomplex.h"
#include "metis.h"
}
#include "complex_interface.hpp"

namespace {
// SuperLU_MT has process-global workspace state. Serialize independent calls;
// its own factorization workers still run in parallel.
std::mutex native_mutex;
using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void check_status(int info, const char* operation) {
    if (info != 0)
        throw std::runtime_error(std::string("SuperLU_MT ") + operation +
                                 " failed with info=" + std::to_string(info));
}

void metis_ordering(int n, const int* ptr, const int* rows, int* perm_c) {
    static_assert(sizeof(idx_t) == sizeof(int), "METIS must use int32 indices");
    std::vector<std::vector<idx_t>> neighbors(n);
    for (int column = 0; column < n; ++column) {
        for (int k = ptr[column]; k < ptr[column + 1]; ++k) {
            int row = rows[k];
            if (row != column) {
                neighbors[column].push_back(row);
                neighbors[row].push_back(column);
            }
        }
    }
    std::vector<idx_t> xadj(n + 1), adjncy;
    for (int i = 0; i < n; ++i) {
        auto& entries = neighbors[i];
        std::sort(entries.begin(), entries.end());
        entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
        if (entries.size() > static_cast<size_t>(INT_MAX) - adjncy.size())
            throw std::overflow_error("METIS adjacency exceeds int32 capacity");
        adjncy.insert(adjncy.end(), entries.begin(), entries.end());
        xadj[i + 1] = static_cast<idx_t>(adjncy.size());
    }
    // METIS expects an address even for a graph with no edges.
    idx_t dummy = 0;
    idx_t count = n;
    std::vector<idx_t> perm(n), inverse(n);
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_SEED] = 0;
    int status = METIS_NodeND(&count, xadj.data(),
                              adjncy.empty() ? &dummy : adjncy.data(), nullptr,
                              options, perm.data(), inverse.data());
    if (status != METIS_OK)
        throw std::runtime_error("METIS_NodeND failed with status=" + std::to_string(status));
    // SuperLU maps original column -> permuted position: METIS's inverse.
    std::copy(inverse.begin(), inverse.end(), perm_c);
}
}

struct libsuperlumt_solve::Impl {
    SuperMatrix A{}, L{}, U{};
    int n = 0, thread_count = 0;
    bool complex_values = false, ready = false;
    double order_time = 0, factor_time = 0;
    std::vector<int> ptr, rows, perm_c, perm_r;
    std::vector<double> real_data;
    std::vector<doublecomplex> complex_data;

    void release() {
        if (L.Store) Destroy_SuperNode_SCP(&L);
        if (U.Store) Destroy_CompCol_NCP(&U);
        // Matrix values and indices belong to vectors, not to SuperLU.
        if (A.Store) Destroy_SuperMatrix_Store(&A);
        A = {}; L = {}; U = {};
        ready = false;
        n = 0;
        std::vector<int>().swap(ptr);
        std::vector<int>().swap(rows);
        std::vector<int>().swap(perm_c);
        std::vector<int>().swap(perm_r);
        std::vector<double>().swap(real_data);
        std::vector<doublecomplex>().swap(complex_data);
    }
    ~Impl() { release(); }
};

libsuperlumt_solve::libsuperlumt_solve() : impl(new Impl) {}
libsuperlumt_solve::~libsuperlumt_solve() {
    std::lock_guard<std::mutex> lock(native_mutex);
    delete impl;
}

void libsuperlumt_solve::close() {
    std::lock_guard<std::mutex> lock(native_mutex);
    impl->release();
}

void libsuperlumt_solve::factor_raw(int n, int nnz, const int* ptr,
        const int* rows, const double* values, bool complex_values,
        int threads, int ordering) {
    std::lock_guard<std::mutex> lock(native_mutex);
    if (n <= 0 || nnz < 0 || threads <= 0 || ordering < 0 || ordering > 4)
        throw std::invalid_argument("Invalid dimensions, thread count, or column ordering");
    if (ptr[0] != 0 || ptr[n] != nnz)
        throw std::invalid_argument("Invalid CSC column pointers");
    for (int i = 0; i < n; ++i)
        if (ptr[i] < 0 || ptr[i] > ptr[i + 1] || ptr[i + 1] > nnz)
            throw std::invalid_argument("CSC column pointers must be nondecreasing");
    for (int k = 0; k < nnz; ++k)
        if (rows[k] < 0 || rows[k] >= n)
            throw std::invalid_argument("CSC row index is out of range");
    for (size_t k = 0; k < static_cast<size_t>(nnz) * (complex_values ? 2 : 1); ++k)
        if (!std::isfinite(values[k]))
            throw std::invalid_argument("Matrix values must be finite");
    if (nnz == 0)
        throw std::runtime_error("SuperLU_MT factorization: matrix is exactly singular");
    impl->release();
    impl->n = n;
    impl->thread_count = threads;
    impl->complex_values = complex_values;
    impl->ptr.assign(ptr, ptr + n + 1);
    impl->rows.assign(rows, rows + nnz);
    impl->perm_c.resize(n);
    impl->perm_r.resize(n);
    if (complex_values) {
        impl->complex_data.resize(nnz);
        for (int k = 0; k < nnz; ++k)
            impl->complex_data[k] = {values[2 * k], values[2 * k + 1]};
        superlumt_create_complex(&impl->A, n, nnz, impl->complex_data.data(),
                                 impl->rows.data(), impl->ptr.data());
    } else {
        impl->real_data.assign(values, values + nnz);
        dCreate_CompCol_Matrix(&impl->A, n, n, nnz, impl->real_data.data(),
            impl->rows.data(), impl->ptr.data(), SLU_NC, SLU_D, SLU_GE);
    }
    auto start = Clock::now();
    try {
        if (ordering == 4)
            metis_ordering(n, ptr, rows, impl->perm_c.data());
        else
            get_perm_c(ordering, &impl->A, impl->perm_c.data());
        impl->order_time = elapsed(start);
        SuperMatrix AC{};
        superlumt_options_t options{};
        Gstat_t stats{};
        const int panel = sp_ienv(1), relax = sp_ienv(2);
        StatAlloc(n, threads, panel, relax, &stats);
        StatInit(n, threads, &stats);
        start = Clock::now();
        int info = 0;
        if (complex_values) {
            superlumt_factor_complex(threads, panel, relax, impl->perm_c.data(),
                impl->perm_r.data(), &impl->A, &AC, &options, &stats,
                &impl->L, &impl->U, &info);
        } else {
            pdgstrf_init(threads, DOFACT, NOTRANS, NO, panel, relax, 1.0, NO, 0.0,
                impl->perm_c.data(), impl->perm_r.data(), nullptr, 0,
                &impl->A, &AC, &options, &stats);
            pdgstrf(&options, &AC, impl->perm_r.data(), &impl->L, &impl->U, &stats, &info);
        }
        impl->factor_time = elapsed(start);
        pxgstrf_finalize(&options, &AC);
        StatFree(&stats);
        check_status(info, "factorization");
        impl->ready = true;
    } catch (...) {
        impl->release();
        throw;
    }
}

void libsuperlumt_solve::solve_raw(int nrhs, double* values, bool complex_values) {
    std::lock_guard<std::mutex> lock(native_mutex);
    if (!impl->ready)
        throw std::runtime_error("SuperLU_MT factors are not available");
    if (nrhs <= 0 || complex_values != impl->complex_values)
        throw std::invalid_argument("RHS count or dtype does not match factors");
    SuperMatrix B{};
    std::vector<doublecomplex> complex_rhs;
    size_t count = static_cast<size_t>(impl->n) * nrhs;
    if (complex_values) {
        complex_rhs.resize(count);
        for (size_t k = 0; k < count; ++k)
            complex_rhs[k] = {values[2 * k], values[2 * k + 1]};
        superlumt_dense_complex(&B, impl->n, nrhs, complex_rhs.data());
    } else {
        dCreate_Dense_Matrix(&B, impl->n, nrhs, values, impl->n, SLU_DN, SLU_D, SLU_GE);
    }
    Gstat_t stats{};
    StatAlloc(impl->n, 1, sp_ienv(1), sp_ienv(2), &stats);
    StatInit(impl->n, 1, &stats);
    int info = 0;
    if (complex_values)
        superlumt_solve_complex(&impl->L, &impl->U, impl->perm_r.data(),
                                impl->perm_c.data(), &B, &stats, &info);
    else
        dgstrs(NOTRANS, &impl->L, &impl->U, impl->perm_r.data(), impl->perm_c.data(),
                &B, &stats, &info);
    StatFree(&stats);
    Destroy_SuperMatrix_Store(&B);
    check_status(info, "solve");
    if (complex_values)
        for (size_t k = 0; k < count; ++k) {
            values[2 * k] = complex_rhs[k].r;
            values[2 * k + 1] = complex_rhs[k].i;
        }
}

int libsuperlumt_solve::size() const { return impl->ready ? impl->n : 0; }
bool libsuperlumt_solve::is_complex() const { return impl->complex_values; }
int libsuperlumt_solve::threads() const { return impl->thread_count; }
long long libsuperlumt_solve::factor_nnz() const {
    if (!impl->ready) return 0;
    return static_cast<long long>(static_cast<SCPformat*>(impl->L.Store)->nnz) +
           static_cast<NCPformat*>(impl->U.Store)->nnz;
}
double libsuperlumt_solve::ordering_seconds() const { return impl->order_time; }
double libsuperlumt_solve::factor_seconds() const { return impl->factor_time; }
