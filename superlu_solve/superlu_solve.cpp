#include "superlu_solve.hpp"

#include <algorithm>
#include <climits>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

extern "C" {
#include "superlu_ddefs.h"
#include "superlu_zdefs.h"
#include "metis.h"
}

#ifndef HAVE_METIS
#error "SuperLU_DIST must be built with the standalone-METIS patch"
#endif

static_assert(sizeof(int_t) == sizeof(int32_t), "32-bit sparse indices required");
static_assert(sizeof(idx_t) == sizeof(int_t), "METIS and DIST index widths must match");
static_assert(sizeof(doublecomplex) == 2 * sizeof(double), "Complex layout mismatch");

namespace {
    // Coordinate local validation/allocation errors before the next native collective.
    void checked(MPI_Comm comm, const std::function<void()>& operation) {
        std::string message;
        try {
            operation();
        } catch (const std::exception& error) {
            message = error.what();
        }

        int rank, failed, first;
        MPI_Comm_rank(comm, &rank);
        failed = message.empty() ? INT_MAX : rank;
        MPI_Allreduce(&failed, &first, 1, MPI_INT, MPI_MIN, comm);

        if (first != INT_MAX) {
            int length = rank == first ? static_cast<int>(message.size()) : 0;
            MPI_Bcast(&length, 1, MPI_INT, first, comm);

            message.resize(length);
            MPI_Bcast(message.data(), length, MPI_CHAR, first, comm);

            throw std::runtime_error("SuperLU_DIST rank " + std::to_string(first) +
                                     ": " + message);
        }
    }

    struct State {
        MPI_Comm comm = MPI_COMM_NULL;
        gridinfo_t grid{};
        superlu_dist_options_t options{};
        SuperMatrix matrix{};
        SuperLUStat_t stats{};

        dScalePermstruct_t real_scale{};
        dLUstruct_t real_lu{};
        dSOLVEstruct_t real_solve{};

        zScalePermstruct_t complex_scale{};
        zLUstruct_t complex_lu{};
        zSOLVEstruct_t complex_solve{};

        std::vector<int_t> ptr, col;
        std::vector<double> real_data;
        std::vector<doublecomplex> complex_data;

        int_t n = 0, rows = 0, first = 0;
        int last_nrhs = 0;

        bool complex_values = false;
        bool grid_ready = false, active = false;
        bool structures = false, matrix_ready = false, numeric_started = false;
        bool factored = false;

        void prepare_csr(int64_t dimension, int64_t offset, int64_t local_rows,
                         int64_t nnz, const int32_t* rowptr, const int32_t* columns,
                         const void* data) {
            if (structures)
                throw std::runtime_error(
                    "Create a new owner to factor another matrix");

            if (dimension <= 0 || dimension > INT_MAX || offset < 0 ||
                local_rows < 0 || local_rows > INT_MAX ||
                offset > dimension - local_rows || nnz < 0 || nnz > INT_MAX)
                throw std::invalid_argument(
                    "Invalid dimensions or native int32 capacity");

            if (rowptr[0] != 0 || rowptr[local_rows] != nnz)
                throw std::invalid_argument("Invalid CSR pointers");
            for (int64_t i = 0; i < local_rows; ++i) {
                if (rowptr[i] < 0 || rowptr[i] > rowptr[i + 1] ||
                    rowptr[i + 1] > nnz)
                    throw std::invalid_argument("Invalid CSR row bounds");
                for (int64_t k = rowptr[i]; k < rowptr[i + 1]; ++k)
                    if (columns[k] < 0 || columns[k] >= dimension ||
                        (k > rowptr[i] && columns[k] <= columns[k - 1]))
                        throw std::invalid_argument(
                            "CSR columns must be canonical and in range");
            }

            n = static_cast<int_t>(dimension);
            rows = static_cast<int_t>(local_rows);
            first = static_cast<int_t>(offset);

            ptr.assign(rowptr, rowptr + local_rows + 1);
            col.assign(columns, columns + nnz);
            col.resize(std::max<int64_t>(nnz, 1));

            if (complex_values) {
                complex_data.resize(std::max<int64_t>(nnz, 1));
                std::memcpy(complex_data.data(), data,
                            static_cast<size_t>(nnz) * sizeof(doublecomplex));
            } else {
                real_data.resize(std::max<int64_t>(nnz, 1));
                std::memcpy(real_data.data(), data,
                            static_cast<size_t>(nnz) * sizeof(double));
            }
        }

        void initialize_matrix(int_t nnz) {
            if (complex_values) {
                zCreate_CompRowLoc_Matrix_dist(
                    &matrix, n, n, nnz, rows, first, complex_data.data(),
                    col.data(), ptr.data(), SLU_NR_loc, SLU_Z, SLU_GE);
                zScalePermstructInit(n, n, &complex_scale);
                zLUstructInit(n, &complex_lu);
            } else {
                dCreate_CompRowLoc_Matrix_dist(
                    &matrix, n, n, nnz, rows, first, real_data.data(),
                    col.data(), ptr.data(), SLU_NR_loc, SLU_D, SLU_GE);
                dScalePermstructInit(n, n, &real_scale);
                dLUstructInit(n, &real_lu);
            }

            matrix_ready = true;
            PStatInit(&stats);
            structures = true;
        }

        int call_solver(void* rhs, int ldb, int nrhs, double* berr) {
            int info = 0;
            if (complex_values) {
                pzgssvx(&options, &matrix, &complex_scale,
                        static_cast<doublecomplex*>(rhs), ldb, nrhs, &grid,
                        &complex_lu, &complex_solve, berr, &stats, &info);
            } else {
                pdgssvx(&options, &matrix, &real_scale,
                        static_cast<double*>(rhs), ldb, nrhs, &grid,
                        &real_lu, &real_solve, berr, &stats, &info);
            }
            PStatPrint(&options, &stats, &grid);
            return info;
        }

        void factor(int64_t dimension, int64_t offset, int64_t local_rows, int64_t nnz,
                    const int32_t* rowptr, const int32_t* columns, const void* data) {
            checked(comm, [&] {
                prepare_csr(dimension, offset, local_rows, nnz, rowptr, columns, data);
            });
            initialize_matrix(static_cast<int_t>(nnz));

            // nrhs=0 performs factorization only.
            double real_dummy = 0;
            doublecomplex complex_dummy{};
            void* rhs = complex_values ? static_cast<void*>(&complex_dummy)
                                       : static_cast<void*>(&real_dummy);
            double berr = 0;
            numeric_started = true;
            int info = call_solver(rhs, std::max<int_t>(rows, 1), 0, &berr);

            checked(comm, [&] {
                if (info)
                    throw std::runtime_error("Factorization returned info=" +
                                             std::to_string(info));
            });

            factored = true;
            options.Fact = FACTORED;
        }

        void validate_rhs(int ldb, int nrhs) const {
            if (!factored)
                throw std::runtime_error("No valid factors");
            if (ldb < std::max<int_t>(rows, 1) || nrhs < 1)
                throw std::invalid_argument(
                    "Invalid RHS leading dimension or column count");
        }

        void resize_rhs_communication(int nrhs) {
            // Follow upstream pddrive1: keep the matrix/refinement mapping and
            // replace only RHS communication counters when nrhs changes.
            auto* communication = complex_values ? complex_solve.gstrs_comm
                                                  : real_solve.gstrs_comm;
            SUPERLU_FREE(communication->B_to_X_SendCnt);
            SUPERLU_FREE(communication->X_to_B_SendCnt);
            SUPERLU_FREE(communication->ptr_to_ibuf);

            if (complex_values) {
                pzgstrs_init(n, rows, nrhs, first, complex_scale.perm_r,
                             complex_scale.perm_c, &grid, complex_lu.Glu_persist,
                             &complex_solve);
            } else {
                pdgstrs_init(n, rows, nrhs, first, real_scale.perm_r,
                             real_scale.perm_c, &grid, real_lu.Glu_persist,
                             &real_solve);
            }
        }

        void solve(void* rhs, int ldb, int nrhs, double* berr) {
            checked(comm, [&] { validate_rhs(ldb, nrhs); });

            if (options.SolveInitialized && last_nrhs != nrhs)
                resize_rhs_communication(nrhs);

            int info = call_solver(rhs, ldb, nrhs, berr);
            last_nrhs = nrhs;

            checked(comm, [&] {
                if (info)
                    throw std::runtime_error("Solve returned info=" +
                                             std::to_string(info));
            });
        }

        void release() {
            if (structures) {
                if (complex_values) {
                    if (numeric_started) {
                        if (options.SolveInitialized)
                            zSolveFinalize(&options, &complex_solve);
                        zDestroy_LU(n, &grid, &complex_lu);
                    }
                    zLUstructFree(&complex_lu);
                    zScalePermstructFree(&complex_scale);
                } else {
                    if (numeric_started) {
                        if (options.SolveInitialized)
                            dSolveFinalize(&options, &real_solve);
                        dDestroy_LU(n, &grid, &real_lu);
                    }
                    dLUstructFree(&real_lu);
                    dScalePermstructFree(&real_scale);
                }

                PStatFree(&stats);
                structures = false;
            }

            if (matrix_ready) {
                Destroy_SuperMatrix_Store_dist(&matrix);
                matrix_ready = false;
            }

            ptr.clear();
            col.clear();
            real_data.clear();
            complex_data.clear();
        }

        void close() {
            if (!active)
                return;

            release();
            if (grid_ready)
                superlu_gridexit(&grid);

            MPI_Comm_free(&comm);
            active = false;
        }
    };

    State* state_from(void* impl) {
        auto* state = static_cast<State*>(impl);
        if (!state || !state->active)
            throw std::runtime_error("SuperLU_DIST owner is closed");

        return state;
    }
} // namespace

libsuperlu_solve::libsuperlu_solve() = default;

std::string libsuperlu_solve::mpi_library_version() {
    char version[MPI_MAX_LIBRARY_VERSION_STRING];
    int length;
    MPI_Get_library_version(version, &length);
    return std::string(version, length);
}

libsuperlu_solve::~libsuperlu_solve() {
    // Never start MPI collectives from Python garbage collection. An unclosed
    // owner is deliberately abandoned; explicit close() releases its resources.
    auto* state = static_cast<State*>(impl);
    if (state && !state->active)
        delete state;
}

void libsuperlu_solve::initialize_raw(void* communicator, bool complex_values,
                                      int ordering, bool equil,
                                      bool tiny_pivot, bool refine, bool log_statistics) {
    if (impl)
        throw std::runtime_error("Owner already initialized");

    MPI_Comm input = *static_cast<MPI_Comm*>(communicator);
    if (input == MPI_COMM_NULL)
        throw std::runtime_error("An active MPI communicator is required");

    std::unique_ptr<State> allocated;
    checked(input, [&] { allocated.reset(new State); });

    auto* state = allocated.release();
    impl = state;
    state->complex_values = complex_values;

    MPI_Comm_dup(input, &state->comm);
    state->active = true;

    int size;
    MPI_Comm_size(state->comm, &size);
    int nprow = 1;
    for (int p = 1; p <= size / p; ++p)
        if (size % p == 0)
            nprow = p;

    superlu_gridinit(state->comm, nprow, size / nprow, &state->grid);
    state->grid_ready = true;

    set_default_options_dist(&state->options);
    state->options.ColPerm = static_cast<colperm_t>(ordering);
    state->options.RowPerm = NOROWPERM;
    state->options.Equil = equil ? YES : NO;
    state->options.ReplaceTinyPivot = tiny_pivot ? YES : NO;
    state->options.IterRefine = refine ? SLU_DOUBLE : NOREFINE;
    state->options.ParSymbFact = NO;
    state->options.PrintStat = log_statistics ? YES : NO;
    state->options.GPURES = NO;
}

void libsuperlu_solve::factor_raw(int64_t n, int64_t first, int64_t rows, int64_t nnz,
                                  const int32_t* ptr, const int32_t* col,
                                  const void* values) {
    state_from(impl)->factor(n, first, rows, nnz, ptr, col, values);
}

void libsuperlu_solve::solve_raw(void* rhs, int ldb, int nrhs, double* berr) {
    state_from(impl)->solve(rhs, ldb, nrhs, berr);
}

void libsuperlu_solve::close() {
    auto* state = static_cast<State*>(impl);
    if (state && state->active)
        state->close();

    delete state;
    impl = nullptr;
}

bool libsuperlu_solve::is_complex() const {
    return state_from(impl)->complex_values;
}
