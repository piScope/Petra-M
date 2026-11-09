#pragma once
#include <cstdint>
#include <string>

// No MPI or dependency types cross the facade boundary.
class libsuperlu_solve {
    void* impl = nullptr;
public:
    libsuperlu_solve();
    ~libsuperlu_solve();

    static std::string mpi_library_version();

    libsuperlu_solve(const libsuperlu_solve&) = delete;
    libsuperlu_solve& operator=(const libsuperlu_solve&) = delete;

    void initialize_raw(void* communicator, bool complex_values, int ordering,
                        bool equil, bool tiny_pivot, bool refine, bool log_statistics);
    void factor_raw(int64_t n, int64_t first, int64_t rows, int64_t nnz,
                    const int32_t* ptr, const int32_t* col, const void* values);
    void solve_raw(void* rhs, int ldb, int nrhs, double* berr);

    bool is_complex() const;
    void close();
};
