#ifndef PETRAM_SUPERLUMT_SOLVE_HPP
#define PETRAM_SUPERLUMT_SOLVE_HPP

// The facade owns all SuperLU buffers and factors. No MFEM or MPI API is used.
class libsuperlumt_solve {
public:
    libsuperlumt_solve();
    ~libsuperlumt_solve();
    libsuperlumt_solve(const libsuperlumt_solve&) = delete;
    libsuperlumt_solve& operator=(const libsuperlumt_solve&) = delete;

    void factor_raw(int n, int nnz, const int* indptr, const int* indices,
                    const double* values, bool complex_values,
                    int threads, int ordering);
    void solve_raw(int nrhs, double* values, bool complex_values);
    void close();
    int size() const;
    bool is_complex() const;
    int threads() const;
    long long factor_nnz() const;
    double ordering_seconds() const;
    double factor_seconds() const;

private:
    struct Impl;
    Impl* impl;
};
#endif
