extern "C" {
#include "slu_mt_zdefs.h"
}
#include "complex_interface.hpp"

void superlumt_create_complex(SuperMatrix* A, int n, int nnz, doublecomplex* values,
                              int* rows, int* ptr) {
    zCreate_CompCol_Matrix(A, n, n, nnz, values, rows, ptr, SLU_NC, SLU_Z, SLU_GE);
}

void superlumt_dense_complex(SuperMatrix* B, int n, int nrhs, doublecomplex* values) {
    zCreate_Dense_Matrix(B, n, nrhs, values, n, SLU_DN, SLU_Z, SLU_GE);
}

void superlumt_factor_complex(int threads, int panel, int relax, int* pc, int* pr,
        SuperMatrix* A, SuperMatrix* AC, superlumt_options_t* options, Gstat_t* stats,
        SuperMatrix* L, SuperMatrix* U, int* info) {
    pzgstrf_init(threads, DOFACT, NOTRANS, NO, panel, relax, 1.0, NO, 0.0,
                 pc, pr, nullptr, 0, A, AC, options, stats);
    pzgstrf(options, AC, pr, L, U, stats, info);
}

void superlumt_solve_complex(SuperMatrix* L, SuperMatrix* U, int* pr, int* pc,
                             SuperMatrix* B, Gstat_t* stats, int* info) {
    zgstrs(NOTRANS, L, U, pr, pc, B, stats, info);
}
