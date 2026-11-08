// Private scalar adapter: upstream's d/z headers define conflicting internal
// types, so include them in separate translation units. Types below are shared
// public SuperLU_MT types supplied by the including translation unit.
void superlumt_create_complex(SuperMatrix*, int, int, doublecomplex*, int*, int*);
void superlumt_dense_complex(SuperMatrix*, int, int, doublecomplex*);
void superlumt_factor_complex(int, int, int, int*, int*, SuperMatrix*, SuperMatrix*,
                              superlumt_options_t*, Gstat_t*, SuperMatrix*,
                              SuperMatrix*, int*);
void superlumt_solve_complex(SuperMatrix*, SuperMatrix*, int*, int*, SuperMatrix*,
                             Gstat_t*, int*);
