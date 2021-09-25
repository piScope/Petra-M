import inspect

###
###
###  scalar function f(x, y, z)
###  vector/matrixr function f(x, y, z, out)
###
###

def make_signature(f, td=False, return_complex=False, shape=None):
    sig = inspect.signature(f)
    pp = sig.parameters

    from numba import types

    if shape is None:
        if return_complex:
            m = types.complex128
        else:
            m = types.float64
        l = len(pp)
    else:
        m = types.void
        l = len(pp)-1
    if td:
        l = l=1

    args = [types.double]*l
    if td:
        args.append(types.double)
    if shape is not None:
        if return_complex:
            args.append(types.CPointer(types.complex128))
        else:
            args.append(types.CPointer(types.float64))

    return l, m(*args)


def create_caller(l, f, td, scalar=False, vector=False, matrix=False,
                  real=False, imag=False):
    from petram.mfem_config import use_parallel
    if use_parallel:
        import mfem.par as mfem
    else:
        import mfem.ser as mfem

    if scalar:
        if td:
            sig = mfem.scalar_sig_t
        else:
            sig = mfem.scalar_sig        
    if vector:
        if td:
            sig = mfem.vector_sig_t
        else:
            sig = mfem.vector_sig        
    if matrix:
        if td:
            sig = mfem.matrix_sig_t
        else:
            sig = mfem.matrix_sig

    if not real and not imag:
        if scalar:
            if l == 1:
                if td:
                    def s_func(ptx, t, sdim):
                        return f(ptx[0], t)
                else:
                    def s_func(ptx, sdim):
                        return f(ptx[0])
            elif l == 2:
                if td:
                    def s_func(ptx, t,  sdim):
                        return f(ptx[0], ptx[1], t)
                else:
                    def s_func(ptx, sdim):
                        return f(ptx[0], ptx[1])
            elif l == 3:
                if td:
                    def s_func(ptx, t, sdim):
                        return f(ptx[0], ptx[1], ptx[2], t)
                else:
                    def s_func(ptx, sdim):
                        return f(ptx[0], ptx[1], ptx[2])
        else:
            if l == 1:
                if td:
                    def s_func(ptx, t, out, ndim, sdim):
                        return f(ptx[0], t, out)
                else:
                    def s_func(ptx, out, ndim, sdim):
                        return f(ptx[0], out)
            elif l == 2:
                if td:
                    def s_func(ptx, t, ndim, sdim):
                        return f(ptx[0], ptx[1], t, out)
                else:
                    def s_func(ptx, out, ndim, sdim):
                        return f(ptx[0], ptx[1], out)
            elif l == 3:
                if td:
                    def s_func(ptx, t, out, ndim, sdim):
                        return f(ptx[0], ptx[1], ptx[2], t, out)
                else:
                    def s_func(ptx, sdim):
                        return f(ptx[0], ptx[1], ptx[2], out)
    elif real:
        if scalar:
            if l == 1:
                if td:
                    def s_func(ptx, t, sdim):
                        return f(ptx[0], t).real
                else:
                    def s_func(ptx, sdim):
                        return f(ptx[0]).real
            elif l == 2:
                if td:
                    def s_func(ptx, t,  sdim):
                        return f(ptx[0], ptx[1], t).real
                else:
                    def s_func(ptx, sdim):
                        return f(ptx[0], ptx[1]).real
            elif l == 3:
                if td:
                    def s_func(ptx, t, sdim):
                        return f(ptx[0], ptx[1], ptx[2], t).real
                else:
                    def s_func(ptx, sdim):
                        return f(ptx[0], ptx[1], ptx[2]).real
        else:
            if l == 1:
                if td:
                    def s_func(ptx, t, out, ndim, sdim):
                        return f(ptx[0], t, out).real
                else:
                    def s_func(ptx, out, ndim, sdim):
                        return f(ptx[0], out).real
            elif l == 2:
                if td:
                    def s_func(ptx, t, out, ndim, sdim):
                        return f(ptx[0], ptx[1], t, out).real
                else:
                    def s_func(ptx, out, ndim, sdim):
                        return f(ptx[0], ptx[1], out).real
            elif l == 3:
                if td:
                    def s_func(ptx, t, out, ndim, sdim):
                        return f(ptx[0], ptx[1], ptx[2], t, out).real
                else:
                    def s_func(ptx, sdim):
                        return f(ptx[0], ptx[1], ptx[2], out).real
    elif imag:
        if scalar:
            if l == 1:
                if td:
                    def s_func(ptx, t, sdim):
                        return f(ptx[0], t).imag
                else:
                    def s_func(ptx, sdim):
                        return f(ptx[0]).imag
            elif l == 2:
                if td:
                    def s_func(ptx, t,  sdim):
                        return f(ptx[0], ptx[1], t).imag
                else:
                    def s_func(ptx, sdim):
                        return f(ptx[0], ptx[1]).imag
            elif l == 3:
                if td:
                    def s_func(ptx, t, sdim):
                        return f(ptx[0], ptx[1], ptx[2], t).imag
                else:
                    def s_func(ptx, sdim):
                        return f(ptx[0], ptx[1], ptx[2]).imag
        else:
            if l == 1:
                if td:
                    def s_func(ptx, t, out, ndim, sdim):
                        return f(ptx[0], t, out).imag
                else:
                    def s_func(ptx, out, ndim, sdim):
                        return f(ptx[0], out).imag
            elif l == 2:
                if td:
                    def s_func(ptx, t, ndim, sdim):
                        return f(ptx[0], ptx[1], t, out).imag
                else:
                    def s_func(ptx, out, ndim, sdim):
                        return f(ptx[0], ptx[1], out).imag
            elif l == 3:
                if td:
                    def s_func(ptx, t, out, ndim, sdim):
                        return f(ptx[0], ptx[1], ptx[2], t, out).imag
                else:
                    def s_func(ptx, sdim):
                        return f(ptx[0], ptx[1], ptx[2], out).imag
                    
        
    from numba import cfunc
    return cfunc(sig)(s_func)
