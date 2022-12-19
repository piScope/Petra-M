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

def generate_caller_scalar(setting, sdim):
    '''
    generate a callder function on the fly

    ex)
    if setting is
        {"iscomplex": (True, False), "kinds": (1, 0),
                       "output": True, size: (10, 1)}

    def _caller(ptx, data):
        ptx = farray(ptx, (sdim,), np.float64)      # for position
        arr0r = farray(data[0], (10,), np.float64)
        arr0i = farray(data[1], (10,), np.float64)
        arr0 = arr0r+1j*arr0i
        arr1 = farray(data[0], (1,), np.float64)

        params = (arr0, arr1)
        return (inner_func(ptx[0], ptx[1], *params))

    here inner_func is a function user provided.

    '''
    if setting['td']:
        text = ['def _caller(ptx, t, data):']
    else:
        text = ['def _caller(ptx, data):']

    text.append("    ptx = farray(ptx, (sdim,), np.float64)")
    count = 0
    params_line = '    params = ('

    for s, kind, size in zip(setting['iscomplex'], setting['kinds'], setting["sizes"]):
        if s:
            t1 = '    arrr' + \
                str(count) + ' = farray(data[' + \
                str(count) + "], ("+str(size) + "), np.float64)"
            t2 = '    arri' + \
                str(count) + ' = farray(data[' + \
                str(count+1) + "], ("+str(size) + "), np.float64)"
            t3 = '    arr'+str(count) + ' = arrr' + \
                str(count) + "+1j*arri" + str(count)

            text.extend((t1, t2, t3))
            params_line += 'arr'+str(count)+','
            count = count + 2
        else:
            t = '    arr' + \
                str(count) + ' = farray(data[' + \
                str(count) + "], ("+str(size) + "), np.float64)"
            text.append(t)

            params_line += 'arr'+str(count)+','
            count = count + 1

    params_line += ')'

    text.append(params_line)

    return_txt = "    return (inner_func("
    for i in range(sdim):
        return_txt = return_txt + "ptx["+str(i)+"], "
    if setting["td"]:
        return_txt = return_txt + "t, "
    return_txt = return_txt + "*params))"
    
    text.append(return_txt)
    
    return '\n'.join(text)


def generate_caller_array(setting):
    '''
    generate a callder function on the fly

    ex)
    if setting is
        {"iscomplex": (True, False), "kinds": (1, 0),
                       "output": True, size: ((3, 3), 1), outsize: (2, 2) }

    def _caller(ptx, data, out_):
        ptx = farray(ptx, (sdim,), np.float64)      # for position
        arr0r = farray(data[0], (3, 3), np.float64)
        arr0i = farray(data[1], (3, 3), np.float64)
        arr0 = arr0r+1j*arr0i

        arr1 = farray(data[0], (1,), np.float64)

        out = farray(out_, (2, 2), np.complex128)

        params = (arr0, arr1, )

        ret = inner_func(ptx[0], ptx[1],  *params)
        for i0 in range(2):
           for i1 in range(2):
              ret[i0,i1] = out[i0, i1]

    here inner_func is a function user provided.

    '''
    if setting['td']:
        text = ['def _caller(ptx, t, data, out_):']
    else:
        text = ['def _caller(ptx, data, out_):']
    text.append("    ptx = farray(ptx, (sdim,), np.float64)")
    count = 0
    params_line = '    params = ('

    for s, kind, size in zip(setting['iscomplex'], setting['kinds'], setting["sizes"]):
        if s:
            if not isinstance(size, tuple):
                size = (size, )
            t1 = '    arrr' + \
                str(count) + ' = farray(data[' + \
                str(count) + "], "+str(size) + ", np.float64)"
            t2 = '    arri' + \
                str(count) + ' = farray(data[' + \
                str(count+1) + "], "+str(size) + ", np.float64)"
            t3 = '    arr'+str(count) + ' = arrr' + \
                str(count) + "+1j*arri" + str(count)

            text.extend((t1, t2, t3))
            params_line += 'arr'+str(count)+','
            count = count + 2
        else:
            t = '    arr' + \
                str(count) + ' = farray(data[' + \
                str(count) + "], ("+str(size) + "), np.float64)"
            text.append(t)

            params_line += 'arr'+str(count)+','
            count = count + 1

    outsize = setting["outsize"]
    if setting["output"]:
        t = '    out = farray(out_,' + str(outsize) + ", np.complex128)"
    else:
        t = '    out = farray(out_,' + str(outsize) + ", np.float64)"
    text.append(t)
    '''
    params_line += 'out, )'
    '''
    params_line += ')'
    text.append(params_line)

    return_txt = "    ret = (inner_func("
    for i in range(sdim):
        return_txt = return_txt + "ptx["+str(i)+"], "
    if setting["td"]:
        return_txt = return_txt + "t, "
    return_txt = return_txt + "*params))"

    text.append(return_txt)

    idx_text = ""
    for k, s in enumerate(setting["outsize"]):
        text.append("    " + " "*k + "for i" + str(k) +
                    " in range(" + str(s) + "):")
        idx_text = idx_text + "i"+str(k)+","
    text.append("     " + " "*len(setting["outsize"]) +
                "out["+idx_text + "]=ret[" + idx_text + "]")

    return '\n'.join(text)


def generate_signature_scalar(setting, sdim):
    '''
    generate a signature to numba-compile a user scalar function

    ex)
    when user function is
        func(ptx, complex_array, float_scalar)

    setting is
        {"iscomplex": (2, 1), "kinds": (1, 0), "output": 2}

    output is
         types.complex128(types.double, types.double, types.complex128[:], types.double,)

    user function is

    '''
    sig = ''
    if setting['output']:
        sig += 'types.complex128('
    else:
        sig += 'types.float64('

    for i in range(sdim):
        sig += 'types.double, '

    if setting['td']:
        sig += 'types.double, '

    for s, kind, in zip(setting['iscomplex'], setting['kinds'],):
        if s:
            if kind == 0:
                sig += 'types.complex128,'
            elif kind == 1:
                sig += 'types.complex128[:], '
            else:
                sig += 'types.complex128[:, :], '
        else:
            if kind == 0:
                sig += 'types.double,'
            elif kind == 1:
                sig += 'types.double[:], '
            else:
                sig += 'types.double[:, :], '

    sig = sig + ")"
    return sig

def generate_signature_array(setting, sdim):
    '''
    generate a signature to numba-compile a user scalar function

    ex)
    when user function is
        func(ptx, complex_array, float_scalar)

    setting is
        {"iscomplex": (2, 1), "kinds": (1, 0), "output": 2}

    output is
         types.complex128[:, :](types.double[:], types.complex128[:], types.double,)

    user function is

    '''
    sig = ''
    if setting['output']:
        if setting['outkind'] == 1:
            sig += 'types.complex128[:]('
        else:
            sig += 'types.complex128[:,:]('
    else:
        if setting['outkind'] == 1:
            sig += 'types.float64[:]('
        else:
            sig += 'types.float64[:,:]'

    for i in range(sdim):
        sig += 'types.double, '
            
    if setting['td']:
        sig += 'types.double, '

    for s, kind, in zip(setting['iscomplex'], setting['kinds'],):
        if s:
            if kind == 0:
                sig += 'types.complex128,'
            elif kind == 1:
                sig += 'types.complex128[:], '
            else:
                sig += 'types.complex128[:, :], '
        else:
            if kind == 0:
                sig += 'types.double,'
            elif kind == 1:
                sig += 'types.double[:], '
            else:
                sig += 'types.double[:, :], '

    sig = sig + ")"
    return sig


