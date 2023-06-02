import inspect


def generate_caller_scalar(setting, sdim):
    '''
    generate a callder function on the fly

    ex)
    if setting is
        {"isdepcomplex": (True, False), "kinds": (1, 0),
                       "output": True, size: (10, 1)}

    def _caller(ptx, sdim, data):
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
        text = ['def _caller(ptx, sdim,  t, data):']
    else:
        text = ['def _caller(ptx, sdim, data):']

    text.append("    ptx = farray(ptx, (sdim,), np.float64)")
    count = 0
    params_line = '    params = ('

    for s, kind, size in zip(setting['isdepcomplex'], setting['kinds'], setting["sizes"]):
        if not isinstance(size, tuple):
            size = (size, )

        if s:
            t1 = '    arrr' + \
                str(count) + ' = farray(data[' + \
                str(count) + "], "+str(size) + ", np.float64)"
            t2 = '    arri' + \
                str(count) + ' = farray(data[' + \
                str(count+1) + "], "+str(size) + ", np.float64)"

            if len(size) == 1 and size[0] == 1:
                t1 += '[0]'
                t2 += '[0]'

            t3 = '    arr'+str(count) + ' = arrr' + \
                str(count) + "+1j*arri" + str(count)

            text.extend((t1, t2, t3))
            params_line += 'arr'+str(count)+','
            count = count + 2

        else:
            t = '    arr' + \
                str(count) + ' = farray(data[' + \
                str(count) + "], "+str(size) + ", np.float64)"

            if len(size) == 1 and size[0] == 1:
                t += '[0]'

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


def generate_caller_array(setting, sdim):
    '''
    generate a callder function on the fly

    ex)
    if setting is
        {"isdepcomplex": (True, False), "kinds": (1, 0),
                       "output": True, size: ((3, 3), 1), outsize: (2, 2) }

    def _caller(ptx, sdim, data, out_):
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
        text = ['def _caller(ptx, sdim, t, data, out_):']
    else:
        text = ['def _caller(ptx, sdim, data, out_):']
    text.append("    ptx = farray(ptx, (sdim,), np.float64)")
    count = 0
    params_line = '    params = ('

    for s, kind, size in zip(setting['isdepcomplex'], setting['kinds'], setting["sizes"]):
        if not isinstance(size, tuple):
            size = (size, )

        if s:
            t1 = '    arrr' + \
                str(count) + ' = farray(data[' + \
                str(count) + "], "+str(size) + ", np.float64)"
            t2 = '    arri' + \
                str(count) + ' = farray(data[' + \
                str(count+1) + "], "+str(size) + ", np.float64)"
            if len(size) == 1 and size[0] == 1:
                t1 += '[0]'
                t2 += '[0]'

            t3 = '    arr'+str(count) + ' = arrr' + \
                str(count) + "+1j*arri" + str(count)

            text.extend((t1, t2, t3))
            params_line += 'arr'+str(count)+','
            count = count + 2
        else:
            if not isinstance(size, tuple):
                size = (size, )
            t = '    arr' + \
                str(count) + ' = farray(data[' + \
                str(count) + "]," + str(size) + ", np.float64)"
            if len(size) == 1 and size[0] == 1:
                t += '[0]'

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
        {"isdepcomplex": (2, 1), "kinds": (1, 0), "output": 2}

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

    for s, kind, in zip(setting['isdepcomplex'], setting['kinds'],):
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
        {"isdepcomplex": (2, 1), "kinds": (1, 0), "output": 2}

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
            sig += 'types.float64[:,:]('

    for i in range(sdim):
        sig += 'types.double, '

    if setting['td']:
        sig += 'types.double, '

    for s, kind, in zip(setting['isdepcomplex'], setting['kinds'],):
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
