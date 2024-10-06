'''
   rational approximation utility

     find_decomposition(func, x, xp=None, viewer=None, mmin=2,  mmax=5, **kwargs)

       find an approximation for function <func> over x range given by <x>

       plot option:
           xp : x range used in plotting
           viewer : a viewer used for plotting. Typically give figure()

       number of terms:  mmin - mmax

     find_decompositions(funcs, x, xp=None, viewer=None, mmin=2,  mmax=5, **kwargs)

       a version using set-valued AAA. funcs are approximated using the same roots.
 

'''
import numpy as np
#from baryrat import aaa

'''
AAA
'''


class aaa_fit():
    def __init__(self, z, w, f, scale=1):
        self.nodes = np.array(z)
        self.weights = np.array(w)
        self.values = np.array(f)*scale

    def __call__(self, x):
        n = np.sum([w*f/(x-z)
                    for w, f, z in zip(self.weights, self.values, self.nodes)])
        d = np.sum([w/(x-z)
                    for w, f, z in zip(self.weights, self.values, self.nodes)])
        return n/d


def aaa(x, f, tol=1e-10, mmax=-1):
    '''
    Y. Tanakzukasa "The AAA algorithm for rational approximation", SIAM Journal on Sci. Comp. (2018)
    '''
    # length
    ll = len(x)
    if mmax < 0:
        mmax = ll/10
    # scale it to one:
    scale = np.max(f) - np.min(f)
    f1 = f/scale

    flags = [True]*ll

    idx = np.argmax(f1)
    flags[idx] = False
    farr = [f1[idx]]
    zarr = [x[idx]]
    weights = [1]

    count = 0
    while count < mmax:
        #print("count", count, maxcount)
        r = aaa_fit(zarr, weights, farr)
        err = [np.abs(f1[i] - r(x[i]))
               if flags[i] else 0 for i in range(ll)]

        if np.max(err) < tol:
            break
        idx = np.argmax(err)
        farr.append(f1[idx])
        zarr.append(x[idx])
        flags[idx] = False

        mat = np.zeros((ll-len(farr), len(farr)))
        ii = 0
        for i in range(ll):
            if not flags[i]:
                continue
            for j in range(len(farr)):
                mat[ii, j] = (f1[i] - farr[j])/(x[i] - zarr[j])
            ii = ii + 1

        u, s, vh = np.linalg.svd(mat, full_matrices=True)
        count = count + 1
        weights = vh[-1, :]

    r = aaa_fit(zarr, weights, farr, scale=scale)

    return r


def aaaa(x, f, tol=1e-10, mmax=-1):
    '''
    array-AAA (set-valued AAA))

       f(N, len(x)) : 2D array

    P. Lietaert. "Automatic rational approximation and linearization of
    nonlinear eigenvalue problems", IMA Journal of Numerical Analysis (2022)
    '''
    # length
    ll = len(x)
    N = f.shape[0]
    if mmax < 0:
        mmax = ll/10
    # scale it to one:
    scales = np.array([np.max(ff) - np.min(ff) for ff in f])

    f1 = np.transpose(f.transpose()/scales)

    flags = [True]*ll

    idx = np.argmax(f1) % ll
    flags[idx] = False
    farrs = np.array([f1[i, idx] for i in range(N)]).reshape(N, 1)
    zarr = [x[idx]]
    weights = [1]

    count = 0
    while count < mmax:
        #print("count", count, maxcount)
        r = [aaa_fit(zarr, weights, farr)
             for farr in farrs]
        err = np.array([[np.abs(f1[j, i] - r[j](x[i]))
                         if flags[i] else 0 for i in range(ll)]
                        for j in range(N)])

        if np.max(err) < tol:
            break
        idx = np.argmax(err) % ll

        tmp = np.array([f1[i, idx] for i in range(N)]).reshape(N, 1)
        farrs = np.hstack((farrs, tmp))
        zarr.append(x[idx])
        flags[idx] = False

        mat = np.zeros(((ll-len(zarr))*N, len(zarr)))
        ii = 0

        for kk in range(N):
            for i in range(ll):
                if not flags[i]:
                    continue
                for j in range(len(zarr)):
                    mat[ii, j] = (f1[kk][i] - farrs[kk, j])/(x[i] - zarr[j])
                ii = ii + 1

        u, s, vh = np.linalg.svd(mat, full_matrices=True)

        count = count + 1
        weights = vh[-1, :]

    ret = [aaa_fit(zarr, weights, farr, scale=s)
           for farr, s in zip(farrs, scales)]

    return ret


def P(r):
    '''
    compute numerator of approximation f(x) = P(x)/Q(x)
    '''
    z = r.nodes
    w = r.weights
    f = r.values

    nterms = len(f)

    poly = np.poly1d([])

    for i in range(nterms):
        tmp = np.poly1d([f[i]*w[i]])
        for j in range(nterms):
            if i == j:
                continue
            tmp = tmp * np.poly1d([1, -z[j]])
        poly = poly + tmp

    return poly


def Q(r):
    '''
    compute denominaator of approximation f(x) = P(x)/Q(x)
    '''
    z = r.nodes
    w = r.weights
    f = r.values

    nterms = len(f)

    poly = np.poly1d([])

    for i in range(nterms):
        tmp = np.poly1d([w[i]])
        for j in range(nterms):
            if i == j:
                continue
            tmp = tmp * np.poly1d([1, -z[j]])

        poly = poly + tmp
    return poly


class poly_fraction():
    def __init__(self, c0, c_arr, d_arr):
        self.c0 = c0
        self.c_arr = c_arr
        self.d_arr = d_arr

    def __call__(self, x):
        value = np.zeros(x.shape, dtype=np.complex128) + self.c0
        for c, d in zip(self.c_arr, self.d_arr):
            value = value + c / (x - d)

        return value

    def __repr__(self):
        txt = ["fractional sum",
               "c0: " + str(self.c0),
               "c and d: "]

        for c, d in zip(self.c_arr, self.d_arr):
            txt.append(str((c, d)))
        return "\n".join(txt)+"\n"


def calc_decomposition(func, x, mmax, xp=None, viewer=None, **kwargs):
    if xp is None:
        xp = x

    f = np.array([func(xx, **kwargs) for xx in x])

    if viewer is not None:
        fp = np.array([func(xx, **kwargs) for xx in xp])
        viewer.plot(xp, fp, 'r')
        if np.iscomplexobj(f):
            viewer.plot(xp, fp.imag, 'b')
        viewer.xlabel("x")

    r = aaa(x, f, tol=0, mmax=mmax)
    # if viewer is not None:
    #    viewer.plot(x, r(x), 'ro')
    #    if np.iscomplexobj(f):
    #        viewer.plot(x, r(x).imag, 'bo')
    poly_p = P(r)
    poly_q = Q(r)
    poly_q_prime = poly_q.deriv()

    roots = np.roots(poly_q)

    c_arr = []
    d_arr = []
    for root in roots:
        c_arr.append(poly_p(root)/poly_q_prime(root))
        d_arr.append(root)

    c_arr = np.array(c_arr)
    d_arr = np.array(d_arr)

    if np.any([d.real > 0 and d.imag == 0 for d in roots]):
        import warnings
        warnings.warn("Decomposition is not stable. ", RuntimeWarning)

    tmp = 0j
    for c, d in zip(c_arr, d_arr):
        tmp = tmp + c/(x[0]-d)

    c0 = func(x[0], **kwargs)-tmp

    f_sum = poly_fraction(c0, c_arr, d_arr)

    fit = np.array([f_sum(xx) for xx in xp])
    if viewer is not None:
        viewer.plot(xp, fit, 'g--')
        if np.iscomplexobj(f):
            viewer.plot(xp, fit.imag, 'g--')

    max_error = np.max(np.abs(f - fit))
    return f_sum, max_error


def calc_decompositions(funcs, x, mmax, xp, viewer=None, **kwargs):
    '''
    array version
    '''
    f = np.vstack([np.array([func(xx, **kwargs) for xx in x])
                   for func in funcs])

    fps = []
    for func in funcs:
        fps.append(np.array([func(xx, **kwargs) for xx in xp]))
    if viewer is not None:
        for fp in fps:
            viewer.plot(np.sqrt(xp), fp, 'r')
            if np.iscomplexobj(f):
                viewer.plot(np.sqrt(xp), fp.imag, 'b')
        viewer.xlabel("sqrt(x)")

    #from baryrat import aaa
    rall = aaaa(x, f, mmax=mmax, tol=0)

    f_sums = []
    fits = []
    for r, func in zip(rall, funcs):
        poly_p = P(r)
        poly_q = Q(r)
        poly_q_prime = poly_q.deriv()

        roots = np.roots(poly_q)

        c_arr = []
        d_arr = []
        for root in roots:
            c_arr.append(poly_p(root)/poly_q_prime(root))
            d_arr.append(root)

        if np.any([d.real > 0 and d.imag == 0 for d in roots]):
            import warnings
            warnings.warn("Decomposition is not stable. ", RuntimeWarning)

        c_arr = np.array(c_arr)
        d_arr = np.array(d_arr)

        tmp = 0j
        for c, d in zip(c_arr, d_arr):
            tmp = tmp + c/(x[0]-d)

        c0 = func(x[0], **kwargs)-tmp
        f_sum = poly_fraction(c0, c_arr, d_arr)
        fit = np.array([f_sum(xx) for xx in xp])
        f_sums.append(f_sum)
        fits.append(fit)

    if viewer is not None:
        for fit in fits:
            viewer.plot(np.sqrt(xp), fit, 'g--')
            if np.iscomplexobj(f):
                viewer.plot(np.sqrt(xp), fit.imag, 'g--')

    errors = [np.max(np.abs(fp - fit)) for fp, fit in zip(fps, fits)]

    return f_sums, errors


def find_decomposition(func, x, xp=None, viewer=None, mmin=2, mmax=8,
                       tol=None, **kwargs):
    '''
    find rational approximation of func (callable)
       x = parameter range to fit
       tol = max error measured fitting points
    '''
    mm = mmin
    while mm <= mmax:
        succsss = False
        fit, err = calc_decomposition(func, x, mm, xp=xp, viewer=viewer, **kwargs)
        if fit is not None:
            success = np.all([d.real < 0 for d in fit.d_arr])
            if success:
                if tol is None or err < tol:
                    break

        mm = mm + 1
    return fit, success, err


def find_decompositions(funcs, x, viewer=None, xp=None,
                        mmin=3, mmax=15, **kwargs):
    if xp is None:
        xp = x

    mm = mmin
    success = False
    while mm <= mmax:
        fit, errors = calc_decompositions(
            funcs, x, mm, xp=xp, viewer=viewer, **kwargs)

        d_arr = fit[0].d_arr
        fail = False
        for d in d_arr:
            if d.imag == 0 and d.real > 0:
                fail = True
        if fail:
            mm = mm + 1
            continue
        for d in d_arr:
            if d.imag == 0 and d.real < 0:
                success = True
                break
        if success:
            break
        mm = mm + 1

    return fit, success, max(errors)
