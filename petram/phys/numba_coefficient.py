
'''

  NumbaCoefficient

   utility to use NumbaCoefficient more easily

'''
from numpy.linalg import inv, det
from numpy import array
from petram.mfem_config import use_parallel

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('NumbaCoefficient')


class NumbaCoefficient():
    def __init__(self, coeff):
        self.is_complex = coeff.IsOutComplex()
        self.mfem_numba_coeff = coeff

        if self.is_complex:
            self.real = coeff.real
            self.imag = coeff.imag
        else:
            self.real = coeff
            self.imag = None

        if self.ndim == 1:
            self._V = mfem.Vector(self.shape[0])
        if self.ndim == 2:
            self._K = mfem.DenseMatrix(self.shape[0], self.shape[1])

    @property
    def complex(self):
        return self.is_complex

    def get_real_coefficient(self):
        return self.real

    def get_imag_coefficient(self):
        return self.imag

    @property
    def sdim(self):
        return self.mfem_numba_coeff.SpaceDimension()

    @property
    def ndim(self):
        return self.mfem_numba_coeff.GetNDim()

    @property
    def shape(self):
        if self.ndim == 0:
            return tuple()
        if self.ndim == 1:
            return (self.mfem_numba_coeff.GetVDim(),)
        if self.ndim == 2:
            return (self.mfem_numba_coeff.GetWidth(),
                    self.mfem_numba_coeff.GetHeight(),)

        assert False, "unsupported dim"
        return None

    @property
    def width(self):
        return self.mfem_numba_coeff.GetWidth()

    @property
    def height(self):
        return self.mfem_numba_coeff.GetHeight()

    @property
    def vdim(self):
        return self.mfem_numba_coeff.GetVDim()

    @property
    def kind(self):
        if self.ndim == 0:
            return "scalar"
        if self.ndim == 1:
            return "vector"
        if self.ndim == 2:
            return "matrix"
        assert False, "unsupported dim"
        return None

    def eval(self, T, ip):
        if self.ndim == 0:
            if self.real is not None:
                ret = self.coeff1.Eval(T, ip)
            else:
                ret = 0
            if self.imag is not None:
                ret = ret + 1j * self.imag.Eval(T, ip)
            return ret
        if self.ndim == 1:
            if self.real is not None:
                self.real.Eval(self._V, T, ip)
            else:
                self._V.Assign(0.0)
            vec = self._V.GetDataArray().copy()
            if self.imag is not None:
                self.imag.Eval(self._V, T, ip)
                vec = vec + 1j * self._V.GetDataArray()
            return vec
        if self.ndim == 2:
            if self.real is not None:
                self.real.Eval(self._K, T, ip)
            else:
                self._K.Assign(0.0)
            mat = self._K.GetDataArray().copy()
            if self.imag is not None:
                self.imag.Eval(self._K, T, ip)
                mat = mat + 1j * self._K.GetDataArray()
            return mat
        return None

    def is_matrix(self):
        return self.mfem_numba_coeff.GetNdim() == 2

    def is_vector(self):
        return self.mfem_numba_coeff.GetNdim() == 1

    def __add__(self, other):
        '''
        ruturn sum coefficient
        '''
        from petram.phys.phys_model import (PhysConstant,
                                            PhysVectorConstant,
                                            PhysMatrixConstant,)
        from petram.phys.pycomplex_coefficient import (PyComplexConstant,
                                                       PyComplexVectorConstant,
                                                       PyComplexMatrixConstant,)
        from petram.mfem_config import numba_debug

        if not isinstance(other, NumbaCoefficient):
            if isinstance(other, (PhysConstant,
                                  PhysVectorConstant,
                                  PhysMatrixConstant,
                                  PyComplexConstant,
                                  PyComplexVectorConstant,
                                  PyComplexMatrixConstant,)):
                params = {"value": other.value}
                dep = (self.mfem_numba_coeff, )
                func = '\n'.join(['def f(ptx, coeff1):',
                                  '    return coeff1 + value'])

            else:
                return NotImplemented
        else:
            assert self.shape == other.shape, "ndim must match to perform sum operation"
            dep = (self.mfem_numba_coeff, other.mfem_numba_coeff)
            params = None
            func = '\n'.join(['def f(ptx, coeff1, coeff2):',
                              '    return coeff1 + coeff2'])

        l = {}
        exec(func, globals(), l)
        if self.ndim == 0:
            coeff = mfem.jit.scalar(sdim=self.sdim,
                                    complex=self.complex,
                                    dependency=dep,
                                    params=params,
                                    debug=numba_debug)(l["f"])
        elif self.ndim == 1:
            coeff = mfem.jit.vector(sdim=self.sdim,
                                    complex=self.complex,
                                    dependency=dep,
                                    debug=numba_debug,
                                    params=params,
                                    shape=self.shape)(l["f"])

        elif self.ndim == 2:
            coeff = mfem.jit.matrix(sdim=self.sdim,
                                    complex=self.complex,
                                    dependency=dep,
                                    debug=numba_debug,
                                    params=params,
                                    shape=self.shape)(l["f"])

        else:
            assert False, "unsupported dim: dim=" + str(self.ndim)

        return NumbaCoefficient(coeff)

    def __pow__(self, exponent):
        raise NotImplementedError

    def __mul__(self, scale):
        from petram.mfem_config import numba_debug

        func = '\n'.join(['def f(ptx, val):',
                          '    return val*scale'])

        l = {}
        if numba_debug:
            print("(DEBUG) numba function\n", func)
        exec(func, globals(), l)

        dep = (self.mfem_numba_coeff, )
        params = {'scale': scale}

        coeff = mfem.jit.matrix(sdim=self.sdim,
                                complex=self.complex,
                                dependency=dep,
                                shape=self.shape,
                                interface="simple",
                                params=params,
                                debug=numba_debug)(l["f"])
        return NumbaCoefficient(coeff)

    def __getitem__(self, arg):
        check = self.kind == 'matrix' or self.kind == 'vector'
        assert check, "slice is valid for vector and matrix"

        from petram.mfem_config import numba_debug

        coeff = None
        dep = (self.mfem_numba_coeff, )

        if self.kind == "vector":
            slice1 = arg
            try:
                a = slice1[0]
            except:
                slice1 = (slice1, )

            func = '\n'.join(['def f(ptx, coeff1):',
                              '    return coeff1[slice1'])
            if numba_debug:
                print("(DEBUG) numba function\n", func)
            l = {}
            exec(func, globals(), l)

            if len(slice1) == 1:
                params = {"slice1": slice1[0]}
                coeff = mfem.jit.scalar(sdim=self.sdim,
                                        complex=self.complex,
                                        dependency=dep,
                                        params=params,
                                        debug=numba_debug)(l["f"])
            elif len(slice1) == 1:
                params = {"slice1": slice1}
                coeff = mfem.jit.vector(sdim=self.sdim,
                                        complex=self.complex,
                                        dependency=dep,
                                        params=params,
                                        shape=(len(slice1),),
                                        debug=numba_debug)(l["f"])

        if self.kind == "matrix":
            slice1, slice2 = arg
            try:
                a = slice1[0]
            except:
                slice1 = (slice1, )
            try:
                a = slice2[0]
            except:
                slice2 = (slice2, )

            func1 = '\n'.join(['def f(ptx, coeff1):',
                               '    return coeff1[slice1, slice2]'])
            func2 = '\n'.join(['def f(ptx, coeff1, out):',
                               '    for ii in range(shape[0]):',
                               '        out[ii] = coeff1[slice1, slice2[ii]]'])
            func3 = '\n'.join(['def f(ptx, coeff1, out):',
                               '    for ii in range(shape[0]):',
                               '        out[ii] = coeff1[slice1[ii], slice2]'])
            func4 = '\n'.join(['def f(ptx, coeff1, out):',
                               '    for ii in range(shape[0]):',
                               '        for jj in range(shape[1]):',
                               '            out[ii, jj] = coeff1[slice1[ii], slice2[jj]]'])

            l = {}
            if len(slice1) == 1 and len(slice2) == 1:
                params = {"slice1": slice1[0], "slice2": slice2[0]}
                if numba_debug:
                    print("(DEBUG) numba function\n", func1)
                exec(func1, globals(), l)
                coeff = mfem.jit.scalar(sdim=self.sdim,
                                        complex=self.complex,
                                        dependency=dep,
                                        params=params,
                                        debug=numba_debug)(l["f"])
            elif len(slice1) == 1 and len(slice2) > 1:
                params = {"slice1": slice1[0],
                          "slice2": array(slice2, dtype=int)}
                if numba_debug:
                    print("(DEBUG) numba function\n", func2)

                exec(func2, globals(), l)
                coeff = mfem.jit.vector(sdim=self.sdim,
                                        complex=self.complex,
                                        dependency=dep,
                                        params=params,
                                        shape=(len(slice2),),
                                        interface="c++",
                                        debug=numba_debug)(l["f"])
            elif len(slice1) > 1 and len(slice2) == 1:
                params = {"slice1": array(slice1, dtype=int),
                          "slice2": slice2[0]}
                if numba_debug:
                    print("(DEBUG) numba function\n", func3)

                exec(func3, globals(), l)
                coeff = mfem.jit.vector(sdim=self.sdim,
                                        complex=self.complex,
                                        dependency=dep,
                                        params=params,
                                        shape=(len(slice1),),
                                        interface="c++",
                                        debug=numba_debug)(l["f"])
            else:
                params = {"slice1": array(slice1, dtype=int),
                          "slice2": array(slice2, dtype=int), }
                if numba_debug:
                    print("(DEBUG) numba function\n", func4)
                exec(func4, globals(), l)
                coeff = mfem.jit.matrix(sdim=self.sdim,
                                        complex=self.complex,
                                        dependency=dep,
                                        params=params,
                                        shape=(len(slice1), len(slice2)),
                                        interface="c++",
                                        debug=numba_debug)(l["f"])

        assert coeff is not None, "coeff is not build during __getitem__ in NumbaCoefficient"
        return NumbaCoefficient(coeff)

    def inv(self):
        from petram.mfem_config import numba_debug

        func = '\n'.join(['def f(ptx, coeff1):',
                          '    return inv(coeff1)'])
        l = {}
        if numba_debug:
            print("(DEBUG) numba function\n", func)
        exec(func, globals(), l)

        dep = (self.mfem_numba_coeff, )

        coeff = mfem.jit.matrix(sdim=self.sdim,
                                complex=self.complex,
                                dependency=dep,
                                shape=self.shape,
                                interface="simple",
                                debug=numba_debug)(l["f"])
        return NumbaCoefficient(coeff)

    def adj(self):
        from petram.mfem_config import numba_debug

        func = '\n'.join(['def f(ptx, mat):',
                          '      determinant = det(mat)',
                          '      mat = inv(mat)',
                          '      return mat * determinant'])

        l = {}
        if numba_debug:
            print("(DEBUG) numba function\n", func)
        exec(func, globals(), l)

        dep = (self.mfem_numba_coeff, )

        coeff = mfem.jit.matrix(sdim=self.sdim,
                                complex=self.complex,
                                dependency=dep,
                                shape=self.shape,
                                interface="simple",
                                debug=numba_debug)(l["f"])
        return NumbaCoefficient(coeff)
