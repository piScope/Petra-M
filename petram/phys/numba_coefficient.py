'''

  NumbaCoefficient

   utility to use NumbaCoefficient more easily

'''
import traceback

from numpy.linalg import inv, det
from numpy import conj as npconj
from numpy import dot as npdot
from numpy import array, zeros, iscomplexobj
from petram.mfem_config import use_parallel

if use_parallel:
    import mfem.par as mfem
    from mpi4py import MPI
    myid = MPI.COMM_WORLD.rank

else:
    import mfem.ser as mfem
    myid = 0

from petram.helper.variables import (Variable,
                                     NativeCoefficientGenBase)


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

    def get_realimag_coefficient(self, real):
        if real:
            return self.get_real_coefficient()
        else:
            return self.get_imag_coefficient()

    # @property
    # def sdim(self):
    #    return self.mfem_numba_coeff.SpaceDimension()

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
        numba_debug = False if myid != 0 else numba_debug

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
            coeff = mfem.jit.scalar(complex=self.complex,
                                    dependency=dep,
                                    params=params,
                                    debug=numba_debug)(l["f"])
        elif self.ndim == 1:
            coeff = mfem.jit.vector(complex=self.complex,
                                    dependency=dep,
                                    debug=numba_debug,
                                    params=params,
                                    shape=self.shape)(l["f"])

        elif self.ndim == 2:
            coeff = mfem.jit.matrix(complex=self.complex,
                                    dependency=dep,
                                    debug=numba_debug,
                                    params=params,
                                    shape=self.shape)(l["f"])

        else:
            assert False, "unsupported dim: dim=" + str(self.ndim)

        return NumbaCoefficient(coeff)

    def __sub__(self, other):
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
        numba_debug = False if myid != 0 else numba_debug

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
                                  '    return coeff1 - value'])

            else:
                return NotImplemented
        else:
            assert self.shape == other.shape, "ndim must match to perform sum operation"
            dep = (self.mfem_numba_coeff, other.mfem_numba_coeff)
            params = None
            func = '\n'.join(['def f(ptx, coeff1, coeff2):',
                              '    return coeff1 - coeff2'])

        l = {}
        exec(func, globals(), l)
        if self.ndim == 0:
            coeff = mfem.jit.scalar(complex=self.complex,
                                    dependency=dep,
                                    params=params,
                                    debug=numba_debug)(l["f"])
        elif self.ndim == 1:
            coeff = mfem.jit.vector(complex=self.complex,
                                    dependency=dep,
                                    debug=numba_debug,
                                    params=params,
                                    shape=self.shape)(l["f"])

        elif self.ndim == 2:
            coeff = mfem.jit.matrix(complex=self.complex,
                                    dependency=dep,
                                    debug=numba_debug,
                                    params=params,
                                    shape=self.shape)(l["f"])

        else:
            assert False, "unsupported dim: dim=" + str(self.ndim)

        return NumbaCoefficient(coeff)

    def __mul__(self, other):
        from petram.phys.phys_model import (PhysConstant,
                                            PhysVectorConstant,
                                            PhysMatrixConstant,)
        from petram.phys.pycomplex_coefficient import (PyComplexConstant,
                                                       PyComplexVectorConstant,
                                                       PyComplexMatrixConstant,)

        from petram.mfem_config import numba_debug
        numba_debug = False if myid != 0 else numba_debug

        dep = (self.mfem_numba_coeff, )

        if not isinstance(other, NumbaCoefficient):
            if isinstance(other, (PhysConstant,
                                  PhysVectorConstant,
                                  PhysMatrixConstant,
                                  PyComplexConstant,
                                  PyComplexVectorConstant,
                                  PyComplexMatrixConstant,)):
                params = {"scale": other.value}
            else:
                params = {"scale": other}  # this assuems other is a number

            func = '\n'.join(['def f(ptx, val):',
                              '    return val*scale'])
            is_other_complex = iscomplexobj(params["scale"])

        else:
            assert len(other.shape) == 0, "scale must be scalar (__mul__)"
            dep = (self.mfem_numba_coeff, other.mfem_numba_coeff)
            params = None
            func = '\n'.join(['def f(ptx, val, scale):',
                              '    return val*scale'])
            is_other_complex = other.complex

        l = {}
        if numba_debug:
            print("(DEBUG) numba function\n", func)
        exec(func, globals(), l)

        do_complex = self.complex or is_other_complex

        if self.ndim == 0:
            coeff = mfem.jit.scalar(complex=do_complex,
                                    dependency=dep,
                                    interface="simple",
                                    params=params,
                                    debug=numba_debug)(l["f"])
        elif self.ndim == 1:
            coeff = mfem.jit.vector(complex=do_complex,
                                    dependency=dep,
                                    debug=numba_debug,
                                    interface="simple",
                                    params=params,
                                    shape=self.shape)(l["f"])

        elif self.ndim == 2:
            coeff = mfem.jit.matrix(complex=do_complex,
                                    dependency=dep,
                                    debug=numba_debug,
                                    interface="simple",
                                    params=params,
                                    shape=self.shape)(l["f"])

        else:
            assert False, "unsupported dim: dim=" + str(self.ndim)

        return NumbaCoefficient(coeff)

    def __rmul__(self, scale):
        return self.__mul__(scale)

    def __div__(self, scale):
        return self.__truediv__(scale)

    def __truediv__(self, other):
        from petram.phys.phys_model import (PhysConstant,
                                            PhysVectorConstant,
                                            PhysMatrixConstant,)
        from petram.phys.pycomplex_coefficient import (PyComplexConstant,
                                                       PyComplexVectorConstant,
                                                       PyComplexMatrixConstant,)

        from petram.mfem_config import numba_debug
        numba_debug = False if myid != 0 else numba_debug

        dep = (self.mfem_numba_coeff, )

        if not isinstance(other, NumbaCoefficient):
            if isinstance(ohter, (PhysConstant,
                                  PhysVectorConstant,
                                  PhysMatrixConstant,
                                  PyComplexConstant,
                                  PyComplexVectorConstant,
                                  PyComplexMatrixConstant,)):
                params = {"scale": other.value}
            else:
                params = {"scale": other}  # this assuems other is a number

            func = '\n'.join(['def f(ptx, val):',
                              '    return val/scale'])
            is_other_complex = iscomplexobj(params["scale"])

        else:
            assert len(other.shape) == 0, "scale must be scalar (__truediv__)"
            dep = (self.mfem_numba_coeff, other.mfem_numba_coeff)

            params = None
            func = '\n'.join(['def f(ptx, val, scale):',
                              '    return val/scale'])
            is_other_complex = other.complex

        l = {}
        if numba_debug:
            print("(DEBUG) numba function\n", func)
        exec(func, globals(), l)

        do_complex = self.complex or is_other_complex

        if self.ndim == 0:
            coeff = mfem.jit.scalar(complex=do_complex,
                                    dependency=dep,
                                    interface="simple",
                                    params=params,
                                    debug=numba_debug)(l["f"])
        elif self.ndim == 1:
            coeff = mfem.jit.vector(complex=do_complex,
                                    dependency=dep,
                                    debug=numba_debug,
                                    interface="simple",
                                    params=params,
                                    shape=self.shape)(l["f"])

        elif self.ndim == 2:
            coeff = mfem.jit.matrix(complex=do_complex,
                                    dependency=dep,
                                    debug=numba_debug,
                                    interface="simple",
                                    params=params,
                                    shape=self.shape)(l["f"])

        else:
            assert False, "unsupported dim: dim=" + str(self.ndim)

        return NumbaCoefficient(coeff)

    def __rdiv__(self, scale):
        from petram.mfem_config import numba_debug
        numba_debug = False if myid != 0 else numba_debug

        func = '\n'.join(['def f(ptx, val):',
                          '    return scale/val'])

        l = {}
        if numba_debug:
            print("(DEBUG) numba function\n", func)
        exec(func, globals(), l)

        dep = (self.mfem_numba_coeff, )
        params = {'scale': scale}

        if self.ndim == 0:
            coeff = mfem.jit.scalar(complex=self.complex,
                                    dependency=dep,
                                    interface="simple",
                                    params=params,
                                    debug=numba_debug)(l["f"])
        elif self.ndim == 1:
            coeff = mfem.jit.vector(complex=self.complex,
                                    dependency=dep,
                                    debug=numba_debug,
                                    interface="simple",
                                    params=params,
                                    shape=self.shape)(l["f"])

        elif self.ndim == 2:
            coeff = mfem.jit.matrix(complex=self.complex,
                                    dependency=dep,
                                    debug=numba_debug,
                                    interface="simple",
                                    params=params,
                                    shape=self.shape)(l["f"])

        else:
            assert False, "unsupported dim: dim=" + str(self.ndim)

        return NumbaCoefficient(coeff)

    def __rtruediv__(self, scale):
        return self.__rdiv__(scale)

    def __pos__(self):
        raise NotImplementedError

    def __neg__(self):
        from petram.mfem_config import numba_debug
        numba_debug = False if myid != 0 else numba_debug

        func = '\n'.join(['def f(ptx, val):',
                          '    return -val'])

        l = {}
        if numba_debug:
            print("(DEBUG) numba function\n", func)
        exec(func, globals(), l)

        dep = (self.mfem_numba_coeff, )

        if self.ndim == 0:
            coeff = mfem.jit.scalar(complex=self.complex,
                                    dependency=dep,
                                    interface="simple",
                                    debug=numba_debug)(l["f"])
        elif self.ndim == 1:
            coeff = mfem.jit.vector(complex=self.complex,
                                    dependency=dep,
                                    debug=numba_debug,
                                    interface="simple",
                                    shape=self.shape)(l["f"])

        elif self.ndim == 2:
            coeff = mfem.jit.matrix(complex=self.complex,
                                    dependency=dep,
                                    debug=numba_debug,
                                    interface="simple",
                                    shape=self.shape)(l["f"])

        else:
            assert False, "unsupported dim: dim=" + str(self.ndim)

        return NumbaCoefficient(coeff)

    def conj(self):
        '''
        generate conjugate
        '''
        from petram.mfem_config import numba_debug
        numba_debug = False if myid != 0 else numba_debug

        func = '\n'.join(['def f(ptx, val):',
                          '    return npconj(val)'])

        l = {}
        if numba_debug:
            print("(DEBUG) numba function\n", func)
        exec(func, globals(), l)

        dep = (self.mfem_numba_coeff, )

        if self.ndim == 0:
            coeff = mfem.jit.scalar(complex=self.complex,
                                    dependency=dep,
                                    interface="simple",
                                    debug=numba_debug)(l["f"])
        elif self.ndim == 1:
            coeff = mfem.jit.vector(complex=self.complex,
                                    dependency=dep,
                                    debug=numba_debug,
                                    interface="simple",
                                    shape=self.shape)(l["f"])

        elif self.ndim == 2:
            coeff = mfem.jit.matrix(complex=self.complex,
                                    dependency=dep,
                                    debug=numba_debug,
                                    interface="simple",
                                    shape=self.shape)(l["f"])

        else:
            assert False, "unsupported dim: dim=" + str(self.ndim)

        return NumbaCoefficient(coeff)

    def dot(self, other):
        '''
        dot product matrix*matrix etc
        '''
        from petram.mfem_config import numba_debug
        numba_debug = False if myid != 0 else numba_debug

        assert len(self.shape) > 0, ".dot is defined only for vector/matrix"

        if not isinstance(other, NumbaCoefficient):
            if isinstance(ohter, (PhysVectorConstant,
                                  PhysMatrixConstant,
                                  PyComplexVectorConstant,
                                  PyComplexMatrixConstant,)):
                params = {"scale": other.value}
            else:
                params = {"scale": other}  # this assuems other is a number

            func = '\n'.join(['def f(ptx, val):',
                              '    return npdot(val, scale)'])
            is_other_complex = iscomplexobj(params["scale"])
            other_shape = params["scale"].shape

        else:
            assert len(other.shape) > 0, "other must be vector/matrix"

            dep = (self.mfem_numba_coeff, other.mfem_numba_coeff)

            params = None
            func = '\n'.join(['def f(ptx, val, scale):',
                              '    return npdot(val, scale)'])
            is_other_complex = other.complex
            other_shape = other.shape

        if len(self.shape) == 2:
            # matrix * matrix or matrix * vector
            if len(other_shape) == 1:
                assert self.shape[1] == other_shape[0], "dimenstion does not match (mat*vec)" + str(
                    self.shape) + ":" + str(other_shape)
                out_shape = (self.shape[0],)
            else:
                assert self.shape[1] == other_shape[0], "dimenstion does not match (mat*mat)" + str(
                    self.shape) + ":" + str(other_shape)
                out_shape = (self.shape[0], other_shape[1])
        elif len(self.shape) == 1:
            assert len(
                other_shape) == 1, "vector.dot only supports inner product"
            assert self.shape[0] == other_shape[0],  "dimenstion does not match (vec*vec)" + str(
                self.shape) + ":" + str(other_shape)
            out_shape = tuple()
        else:
            assert False, "should not come here (.dot)"

        is_complex = self.complex or is_other_complex

        if numba_debug:
            print("(DEBUG) numba function\n", func)
        l = {}
        exec(func, globals(), l)

        if len(out_shape) == 0:
            coeff = mfem.jit.scalar(complex=is_complex,
                                    dependency=dep,
                                    interface="simple",
                                    debug=numba_debug)(l["f"])
        elif len(out_shape) == 1:
            coeff = mfem.jit.vector(complex=is_complex,
                                    dependency=dep,
                                    debug=numba_debug,
                                    interface="simple",
                                    shape=out_shape)(l["f"])

        elif len(out_shape) == 2:
            coeff = mfem.jit.matrix(complex=is_complex,
                                    dependency=dep,
                                    debug=numba_debug,
                                    interface="simple",
                                    shape=out_shape)(l["f"])

        else:
            assert False, "unsupported dim: dim=" + str(self.ndim)

        return NumbaCoefficient(coeff)

    def __abs__(self, other):
        raise NotImplementedError

    def __pow__(self, exponent):
        raise NotImplementedError

    def __getitem__(self, arg):
        check = self.kind == 'matrix' or self.kind == 'vector'
        assert check, "slice is valid for vector and matrix"

        from petram.mfem_config import numba_debug
        numba_debug = False if myid != 0 else numba_debug

        coeff = None
        dep = (self.mfem_numba_coeff, )

        if self.kind == "vector":
            slice1 = arg

            try:
                a = slice1[0]
            except:
                slice1 = (slice1, )

            func = '\n'.join(['def f(ptx, coeff1):',
                              '    return coeff1[slice1]'])
            if numba_debug:
                print("(DEBUG) numba function\n", func)
            l = {}
            exec(func, globals(), l)

            if len(slice1) == 1:
                params = {"slice1": slice1[0]}
                coeff = mfem.jit.scalar(complex=self.complex,
                                        dependency=dep,
                                        params=params,
                                        debug=numba_debug)(l["f"])
            else:
                params = {"slice1": array(slice1, dtype=int)}
                coeff = mfem.jit.vector(complex=self.complex,
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
                coeff = mfem.jit.scalar(complex=self.complex,
                                        dependency=dep,
                                        params=params,
                                        debug=numba_debug)(l["f"])
            elif len(slice1) == 1 and len(slice2) > 1:
                params = {"slice1": slice1[0],
                          "slice2": array(slice2, dtype=int)}
                if numba_debug:
                    print("(DEBUG) numba function\n", func2)

                exec(func2, globals(), l)
                coeff = mfem.jit.vector(complex=self.complex,
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
                coeff = mfem.jit.vector(complex=self.complex,
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
                coeff = mfem.jit.matrix(complex=self.complex,
                                        dependency=dep,
                                        params=params,
                                        shape=(len(slice1), len(slice2)),
                                        interface="c++",
                                        debug=numba_debug)(l["f"])

        assert coeff is not None, "coeff is not build during __getitem__ in NumbaCoefficient"
        return NumbaCoefficient(coeff)

    def inv(self):
        from petram.mfem_config import numba_debug
        numba_debug = False if myid != 0 else numba_debug

        func = '\n'.join(['def f(ptx, coeff1):',
                          '    return inv(coeff1)'])
        l = {}
        if numba_debug:
            print("(DEBUG) numba function\n", func)
        exec(func, globals(), l)

        dep = (self.mfem_numba_coeff, )

        coeff = mfem.jit.matrix(complex=self.complex,
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

        coeff = mfem.jit.matrix(complex=self.complex,
                                dependency=dep,
                                shape=self.shape,
                                interface="simple",
                                debug=numba_debug)(l["f"])
        return NumbaCoefficient(coeff)


'''

  convert text to numba jittec coefficient

'''


def _expr_to_numba_coeff(txt, jitter, ind_vars, conj, scale, g, l,
                         return_complex, **kwargs):

    ind_vars = [xx.strip() for xx in ind_vars.split(',')]
    code = compile(txt.strip(), '<string>', 'eval')
    names = code.co_names

    dependency = []
    dep_names = []

    l = l.copy()
    for n in g:
        if isinstance(g[n], Variable):
            if n not in l:
                l[n] = g[n]
            else:
                assert False, "name confict: " + n + " is defined in local namespace"

    for n in names:
        if n in ind_vars:
            continue

        dep = None
        if n in l:
            if isinstance(l[n], Variable):
                gg = l[n]
                dep = gg.get_jitted_coefficient(ind_vars, l)
            else:
                continue
        elif n in g:
            # if isinstance(g[n], Variable):
            #    gg = g[n]
            #    dep = gg.get_jitted_coefficient(ind_vars, l)
            if isinstance(g[n], NativeCoefficientGenBase):
                from petram.phys.coefficient import call_nativegen
                c1 = call_nativegen(g[n], l, g, True, conj, scale)
                c2 = call_nativegen(g[n], l, g, False, conj, scale)
                dep = (c1, c2)
            else:
                continue
        if dep is None:
            #
            return None

        dependency.append(dep)
        dep_names.append(n)

    if jitter == mfem.jit.scalar:
        return_type = "complex128" if return_complex else "float64"
    elif jitter == mfem.jit.vector:
        return_type = "complex128[:]" if return_complex else "float64[:]"
    elif jitter == mfem.jit.matrix:
        return_type = "complex128[:,:]" if return_complex else "float64[:,:]"
    else:
        assert False, "unknown jitter: " + str(type(jitter))

    def create_func(objmode=False):
        f0 = 'def _func_(ptx, '
        for n in dep_names:
            f0 += n + ', '
        f0 += '):'

        func_txt = [f0]
        for k, xx in enumerate(ind_vars):
            func_txt.append("   " + xx + " = ptx[" + str(k) + "]")
        if objmode:
            func_txt.append("   with objmode(_out_='"+return_type+"'):")
            func_txt.append("       _out_ =" + txt)
        else:
            func_txt.append("   _out_ =" + txt)
        func_txt.append("   if isinstance(_out_, list):")
        func_txt.append("         _out_ = np.array(_out_)")
        func_txt.append("   elif isinstance(_out_, tuple):")
        func_txt.append("         _out_ = np.array(_out_)")
        if scale != 1:
            func_txt.append("   _out_ = _out_ * " + str(scale))
        if conj:
            func_txt.append("   _out_ = np.conj(_out_)")
        # func_txt.append("   print(_out_)")

        if jitter == mfem.jit.scalar:
            if return_complex:
                func_txt.append("   return np.complex128(_out_)")
            else:
                func_txt.append("   return np.float64(_out_)")
        else:
            if return_complex:
                func_txt.append("   return _out_.astype(np.complex128)")
            else:
                func_txt.append("   return _out_.astype(np.float64)")
        return func_txt

    func_txt = "\n".join(create_func())

    from petram.mfem_config import (numba_debug,
                                    get_allow_python_function_coefficient)

    numba_debug = False if myid != 0 else numba_debug

    if numba_debug:
        print("(DEBUG) wrapper function\n", func_txt)
    exec(func_txt, g, l)

    try:
        import traceback
        coeff = jitter(sdim=len(ind_vars), complex=return_complex, debug=numba_debug,
                       dependency=dependency, **kwargs)(l["_func_"])
        del l["_func_"]
    except AssertionError:
        if get_allow_python_function_coefficient() == "error":
            if myid == 0:
                traceback.print_exc()
                print("Can not JIT coefficient")
            return None

        if get_allow_python_function_coefficient() == "warn":
            if myid == 0:
                traceback.print_exc()
                print("problematic function is following")
                print(func_txt)

        if myid == 0:
            print("!!!! Failed to compile with nonpython mode. (next) Try object mode")

    except BaseException:
        if get_allow_python_function_coefficient() == "error":
            if myid == 0:
                traceback.print_exc()
                print("Can not JIT coefficient")
            return None

        if get_allow_python_function_coefficient() == "warn":
            if myid == 0:
                traceback.print_exc()
                print("problematic function is following")
                print(func_txt)

        if myid == 0:
            print("!!!! Failed to compile with nonpython mode. (next) Try object mode")

    else:
        return NumbaCoefficient(coeff)

    # for the moment suppress objmode creation
    # return None

    func_txt = "\n".join(create_func(True))

    from petram.mfem_config import (numba_debug,
                                    get_allow_python_function_coefficient)

    numba_debug = False if myid != 0 else numba_debug

    if numba_debug:
        print("(DEBUG) wrapper function\n", func_txt)

    exec(func_txt, g, l)

    try:
        from numba import objmode
        coeff = jitter(sdim=len(ind_vars), complex=return_complex, debug=numba_debug,
                       dependency=dependency, params={"objmode": objmode},
                       **kwargs)(l["_func_"])
        del l["_func_"]
    except AssertionError:
        import traceback
        traceback.print_exc()

        print("Can not JIT coefficient. No possible recoverly. ")
        return None
    except BaseException:
        import traceback
        traceback.print_exc()
        print("Can not JIT coefficient. No possible recovery.")
        return None

    return NumbaCoefficient(coeff)


def expr_to_numba_coeff(exprs, jitter, ind_vars, conj, scale, g, l, return_complex,
                        **kwargs):
    '''
    # generate a wrapper for multiple inputs
    def _func_(ptx, p1, p2, out)
        deps = [p1, p2]
        count = 0
        depcount = 0
        for i0 in range(shape[0]):
            for i1 in range(shape[1]):
                if isconst[count]:
                   value = consts[count]
                else:
                   value = deps[depcount]
                   depcount = depcount + 1
                out[i0:i1] = value
                count = count +1
    '''

    consts = [-1]*len(exprs)
    isconst = zeros(len(exprs))
    deps = []
    dep_names = []

    from petram.mfem_config import get_allow_python_function_coefficient
    if get_allow_python_function_coefficient() == "always use Python coeff.":
        return None

    if len(exprs) > 1:
        jitter2 = mfem.jit.scalar
        assert "shape" in kwargs, "multile expression array, but shape is not given"
        shape = kwargs.pop("shape")
    else:
        jitter2 = jitter

    nbcs = []
    for k, ee in enumerate(exprs):
        if isinstance(ee, str):
            nbc = _expr_to_numba_coeff(
                ee, jitter2, ind_vars, conj, scale, g, l, return_complex, **kwargs)
            if nbc is None:
                return None
            deps.append(nbc.mfem_numba_coeff)
            dep_names.append("p"+str(k))
            nbcs.append(nbc)
        else:
            isconst[k] = 1
            if scale != 1:
                ee = ee * scale
            if conj:
                ee = npconj(ee)
            consts[k] = ee
    if len(exprs) == 1:
        return nbc

    consts = array(consts)
    ind_vars = [xx.strip() for xx in ind_vars.split(',')]

    f0 = 'def _func_(ptx, '
    for n in dep_names:
        f0 += n + ', '
    f0 += 'out):'

    func_txt = [f0]
    func_txt.append("    count = 0")
    func_txt.append("    depcount = 0")
    func_txt.append("    deps = [" + ",".join(dep_names) + "]")

    idx_text = ""
    for k, s in enumerate(shape):
        func_txt.append("    " + " "*k + "for i" + str(k) +
                        " in range(" + str(s) + "):")
        idx_text = idx_text + "i"+str(k)+","

    func_txt.append("     " + " "*len(shape) + "if isconst[count] == 1:")
    func_txt.append("     " + " "*len(shape) + "    value = consts[count]")
    func_txt.append("     " + " "*len(shape) + "else:")
    func_txt.append("     " + " "*len(shape) + "    value = deps[depcount]")
    func_txt.append("     " + " "*len(shape) + "    depcount = depcount + 1")
    func_txt.append("     " + " "*len(shape) + "out["+idx_text + "]=value")
    func_txt.append("     " + " "*len(shape) + "count = count + 1")

    func_txt = "\n".join(func_txt)

    g = globals()
    l = {}

    from petram.mfem_config import numba_debug
    numba_debug = False if myid != 0 else numba_debug

    if numba_debug:
        print("(DEBUG) wrapper function\n", func_txt)
    exec(func_txt, g, l)

    params = {}
    params["isconst"] = isconst
    params["consts"] = consts

    coeff = jitter(sdim=len(ind_vars),
                   complex=return_complex,
                   debug=numba_debug,
                   shape=shape,
                   dependency=deps,
                   interface="c++",
                   params=params,
                   **kwargs)(l["_func_"])

    ret = NumbaCoefficient(coeff)
    del l["_func_"]
    return ret


'''
convert function to numba coefficient
'''


def func_to_numba_coeff_scalar(func, complex=False,
                               params=None, dependency=None):
    from petram.mfem_config import numba_debug
    numba_debug = False if myid != 0 else numba_debug

    if dependency is None:
        dependency = tuple()

    dependency = [(x.mfem_numba_coeff if isinstance(x, NumbaCoefficient) else x)
                  for x in dependency]

    jitter = mfem.jit.scalar(complex=complex, params=params,
                             debug=numba_debug, dependency=dependency)

    mfem_coeff1 = jitter(func)
    return NumbaCoefficient(mfem_coeff1)


def func_to_numba_coeff_vector(func, shape=(0,), complex=False,
                               params=None, dependency=None):
    from petram.mfem_config import numba_debug
    numba_debug = False if myid != 0 else numba_debug

    if dependency is None:
        dependency = tuple()

    dependency = [(x.mfem_numba_coeff if isinstance(x, NumbaCoefficient) else x)
                  for x in dependency]

    jitter = mfem.jit.vector(shape=shape, complex=complex, params=params,
                             debug=numba_debug, dependency=dependency)

    mfem_coeff1 = jitter(func)
    return NumbaCoefficient(mfem_coeff1)


def func_to_numba_coeff_matrix(func, shape=(0, 0), complex=False,
                               params=None, dependency=None):

    from petram.mfem_config import numba_debug
    numba_debug = False if myid != 0 else numba_debug

    if dependency is None:
        dependency = tuple()

    dependency = [(x.mfem_numba_coeff if isinstance(x, NumbaCoefficient) else x)
                  for x in dependency]

    jitter = mfem.jit.matrix(shape=shape, complex=complex, params=params,
                             debug=numba_debug, dependency=dependency)

    mfem_coeff1 = jitter(func)
    return NumbaCoefficient(mfem_coeff1)
