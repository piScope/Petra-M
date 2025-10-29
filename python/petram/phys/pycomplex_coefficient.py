'''

   Complex Coefficient

    Handle Complex Coefficint as a single coefficint
    These classes are derived from RealImagCoefficientGen.

'''
import abc
from abc import ABC, abstractmethod

from numpy.linalg import inv, det

from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem


from petram.phys.phys_model import (PhysCoefficient,
                                    VectorPhysCoefficient,
                                    MatrixPhysCoefficient,
                                    PhysConstant,
                                    PhysVectorConstant,
                                    PhysMatrixConstant,
                                    Coefficient_Evaluator)

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('PyComplexCoefficient')


'''
  Real/Imag Coefficient (mfem coefficients)

'''


class PyRealCoefficient(mfem.PyCoefficient):
    def __init__(self, coeff):
        self.coeff = coeff
        mfem.PyCoefficient.__init__(self)

    def Eval(self, T, ip):
        v = self.coeff.eval(T, ip)
        return v.real


class PyImagCoefficient(mfem.PyCoefficient):
    def __init__(self, coeff):
        self.coeff = coeff
        mfem.PyCoefficient.__init__(self)

    def Eval(self, T, ip):
        v = self.coeff.eval(T, ip)
        return v.imag


class PyRealVectorCoefficient(mfem.VectorPyCoefficient):
    def __init__(self, coeff):
        self.coeff = coeff
        mfem.VectorPyCoefficient.__init__(self, coeff.vdim)

    def Eval(self, K, T, ip):
        M = self.coeff.eval(T, ip)
        K.SetSize(M.shape[0])
        return K.Assign(M.real)


class PyImagVectorCoefficient(mfem.VectorPyCoefficient):
    def __init__(self, coeff):
        self.coeff = coeff
        mfem.VectorPyCoefficient.__init__(self, coeff.vdim)

    def Eval(self, K, T, ip):
        M = self.coeff.eval(T, ip)
        K.SetSize(M.shape[0])
        return K.Assign(M.imag)


class PyRealMatrixCoefficient(mfem.MatrixPyCoefficient):
    def __init__(self, coeff):
        self.coeff = coeff
        assert coeff.height == coeff.width, "not supported"
        mfem.MatrixPyCoefficient.__init__(self, coeff.height)

    def Eval(self, K, T, ip):
        M = self.coeff.eval(T, ip)
        K.SetSize(M.shape[0], M.shape[1])
        return K.Assign(M.real)


class PyImagMatrixCoefficient(mfem.MatrixPyCoefficient):
    def __init__(self, coeff):
        self.coeff = coeff
        assert coeff.height == coeff.width, "not supported"
        mfem.MatrixPyCoefficient.__init__(self, coeff.height)

    def Eval(self, K, T, ip):
        M = self.coeff.eval(T, ip)
        K.SetSize(M.shape[0], M.shape[1])
        return K.Assign(M.imag)


'''

  Complex Coefficient.

'''


class RealImagCoefficientGen(ABC):
    # abstract
    def __init__(self, kind, vdim=None, width=None, height=None):
        self._kind = kind
        if vdim is not None:
            self._vdim = vdim
        if width is not None:
            self._width = width
        if height is not None:
            self._height = height

    @abstractmethod
    def get_real_coefficient(self):
        pass

    @abstractmethod
    def get_imag_coefficient(self):
        pass

    @abstractmethod
    def get_realimag_coefficient(self, real):
        pass

    @abstractmethod
    def eval(self, T, ip):
        pass

    @property
    def kind(self):
        return self._kind

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def vdim(self):
        return self._vdim

    def is_matrix(self):
        return self.kind == 'matrix'

    def is_vector(self):
        return self.kind == 'vector'

    def __pow__(self, exponent):
        assert self.kind == 'scalar', "Pow is defined only for scalar"

        class PyComplexPowCoefficient(CC_Scalar):
            def __init__(self, coeff, pow):
                self.pow = pow
                CC_Scalar.__init__(self)
                self.coeff = coeff

            def eval(self, T, ip):
                v = self.coeff.eval(T, ip)
                v = (v)**(self.pow)
                return v
        return PyComplexPowCoefficient(self, exponent)

    def __add__(self, other):
        assert self.kind == self.kind, "__add__ must add the same coefficient types"

        class PyComplexSumSCoefficient(CC_Scalar):
            def __init__(self, coeff1, coeff2):
                CC_Scalar.__init__(self)
                self.coeff = coeff1
                self.coeff2 = coeff2

            def eval(self, T, ip):
                v1 = self.coeff.eval(T, ip)
                v2 = self.coeff2.eval(T, ip)
                return v1 + v2

        class PyComplexSumVCoefficient(CC_Vector):
            def __init__(self, coeff1, coeff2):
                CC_Vector.__init__(self, coeff1.vdim)
                self.coeff = coeff1
                self.coeff2 = coeff2

            def eval(self, T, ip):
                v1 = self.coeff.eval(T, ip)
                v2 = self.coeff2.eval(T, ip)
                return v1 + v2

        class PyComplexSumMCoefficient(CC_Matrix):
            def __init__(self, coeff1, coeff2):
                self.coeff = coeff1
                self.coeff2 = coeff2
                CC_Matrix.__init__(self, coeff1.height, coeff1.width)

            def eval(self, T, ip):
                v1 = self.coeff.eval(T, ip)
                v2 = self.coeff2.eval(T, ip)
                return v1 + v2
        if self.kind == "scalar":
            return PyComplexSumSCoefficient(self, other)
        if self.kind == "vector":
            return PyComplexSumVCoefficient(self, other)
        if self.kind == "matrix":
            return PyComplexSumMCoefficient(self, other)

    def __radd__(self, other):
        from petram.mfem_config import numba_debug
        from petram.phys.numba_coefficient import NumbaCoefficient

        if isinstance(other, NumbaCoefficient):
            if numba_debug:
                dprint1(
                    "in __radd__: NumbaCoefficient is converted to PyComplexCoefficient")
            obj = complex_coefficient_from_real_and_imag(
                other.real, other.imag)
            return obj + self
        else:
            return NotImplemented

    def __mul__(self, scale):
        assert not isinstance(
            scale, RealImagCoefficientGen), "multiplication is not defined"

        class PyComplexProductXCoefficient(CC_Scalar):
            def __init__(self, coeff, scale=1.0):
                self.scale = scale
                self.coeff = coeff
                CC_Scalar.__init__(self)

            def eval(self, T, ip):
                v = self.coeff.eval(T, ip)
                v *= self.scale
                return v
        obj = PyComplexProductXCoefficient(self, scale)

    def __getitem__(self, arg):
        '''
           A{1]  --> arg is int
           A{[1, 2]]  --> arg is list
           A{[1, 2], [2,3]]  --> arg is tuple
        '''
        check = self.kind == 'matrix' or self.kind == 'vector'
        assert check, "slice is valid for vector and matrix"

        class PyComplexVectorSliceVectorCoefficient(CC_Vector):
            def __init__(self, coeff, slice1):
                CC_Vector.__init__(self, len(slice1))
                self.slice1 = slice1
                self.coeff = coeff

            def eval(self, T, ip):
                mat = self.coeff.eval(T, ip)
                return mat[self.slice1]

            @property
            def vdim(self):
                return self._vdim

        class PyComplexVectorSliceScalarCoefficient(CC_Scalar):
            def __init__(self, coeff, slice1):
                CC_Scalar.__init__(self)
                self.slice1 = slice1
                self.coeff = coeff

            def eval(self, T, ip):
                mat = self.coeff.eval(T, ip)
                return mat[self.slice1]

        class PyComplexMatrixSliceScalarCoefficient(CC_Scalar):
            def __init__(self, coeff, slice1, slice2):
                CC_Scalar.__init__(self)
                self.coeff = coeff
                self.slice1 = slice1
                self.slice2 = slice2

            def eval(self, T, ip):
                mat = self.coeff.eval(T, ip)
                return mat[self.slice1, self.slice2]

        class PyComplexMatrixSliceVectorCoefficient(CC_Vector):
            def __init__(self, coeff, slice1, slice2):
                self.slice1 = slice1
                self.slice2 = slice2
                if len(slice1) == 1 and len(slice2) > 1:
                    vdim = len(slice2)
                elif len(slice2) == 1 and len(slice1) > 1:
                    vdim = len(slice1)
                else:
                    assert False, "SliceVector output should be a vector" + \
                        str(slice1) + "/" + str(slice2)
                CC_Vector.__init__(self, vdim)
                self.coeff = coeff

            def eval(self, T, ip):
                mat = self.coeff.eval(T, ip)
                return mat[self.slice1, :][:, self.slice2].flatten()

            @property
            def vdim(self):
                return self._vdim

        class PyComplexMatrixSliceMatrixCoefficient(CC_Matrix):
            def __init__(self, coeff, slice1, slice2):

                self.slice1 = slice1
                self.slice2 = slice2
                CC_Matrix.__init__(self, len(slice1), len(slice2))
                self.coeff = coeff

            def eval(self, T, ip):
                mat = self.coeff.eval(T, ip)
                ret = mat[self.slice1, :][:, self.slice2]
                return ret

        if self.kind == "vector":
            slice1 = arg
            try:
                a = slice1[0]
            except:
                slice1 = (slice1, )

            if len(slice1) == 1:
                return PyComplexVectorSliceScalarCoefficient(self, slice1[0])
            elif len(slice1) > 1:
                return PyComplexVectorSliceVectorCoefficient(self, slice1)
            else:
                assert False, "slice size must be greater than 1"

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

            if len(slice1) == 1 and len(slice2) == 1:
                return PyComplexMatrixSliceScalarCoefficient(self, slice1[0], slice2[0])

            elif len(slice1) == 1 and len(slice2) > 1:
                return PyComplexMatrixSliceVectorCoefficient(self, slice1, slice2)

            elif len(slice2) == 1 and len(slice1) > 1:
                return PyComplexMatrixSliceVectorCoefficient(self, slice1, slice2)

            elif len(slice2) > 1 and len(slice1) > 1:
                return PyComplexMatrixSliceMatrixCoefficient(self, slice1, slice2)
            else:
                assert False, "slice size must be greater than 1"

    def inv(self):
        '''
        return inverse
        '''
        assert self.kind == 'matrix', "inverse is only valid for matrix"

        class PyComplexMatrixInvCoefficient(CC_Matrix):
            def __init__(self, coeff):
                CC_Matrix.__init__(self, coeff.height, coeff.width)
                self.coeff = coeff

            def eval(self, T, ip):
                mat = self.coeff.eval(T, ip)
                mat = inv(mat)
                return mat
        return PyComplexMatrixInvCoefficient(self)

    def adj(self):
        class PyComplexMatrixAdjCoefficient(CC_Matrix):
            def __init__(self, coeff):
                CC_Matrix.__init__(self, coeff.height, coeff.width)
                self.coeff = coeff

            def eval(self, T, ip):
                mat = self.coeff.eval(T, ip)
                determinant = det(mat)
                mat = inv(mat)
                return mat * determinant
        return PyComplexMatrixAdjCoefficient(self)

# constant coefficients


class PyComplexConstantBase(RealImagCoefficientGen):
    def get_realimag_coefficient(self, real):
        if real:
            return self.get_real_coefficient()
        else:
            return self.get_imag_coefficient()


class PyComplexConstant(PyComplexConstantBase):
    def __init__(self, value):
        self.value = value
        self._kind = "scalar"

    def eval(self, T, ip):
        return self.value

    def get_real_coefficient(self):
        return PhysConstant(self.value.real)

    def get_imag_coefficient(self):
        return PhysConstant(self.value.imag)

    def __pow__(self, exponent):
        return PyComplexConstant((self.value)**exponent)

    def __mul__(self, scale):
        return PyComplexConstant((self.value)*scale)

    def __add__(self, other):
        if isinstance(other, PyComplexConstant):
            return PyComplexConstant(self.value + other.value)
        return other.__add__(self)


class PyComplexVectorConstant(PyComplexConstantBase):
    def __init__(self, value):
        self._vdim = len(value)
        self._kind = "vector"

        vec1 = mfem.Vector(value.real.flatten())
        vec2 = mfem.Vector(value.imag.flatten())
        self.value = (vec1.GetDataArray() + 1j*vec2.GetDataArray()).flatten()

    def get_imag_coefficient(self):
        return PhysVectorConstant(self.value.imag)

    def eval(self, T, ip):
        return self.value

    def get_real_coefficient(self):
        return PhysVectorConstant(self.value.real)

    def get_imag_coefficient(self):
        return PhysVectorConstant(self.value.imag)

    def __mul__(self, scale):
        return PyComplexVectorConstant((self.value)*scale)

    def __add__(self, other):
        if isinstance(other, PyComplexVectorConstant):
            return PyComplexVectorConstant(self.value + other.value)
        return other.__add__(self)

    def __getitem__(self, slice1):
        vec = self.value
        try:
            _a = slice1[0]
        except:
            slice1 = (slice1, )

        if len(slice1) == 1:
            value = vec[slice1[0]]
            return PyComplexConstant(value)

        value = vec[slice1]
        return PyComplexVectorConstant(value.flatten())


class PyComplexMatrixConstant(RealImagCoefficientGen):
    def __init__(self, value):
        self._width = value.shape[1]
        self._height = value.shape[0]
        self._kind = "matrix"

        mat1 = mfem.DenseMatrix(value.real)
        mat2 = mfem.DenseMatrix(value.imag)
        self.value = mat1.GetDataArray() + 1j*mat2.GetDataArray()

    def eval(self, T, ip):
        return self.value

    def get_real_coefficient(self):
        return PhysMatrixConstant(self.value.real)

    def get_imag_coefficient(self):
        return PhysMatrixConstant(self.value.imag)

    def get_realimag_coefficient(self, real):
        if real:
            return self.get_real_coefficient()
        else:
            return self.get_imag_coefficient()

    def __mul__(self, scale):
        return PyComplexMatrixConstant((self.value)*scale)

    def __add__(self, other):
        if isinstance(other, PyComplexMatrixConstant):
            return PyComplexMatrixConstant(self.value + other.value)
        return other.__add__(self)

    def __getitem__(self, arg):
        mat = self.value

        slice1, slice2 = arg

        try:
            _a = slice1[0]
        except:
            slice1 = (slice1, )
        try:
            _a = slice2[0]
        except:
            slice2 = (slice2, )
        if len(slice1) == 1 and len(slice2) == 1:
            value = mat[slice1[0], slice2[0]]
            return PyComplexConstant(value)
        if len(slice1) > 1 and len(slice2) > 1:
            value = mat[slice1, :][:, slice2]
            return PyComplexMatrixConstant(value)

        value = mat[slice1, slice2]
        return PyComplexVectorConstant(value.flatten())

    def inv(self):
        mat = inv(self.value)
        return PyComplexMatrixConstant(mat)

    def adj(self):
        mat = self.value
        determinant = det(mat)
        mat = inv(mat)
        return PyComplexMatrixConstant(mat * determinant)


# ComplexCoefficientBase
#   when real and imaginary parts are given by know MFEM coefficient

class PyComplexCoefficientBase(RealImagCoefficientGen):
    def __init__(self, coeff1, coeff2):
        check1 = isinstance(coeff1,
                            (mfem.Coefficient,
                             mfem.VectorCoefficient,
                             mfem.MatrixCoefficient))
        check2 = isinstance(coeff2,
                            (mfem.Coefficient,
                             mfem.VectorCoefficient,
                             mfem.MatrixCoefficient)) or coeff2 is None

        assert check1, "coefficient must be mfem coefficient"
        assert check2, "coefficient must be mfem coefficient"

        self.coeff1 = coeff1
        self.coeff2 = coeff2

    def get_real_coefficient(self):
        return self.coeff1

    def get_imag_coefficient(self):
        return self.coeff2

    def get_realimag_coefficient(self, real):
        if real:
            return self.get_real_coefficient()
        else:
            return self.get_imag_coefficient()


class PyComplexCoefficient(PyComplexCoefficientBase):
    def __init__(self, coeff1, coeff2):
        PyComplexCoefficientBase.__init__(self, coeff1, coeff2)
        self._kind = "scalar"

    def eval(self, T, ip):
        if self.coeff1 is not None:
            vec = complex(self.coeff1.Eval(T, ip))
        else:
            vec = 0.0
        if self.coeff2 is not None:
            vec += 1j * self.coeff2.Eval(T, ip)
        return vec


class PyComplexVectorCoefficient(PyComplexCoefficientBase):
    def __init__(self, coeff1, coeff2):
        PyComplexCoefficientBase.__init__(self, coeff1, coeff2)
        self._kind = "vector"
        if self.coeff1 is not None:
            self._vdim = self.coeff1.GetVDim()
        elif self.coeff2 is not None:
            self._vdim = self.coeff1.GetVDim()
        else:
            assert False, "Either Real or Imag should be non zero"
        self._V = mfem.Vector(self._vdim)

    def eval(self, T, ip):
        if self.coeff1 is not None:
            self.coeff1.Eval(self._V, T, ip)
        else:
            self._V.Assign(0.0)
        mat = self._V.GetDataArray().astype(complex)
        if self.coeff2 is not None:
            self.coeff2.Eval(self._V, T, ip)
            mat += 1j * self._V.GetDataArray()
        return mat


class PyComplexMatrixCoefficient(PyComplexCoefficientBase):
    def __init__(self, coeff1, coeff2):
        PyComplexCoefficientBase.__init__(self, coeff1, coeff2)
        self._kind = "matrix"
        if self.coeff1 is not None:
            self._height = self.coeff1.GetHeight()
            self._width = self.coeff1.GetWidth()
        elif self.coeff2 is not None:
            self._height = self.coeff1.GetHeight()
            self._width = self.coeff1.GetWidth()
        else:
            assert False, "Either Real or Imag should be non zero"

        self._K = mfem.DenseMatrix(self._height, self._width)

    def eval(self, T, ip):
        if self.coeff1 is not None:
            self.coeff1.Eval(self._K, T, ip)
        else:
            self._K.Assign(0.0)
        mat = self._K.GetDataArray().astype(complex)
        if self.coeff2 is not None:
            self.coeff2.Eval(self._K, T, ip)
            mat += 1j * self._K.GetDataArray()
        return mat


def complex_coefficient_from_real_and_imag(coeffr, coeffi):
    if (isinstance(coeffr, mfem.MatrixCoefficient) or
            isinstance(coeffi, mfem.MatrixCoefficient)):
        return PyComplexMatrixCoefficient(coeffr, coeffi)

    elif (isinstance(coeffr, mfem.VectorCoefficient) or
          isinstance(coeffi, mfem.VectorCoefficient)):
        return PyComplexVectorCoefficient(coeffr, coeffi)

    elif (isinstance(coeffr, mfem.Coefficient) or
          isinstance(coeffi, mfem.Coefficient)):
        return PyComplexCoefficient(coeffr, coeffi)


# CCBase : for generic Python written complex coefficient
class CCBase(RealImagCoefficientGen):
    def __init__(self, kind, vdim=None, height=None, width=None):
        RealImagCoefficientGen.__init__(
            self, kind, vdim=vdim, height=height, width=width)

        from petram.debug import handle_allow_python_function_coefficient
        msg = "Python function coefficient is created (complex)"
        handle_allow_python_function_coefficient(msg)

    def get_real_coefficient(self):
        if self.kind == 'scalar':
            return PyRealCoefficient(self)
        if self.kind == 'vector':
            return PyRealVectorCoefficient(self)
        if self.kind == 'matrix':
            return PyRealMatrixCoefficient(self)
        assert False, "unsupported kind"
        return None

    def get_imag_coefficient(self):
        if self.kind == 'scalar':
            return PyImagCoefficient(self)
        if self.kind == 'vector':
            return PyImagVectorCoefficient(self)
        if self.kind == 'matrix':
            return PyImagMatrixCoefficient(self)
        assert False, "unsupported kind"
        return None

    def get_realimag_coefficient(self, real):
        if real:
            return self.get_real_coefficient()
        return self.get_imag_coefficient()


class CC_Scalar(CCBase):
    def __init__(self):
        CCBase.__init__(self, "scalar")


class CC_Vector(CCBase):
    def __init__(self, vdim):
        CCBase.__init__(self, "vector", vdim=vdim)


class CC_Matrix(CCBase):
    def __init__(self, height, width):
        CCBase.__init__(self, "matrix", height=height, width=width)
