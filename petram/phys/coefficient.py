'''

   coefficient generation funcitons

'''
from petram.debug import handle_allow_python_function_coefficient
import numpy as np

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

from petram.helper.variables import (Variable,
                                     NativeCoefficientGenBase)

from petram.phys.pycomplex_coefficient import (CC_Matrix,
                                               CC_Vector,
                                               CC_Scalar,
                                               PyComplexConstant,
                                               PyComplexVectorConstant,
                                               PyComplexMatrixConstant,
                                               complex_coefficient_from_real_and_imag)

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('Coefficient')


def call_nativegen(v, l, g, real, conj, scale):
    vv = v(l, g)
    if real:
        coeff = vv[0]
        if scale != 1.0 and coeff is not None:
            coeff = v.scale_coeff(coeff, scale)

        return coeff
    else:
        if conj:
            assert False, "conj is not supported for NativeCoefficient"
        else:
            coeff = vv[1]
            if scale != 1.0 and coeff is not None:
                coeff = v.scale_coeff(coeff, scale)

            return coeff


def MCoeff(dim, exprs, ind_vars, l, g, return_complex=False,
           return_mfem_constant=False, **kwargs):
    if isinstance(exprs, str):
        exprs = [exprs]
    if isinstance(exprs, NativeCoefficientGenBase):
        exprs = [exprs]

    class MCoeff_Base(object):
        def __init__(self, conj=False, scale=1.0):
            self.conj = conj
            self.scale = scale

        def proc_value(self, val):
            val = val * self.scale
            if self.conj:
                val = np.conj(val)
            return val

    class MCoeff(MatrixPhysCoefficient, MCoeff_Base):
        def __init__(self, sdim, exprs, ind_vars, l, g,
                     scale=1.0, conj=False, **kwargs):
            MCoeff_Base.__init__(self, conj=conj, scale=scale)
            MatrixPhysCoefficient.__init__(
                self, sdim, exprs, ind_vars, l, g, **kwargs)

        def EvalValue(self, x):
            val = super(MCoeff, self).EvalValue(x)
            val = self.proc_value(val)

            if np.iscomplexobj(val):
                if self.real:
                    return val.real
                else:
                    return val.imag
            elif not self.real:
                return val * 0.0
            else:
                return val

    class MCoeffCC(Coefficient_Evaluator, MCoeff_Base, CC_Matrix):
        def __init__(self, dim, exprs, ind_vars, l,
                     g, conj=False, scale=1.0, **kwargs):
            MCoeff_Base.__init__(self, conj=conj, scale=scale)
            # real is not used...
            Coefficient_Evaluator.__init__(
                self, exprs, ind_vars, l, g, real=True)
            CC_Matrix.__init__(self, dim, dim)

        def eval(self, T, ip):
            for n, v in self.variables:
                v.set_point(T, ip, self.g, self.l)
            x = T.Transform(ip)
            val = Coefficient_Evaluator.EvalValue(self, x)
            val = val.reshape(self.height, self.width)
            return self.proc_value(val)

    conj = kwargs.get('conj', False)
    real = kwargs.get('real', True)
    scale = kwargs.get('scale', 1.0)

    if any([isinstance(ee, str) for ee in exprs]):
        # if it is one liner array expression. try mfem.jit
        from petram.phys.numba_coefficient import expr_to_numba_coeff

        coeff = expr_to_numba_coeff(exprs, mfem.jit.matrix,
                                    ind_vars, conj, scale, g, l,
                                    return_complex, shape=(dim, dim))
        if coeff is None:
            msg = "JIT is not possbile. Continuing with Python mode"
            handle_allow_python_function_coefficient(msg)
        else:
            if return_complex:
                return coeff
            else:
                if real:
                    return coeff.real
                else:
                    return coeff.imag

        if return_complex:
            return MCoeffCC(dim, exprs, ind_vars, l, g, **kwargs)
        else:
            return MCoeff(dim, exprs, ind_vars, l, g, **kwargs)
    else:
        e = exprs

        if isinstance(e[0], NativeCoefficientGenBase):
            if return_complex:
                c1 = call_nativegen(e[0], l, g, True, conj, scale)
                c2 = call_nativegen(e[0], l, g, False, conj, scale)
                return complex_coefficient_from_real_and_imag(c1, c2)
            else:
                return call_nativegen(e[0], l, g, real, conj, scale)

        e = np.array(e, copy=False).reshape(dim, dim)
        e = e * scale
        if conj:
            e = np.conj(e)

        if return_complex:
            assert not return_mfem_constant, "return_complex and return_mfem_constant can not be used togeter"
            e = e.astype(complex)
            return PyComplexMatrixConstant(e)
        else:
            if np.iscomplexobj(e):
                if real:
                    e = e.real
                else:
                    e = e.imag
            elif not real:
                e = np.array(e * 0.0, dtype=float, copy=False)
            else:
                e = np.array(e, dtype=float, copy=False)

            if return_mfem_constant:
                return mfem.MatrixConstantCoefficient(e)
            else:
                return PhysMatrixConstant(e)


def DCoeff(dim, exprs, ind_vars, l, g,
           return_mfem_constant=False, **kwargs):
    if isinstance(exprs, str):
        exprs = [exprs]
    if isinstance(exprs, NativeCoefficientGenBase):
        exprs = [exprs]

    class DCoeff(MatrixPhysCoefficient):
        def __init__(self, *args, **kwargs):
            self.conj = kwargs.pop('conj', False)
            self.scale = kwargs.pop('scale', 1.0)
            self.space_dim = args[0]
            super(DCoeff, self).__init__(*args, **kwargs)

        def EvalValue(self, x):
            from petram.phys.phys_model import Coefficient_Evaluator
            val = Coefficient_Evaluator.EvalValue(self, x)
            val = np.diag(val)
            val = val * self.scale
            if self.conj:
                val = np.conj(val)

            if np.iscomplexobj(val):
                if self.real:
                    return val.real
                else:
                    return val.imag
            elif not self.real:
                return val * 0.0
            else:
                return val

    conj = kwargs.get('conj', False)
    real = kwargs.get('real', True)
    scale = kwargs.get('scale', 1.0)

    #print("matrix exprs", exprs)

    if any([isinstance(ee, str) for ee in exprs]):
        from petram.mfem_config import allow_python_function_coefficient
        if allow_python_function_coefficient != "allow":
            print("JIT is not supported in DCoeff")
            assert False, "can not jit coefficient"
        return DCoeff(dim, exprs, ind_vars, l, g, **kwargs)
    else:
        e = exprs

        if isinstance(e[0], NativeCoefficientGenBase):
            return call_nativegen(e[0], l, g, real, conj, scale)

        e = e * scale
        e = np.diag(e)
        if np.iscomplexobj(e):
            if conj:
                e = np.conj(e)
            if real:
                e = e.real
            else:
                e = e.imag
        elif not real:
            e = np.array(e * 0.0, dtype=float, copy=False)
        else:
            e = np.array(e, dtype=float, copy=False)

        if return_mfem_constant:
            return mfem.MatrixConstantCoefficient(e)
        else:
            return PhysMatrixConstant(e)


def VCoeff(dim, exprs, ind_vars, l, g, return_complex=False,
           return_mfem_constant=False, **kwargs):
    if isinstance(exprs, str):
        exprs = [exprs]
    if isinstance(exprs, NativeCoefficientGenBase):
        exprs = [exprs]

    class Vcoeff_Base(object):
        def __init__(self, conj=False, scale=1.0):
            self.conj = conj
            self.scale = scale

        def proc_value(self, val):
            val = val * self.scale
            if self.conj:
                val = np.conj(val)
            return val

    class VCoeff(VectorPhysCoefficient, Vcoeff_Base):
        def __init__(self, dim, exprs, ind_vars, l, g,
                     conj=False, scale=1.0, **kwargs):
            Vcoeff_Base.__init__(self, conj=conj, scale=scale)
            VectorPhysCoefficient.__init__(
                self, dim, exprs, ind_vars, l, g, **kwargs)

        def EvalValue(self, x):
            val = super(VCoeff, self).EvalValue(x)
            val = self.proc_value(val)

            if np.iscomplexobj(val):
                if self.real:
                    return val.real
                else:
                    return val.imag
            elif not self.real:
                return val * 0.0
            else:
                return val

    class VCoeffCC(Coefficient_Evaluator, Vcoeff_Base, CC_Vector):
        def __init__(self, dim, exprs, ind_vars, l,
                     g, conj=False, scale=1.0, **kwargs):
            Vcoeff_Base.__init__(self, conj=conj, scale=scale)
            # real is not used...
            Coefficient_Evaluator.__init__(
                self, exprs, ind_vars, l, g, real=True)
            CC_Vector.__init__(self, dim)

        def eval(self, T, ip):
            for n, v in self.variables:
                v.set_point(T, ip, self.g, self.l)
            x = T.Transform(ip)
            val = Coefficient_Evaluator.EvalValue(self, x)
            return self.proc_value(val)

    conj = kwargs.get('conj', False)
    real = kwargs.get('real', True)
    scale = kwargs.get('scale', 1.0)

    #print("vector exprs", exprs)

    if any([isinstance(ee, str) for ee in exprs]):
        # if it is one liner array expression. try mfem.jit
        from petram.phys.numba_coefficient import expr_to_numba_coeff
        coeff = expr_to_numba_coeff(exprs, mfem.jit.vector,
                                    ind_vars, conj, scale, g, l,
                                    return_complex, shape=(dim, ))
        if coeff is None:
            msg = "JIT is not possbile. Continuing with Python mode"
            handle_allow_python_function_coefficient(msg)

        else:
            if return_complex:
                return coeff
            else:
                if real:
                    return coeff.real
                else:
                    return coeff.imag

        if return_complex:
            return VCoeffCC(dim, exprs, ind_vars, l, g, **kwargs)
        else:
            return VCoeff(dim, exprs, ind_vars, l, g, **kwargs)

    else:
        e = exprs

        if isinstance(e[0], NativeCoefficientGenBase):
            if return_complex:
                c1 = call_nativegen(e[0], l, g, True, conj, scale)
                c2 = call_nativegen(e[0], l, g, False, conj, scale)
                return complex_coefficient_from_real_and_imag(c1, c2)
            else:
                return call_nativegen(e[0], l, g, real, conj, scale)

        e = np.array(e, copy=False)
        e = e * scale

        if return_complex:
            assert not return_mfem_constant, "return_complex and return_mfem_constant can not be used togeter"
            e = e.astype(complex)
            return PyComplexVectorConstant(e)
        else:
            if np.iscomplexobj(e):
                if conj:
                    e = np.conj(e)
                if real:
                    e = e.real
                else:
                    e = e.imag
            elif not real:
                e = np.array(e * 0.0, dtype=float, copy=False)
            else:
                e = np.array(e, dtype=float, copy=False)
            if return_mfem_constant:
                return mfem.VectorConstantCoefficient(e)
            else:
                return PhysVectorConstant(e)


def SCoeff(exprs, ind_vars, l, g, return_complex=False,
           return_mfem_constant=False, **kwargs):
    if isinstance(exprs, str):
        exprs = [exprs]
    if isinstance(exprs, NativeCoefficientGenBase):
        exprs = [exprs]

    class Scoeff_Base(object):
        def __init__(self, component=None, conj=False, scale=1.0):
            self.component = component
            self.conj = conj
            self.scale = scale

        def proc_value(self, val):
            if self.component is None:
                if self.conj:
                    val = np.conj(val)
                v = val
            else:
                if len(val.shape) == 0:
                    val = [val]
                if self.conj:
                    v = np.conj(val)[self.component]
                else:
                    v = val[self.component]
            v = v * self.scale
            return v

    class SCoeff(PhysCoefficient, Scoeff_Base):
        def __init__(self, exprs, ind_vars, l, g, component=None,
                     conj=False, scale=1.0, **kwargs):
            Scoeff_Base.__init__(
                self,
                component=component,
                conj=conj,
                scale=scale)
            super(SCoeff, self).__init__(exprs, ind_vars, l, g, **kwargs)

        def EvalValue(self, x):
            val = super(SCoeff, self).EvalValue(x)
            v = self.proc_value(val)
            if np.iscomplexobj(v):
                if self.real:
                    return v.real
                else:
                    return v.imag
            elif not self.real:
                return 0.0
            else:
                return v

    class SCoeffCC(Coefficient_Evaluator, Scoeff_Base, CC_Scalar):
        def __init__(self, exprs, ind_vars, l, g,
                     component=None, conj=False, scale=1.0, **kwargs):
            Scoeff_Base.__init__(
                self,
                component=component,
                conj=conj,
                scale=scale)
            # real is not used...
            Coefficient_Evaluator.__init__(
                self, exprs, ind_vars, l, g, real=True)
            CC_Scalar.__init__(self)

        def eval(self, T, ip):
            for n, v in self.variables:
                v.set_point(T, ip, self.g, self.l)
            x = T.Transform(ip)
            val = Coefficient_Evaluator.EvalValue(self, x)
            if len(self.co) == 1 and len(val) == 1:
                val = val[0]
            return self.proc_value(val)

    component = kwargs.get('component', None)
    conj = kwargs.get('conj', False)
    real = kwargs.get('real', True)
    scale = kwargs.get('scale', 1.0)

    if any([isinstance(ee, str) for ee in exprs]):
        if len(exprs) == 1:
            # if it is one liner array expression. try mfem.jit
            from petram.phys.numba_coefficient import expr_to_numba_coeff
            coeff = expr_to_numba_coeff(exprs, mfem.jit.scalar,
                                        ind_vars, conj, scale, g, l, return_complex)
            if coeff is None:
                msg = "JIT is not possbile. Continuing with Python mode"
                handle_allow_python_function_coefficient(msg)
            else:
                if return_complex:
                    return coeff
                else:
                    if real:
                        return coeff.real
                    else:
                        return coeff.imag
        else:
            # should not come here
            assert False, "Scalar coefficient can not use mutliple expressions"

        if return_complex:
            return SCoeffCC(exprs, ind_vars, l, g, **kwargs)
        else:
            return SCoeff(exprs, ind_vars, l, g, **kwargs)
    else:
        # conj is ignored..(this doesn't no meaning...)
        # print("exprs",exprs)
        if component is None:
            v = exprs[0]  # exprs[0]
        else:
            # weakform10 didn't work with-> exprs[0][component]
            v = exprs[component]

        if isinstance(v, NativeCoefficientGenBase):
            if return_complex:
                c1 = call_nativegen(v, l, g, True, conj, scale)
                c2 = call_nativegen(v, l, g, False, conj, scale)
                return complex_coefficient_from_real_and_imag(c1, c2)
            else:
                return call_nativegen(v, l, g, real, conj, scale)

        v = v * scale

        if return_complex:
            v = complex(v)
            if conj:
                v = np.conj(v)
            assert not return_mfem_constant, "return_complex and return_mfem_constant can not be used togeter"
            return PyComplexConstant(v)
        else:
            if np.iscomplexobj(v):
                if conj:
                    v = np.conj(v)
                if real:
                    v = v.real
                else:
                    v = v.imag
            elif not real:
                v = 0.0
            else:
                pass
            v = float(v)
            if return_mfem_constant:
                return mfem.ConstantCoefficient(v)
            else:
                return PhysConstant(v)


def sum_coefficient(c_arr):
    '''
    return sum_coefficient made from list of coefficient
    '''
    if len(c_arr) == 0:
        return None
    if len(c_arr) == 1:
        return c_arr[0]
    kind = ''
    for c in c_arr:
        if isinstance(c, mfem.Coefficient):
            if kind != '' and kind != 's':
                assert False, "can not mix diffenrnt kind of coefficient"
            kind = 's'
        if isinstance(c, mfem.VectorCoefficient):
            if kind != '' and kind != 'v':
                assert False, "can not mix diffenrnt kind of coefficient"
            kind = 'v'
        if isinstance(c, mfem.MatrixCoefficient):
            if kind != '' and kind != 'm':
                assert False, "can not mix diffenrnt kind of coefficient"
            kind = 'm'

    if kind == 's':
        c = mfem.SumCoefficient(c_arr[0], c_arr[1])
        for cc in c_arr[2:]:
            c = mfem.SumCoefficient(c, cc)
    elif kind == 'v':
        c = mfem.VectorSumCoefficient(c_arr[0], c_arr[1])
        for cc in c_arr[2:]:
            c = mfem.VectorSumCoefficient(c, cc)
    elif kind == 'm':
        c = mfem.MatrixSumCoefficient(c_arr[0], c_arr[1])
        for cc in c_arr[2:]:
            c = mfem.MatrixSumCoefficient(c, cc)

    return c
