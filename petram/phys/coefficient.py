'''

   coefficient generation funcitons

'''
import abc
from abc import ABC, abstractmethod

import numpy as np

from petram.phys.phys_model  import PhysCoefficient
from petram.phys.phys_model  import VectorPhysCoefficient
from petram.phys.phys_model  import MatrixPhysCoefficient
from petram.phys.phys_model  import PhysConstant, PhysVectorConstant, PhysMatrixConstant
from petram.phys.phys_model  import Coefficient_Evaluator

from petram.helper.variables import NativeCoefficientGenBase

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

def MCoeff(dim, exprs, ind_vars, l, g, return_complex=False, **kwargs):
    if isinstance(exprs, str): exprs = [exprs]
    if isinstance(exprs, NativeCoefficientGenBase): exprs = [exprs]

    class Mcoeff_Base(object):
        def __init__(self,  conj=False, scale=1.0):
            self.conj = conj
            self.scale = scale

        def proc_value(self, val):
            val = val * self.scale
            if self.conj: val=np.conj(val)
            return val
            
    class MCoeff(MatrixPhysCoefficient, Mcoeff_Base):
       def __init___(self, sdim, exprs, ind_vars, l, g, conj=False, scale=1.0, **kwargs):
           Mcoeff_Base.__init__(self, conj=conj, scale=scale)           
           MatrixPhysCoefficient.__init__(self, sdim, exprs,  ind_vars, l, g, **kwargs)

       def EvalValue(self, x):
           val = super(MCoeff, self).EvalValue(x)
           val = self.proc_value(val)

           if np.iscomplexobj(val):
               if self.real:
                  return val.real
               else:
                  return val.imag
           elif not self.real:
               return val*0.0
           else:
               return val
           
    class MCoeffCC(Coefficient_Evaluator, Mcoeff_Base, PyComplexMatrixCoefficient):
        def __init__(self, c1, c2, exprs, ind_vars, l, g, conj=False, scale=1.0, **kwargs):
            Mcoeff_Base.__init__(self, conj=conj, scale=scale)            
            ## real is not used...
            Coefficient_Evaluator.__init__(self, exprs, ind_vars, l, g, real=True)
            PyComplexMatrixCoefficient.__init__(self, c1, c2)

        def Eval(self, K, T, ip):
            x = T.Transform(ip)
            val = Coefficient_Evaluator.EvalValue(self, x)
            return self.proc_value(val)            

    conj = kwargs.get('conj', False)
    real = kwargs.get('real', True)
    scale = kwargs.get('scale', 1.0)

    #print("matrix exprs", exprs)    

    if any([isinstance(ee, str) for ee in exprs]):
        if return_complex:
            kwargs['real'] = True
            c1 = MCoeff(dim, exprs, ind_vars, l, g, **kwargs)
            kwargs['real'] = False
            c2 = MCoeff(dim, exprs, ind_vars, l, g, **kwargs)
            return MCoeffCC(c1, c2, dim, exprs, ind_vars, l, g, **kwargs)
        else:
            return MCoeff(dim, exprs, ind_vars, l, g, **kwargs)
    else:
        e = exprs

        if isinstance(e[0], NativeCoefficientGenBase):        
            if return_complex:
                c1 = call_nativegen(e[0], l, g, True,  conj, scale)
                c2 = call_nativegen(v, l, g, False, conj, scale)                
                return complex_coefficient_from_real_and_imag(c1, c2)
            else:
                return call_nativegen(e[0], l, g, real, conj, scale)                                 

        e = np.array(e, copy=False).reshape(dim, dim)
        e = e * scale
        if conj: e = np.conj(e)
        
        if return_complex:
            e = e.astype(complex)
            return PyComplexMatrixConstant(e)
        else:
            if np.iscomplexobj(e):
                if real:  e = e.real
                else: e = e.imag
            elif not real:
                e = np.array(e*0.0, dtype=float, copy=False)           
            else:
                e = np.array(e, dtype=float, copy=False)

            return PhysMatrixConstant(e)
     
def DCoeff(dim, exprs, ind_vars, l, g, **kwargs):
    if isinstance(exprs, str): exprs = [exprs]
    if isinstance(exprs, NativeCoefficientGenBase): exprs = [exprs]
    
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
           if self.conj: val=np.conj(val)

           if np.iscomplexobj(val):
               if self.real:
                  return val.real
               else:
                  return val.imag
           elif not self.real:
               return val*0.0
           else:
               return val
           
    conj = kwargs.get('conj', False)
    real = kwargs.get('real', True)
    scale = kwargs.get('scale', 1.0)

    #print("matrix exprs", exprs)
    
    if any([isinstance(ee, str) for ee in exprs]):
        return DCoeff(dim, exprs, ind_vars, l, g, **kwargs)
    else:
        e = exprs        
        
        if isinstance(e[0], NativeCoefficientGenBase):
            return call_nativegen(e[0], l, g, real, conj, scale)
        
        e = e * scale
        e = np.diag(e)               
        if np.iscomplexobj(e):
            if conj:  e = np.conj(e)
            if real:  e = e.real
            else: e = e.imag
        elif not real:
            e = np.array(e*0.0, dtype=float, copy=False)           
        else:
            e = np.array(e, dtype=float, copy=False)
        return PhysMatrixConstant(e)
     
def VCoeff(dim, exprs, ind_vars, l, g, **kwargs):
    if isinstance(exprs, str): exprs = [exprs]
    if isinstance(exprs, NativeCoefficientGenBase): exprs = [exprs]
    
    class VCoeff(VectorPhysCoefficient):
       def __init__(self, *args, **kwargs):
           #print("VCoeff, args", args[:2])
           self.conj = kwargs.pop('conj', False)
           self.scale = kwargs.pop('scale', 1.0)           
           super(VCoeff, self).__init__(*args, **kwargs)
           
       def EvalValue(self, x):
           val = super(VCoeff, self).EvalValue(x)
           val = val * self.scale        
           if self.conj: val=np.conj(val)
           
           if np.iscomplexobj(val):
               if self.real:
                  return val.real
               else:
                  return val.imag
           elif not self.real:
               return val*0.0
           else:
               return val
           
    conj = kwargs.get('conj', False)
    real = kwargs.get('real', True)
    scale = kwargs.get('scale', 1.0)

    #print("vector exprs", exprs)
    
    if any([isinstance(ee, str) for ee in exprs]):
        return VCoeff(dim, exprs, ind_vars, l, g, **kwargs)
    else:
        e = exprs        
        
        if isinstance(e[0], NativeCoefficientGenBase):
            return call_nativegen(e[0], l, g, real, conj, scale)                      

        e = np.array(e, copy=False)
        e = e * scale        
        if np.iscomplexobj(e):
            if conj:  e = np.conj(e)
            if real:  e = e.real
            else: e = e.imag
        elif not real:
            e = np.array(e*0.0, dtype=float, copy=False)           
        else:
            e = np.array(e, dtype=float, copy=False)
        return PhysVectorConstant(e)
     
def SCoeff(exprs, ind_vars, l, g, return_complex=False, **kwargs):
    if isinstance(exprs, str): exprs = [exprs]
    if isinstance(exprs, NativeCoefficientGenBase): exprs = [exprs]

    class Scoeff_Base(object):
        def __init__(self,  component=None, conj=False, scale=1.0):
            self.component = component
            self.conj = conj
            self.scale = scale

        def proc_value(self, val):
            if self.component is None:
                if self.conj: val=np.conj(val)
                v =  val
            else:
                if len(val.shape) == 0: val = [val]
                if self.conj:
                    v=np.conj(val)[self.component]
                else:
                    v =  val[self.component]
            v = v * self.scale
            return v
          
    class SCoeff(PhysCoefficient, Scoeff_Base):
        def __init__(self, exprs, ind_vars, l, g, component=None, conj=False, scale=1.0, **kwargs):
            Scoeff_Base.__init__(self, component=component, conj=conj, scale=scale)
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

    class SCoeffCC(Coefficient_Evaluator, Scoeff_Base, PyComplexCoefficient):
        def __init__(self, c1, c2, exprs, ind_vars, l, g, component=None, conj=False, scale=1.0, **kwargs):
            Scoeff_Base.__init__(self, component=component, conj=conj, scale=scale)            
            ## real is not used...
            Coefficient_Evaluator.__init__(self, exprs, ind_vars, l, g, real=True)
            PyComplexCoefficient.__init__(self, c1, c2)
            
        def Eval(self, T, ip):
            x = T.Transform(ip)
            val = Coefficient_Evaluator.EvalValue(self, x)
            if len(self.co) == 1 and len(val) == 1:
                val = val[0]
            return self.proc_value(val)
            
               
    component = kwargs.get('component', None)
    conj = kwargs.get('conj', False)
    real = kwargs.get('real', True)
    scale = kwargs.get('scale', 1.0)

    #print("scalar exprs", exprs)    

    if any([isinstance(ee, str) for ee in exprs]):
        if return_complex:
            kwargs['real'] = True
            c1 = SCoeff(exprs, ind_vars, l, g, **kwargs)            
            kwargs['real'] = False
            c2 = SCoeff(exprs, ind_vars, l, g, **kwargs)            
            return SCoeffCC(c1, c2, exprs, ind_vars, l, g, **kwargs)
        else:
            return SCoeff(exprs, ind_vars, l, g, **kwargs)
    else:
        # conj is ignored..(this doesn't no meaning...)
        #print("exprs",exprs)
        if component is None:
            v = exprs[0]         ## exprs[0]
        else:
            v = exprs[component] ## weakform10 didn't work with-> exprs[0][component]
          
        if isinstance(v, NativeCoefficientGenBase):
            if return_complex:
                c1 = call_nativegen(v, l, g, True,  conj, scale)
                c2 = call_nativegen(v, l, g, False, conj, scale)                
                return complex_coefficient_from_real_and_imag(c1, c2)
            else:
                return call_nativegen(v, l, g, real, conj, scale)                                 
                 
        v = v * scale

        if return_complex:
            v = complex(v)
            if conj:  v = np.conj(v)
            return PyComplexConstant(v)
        else:
            if np.iscomplexobj(v):
                if conj:  v = np.conj(v)
                if real:  v = v.real
                else: v = v.imag
            elif not real:
                v = 0.0
            else:
                pass
            v =  float(v)
            return PhysConstant(v)

'''

   Complex Coefficient 

    Handle Complex Coefficint as a single coefficint
    These classes are derived from RealImagCoefficientGen.

'''

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem


class RealImagCoefficientGen(ABC):
   # abstract
   @abstractmethod
   def __init__(self):
       pass
    
   @abstractmethod
   def get_real_coefficient(self):
       pass
   
   @abstractmethod      
   def get_imag_coefficient(self):
       pass
   
   @abstractmethod   
   def get_realimag_coefficient(self, real):
       pass              

class PyComplexConstantBase(RealImagCoefficientGen):
   def get_realimag_coefficient(self, real):
       if real:
           return self.get_real_coefficient()
       else:
           return self.get_imag_coefficient()
    
class PyComplexConstant(PyComplexConstantBase):
   def __init__(self, value):
       self.value = value
       
   def Eval(self, T, ip):
       return self.value
       
   def get_real_coefficient(self):
       return PhysConstant(self.value.real)
      
   def get_imag_coefficient(self):
       return PhysConstant(self.value.imag)
   
class PyComplexVectorConstant(PyComplexConstantBase):
   def __init__(self, value):
       self.value = value
       self.vdim = len(value)
    
   def Eval(self, V,  T, ip):
       return self.value
       
   def get_real_coefficient(self):
       return PhysVectorConstant(self.value.real)
      
   def get_imag_coefficient(self):
       return PhysVectorConstant(self.value.imag)

   
class PyComplexMatrixConstant(RealImagCoefficientGen):
   def __init__(self, value):
       self.value = value
       self.width = value.shape[1]
       self.height = value.shape[0]
       
   def Eval(self, K,  T, ip):
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
       
    
class PyComplexCoefficientBase(RealImagCoefficientGen):
   def __init__(self, coeff1, coeff2):
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
       
   @abstractmethod
   def Eval(self, *args):
       pass
   
class PyComplexCoefficient(PyComplexCoefficientBase):
   def Eval(self, T, ip):
       if self.coeff1 is not None:
           v = complex(self.coeff1.Eval(T, ip))
       else:
           v = 0.0
       if self.coeff2 is not None:
           v += 1j*self.coeff2.Eval(T, ip)
       return v
   
class PyComplexVectorCoefficient(PyComplexCoefficientBase):
   def __init__(self, coeff1, coeff2):
       self.coeff1 = coeff1
       self.coeff2 = coeff2
       if self.coeff1 is not None:
           self.vdim  = self.coeff1.GetVDim()
       elif self.coeff2 is not None:
           self.vdim  = self.coeff1.GetVDim()
       else:
           assert False, "Either Real or Imag should be non zero"
   
   def Eval(self, V, T, ip):
       if self.coeff1 is not None:       
           self.coeff1.Eval(V, T, ip)
       else:
           V.Assign(0.0)
       M = V.GetDataArray().astype(complex)
       if self.coeff2 is not None:
           self.coeff2.Eval(V, T, ip)
           M += 1j*V.GetDataArray()           
       return M
   
class PyComplexMatrixCoefficient(PyComplexCoefficientBase):
   def __init__(self, coeff1, coeff2):
       self.coeff1 = coeff1
       self.coeff2 = coeff2
       if self.coeff1 is not None:
           self.hight = self.coeff1.GetHight()
           self.width = self.coeff1.GetWidth()           
       elif self.coeff2 is not None:
           self.hight = self.coeff1.GetHight()
           self.width = self.coeff1.GetWidth()           
       else:
           assert False, "Either Real or Imag should be non zero"
   
   def Eval(self, K, T, ip):
       if self.coeff1 is not None:
           self.coeff1.Eval(K, T, ip)
       else:
           K.Assign(0.0)
       M = K.GetDataArray().astype(complex)
       if self.coeff2 is not None:
           self.coeff2.Eval(K, T, ip)
           M += 1j*K.GetDataArray()           
       return M

def complex_coefficient_from_real_and_imag(coeffr, coeffi):
    if isinstance(mfem.MatrixPyCoefficient):
        return PyComplexMatrixCoefficient(coeffr, coeffi)
    elif isinstance(mfem.VectorPyCoefficient):
        return PyComplexVectorCoefficient(coeffr, coeffi)
    elif isinstance(mfem.PyCoefficient):        
        return PyComplexCoefficient(coeffr, coeffi)
    
class PyRealCoefficient(mfem.PyCoefficient):
   def __init__(self, coeff):
       self.coeff = coeff
       mfem.PyCoefficient.__init__(self)
       
   def Eval(self, T, ip):
       v = self.coeff.Eval(T, ip)
       return v.real

class PyImagCoefficient(mfem.PyCoefficient):
   def __init__(self, coeff):
       self.coeff = coeff
       mfem.PyCoefficient.__init__(self)       

   def Eval(self, T, ip):
       v = self.coeff.Eval(T, ip)
       return v.imag

class PyRealVectorCoefficient(mfem.VectorPyCoefficient):
   def __init__(self, coeff):
       self.coeff = coeff
       mfem.VectorPyCoefficient.__init__(self, coeff.vdim)
       
   def Eval(self, K,  T, ip):
       M = self.coeff.Eval(K, T, ip)
       return K.Assign(M.real)

class PyImagVectorCoefficient(mfem.VectorPyCoefficient):
   def __init__(self, coeff):
       self.coeff = coeff
       mfem.VectorPyCoefficient.__init__(self, coeff.vdim)
       
   def Eval(self, K, T, ip):
       M = self.coeff.Eval(K, T, ip)
       return K.Assign(M.imag)
   
class PyRealMatrixCoefficient(mfem.MatrixPyCoefficient):
   def __init__(self, coeff):
       self.coeff = coeff
       assert coeff.height == coeff.width, "not supported"
       mfem.MatrixPyCoefficient.__init__(self, coeff.height)
       
   def Eval(self, K,  T, ip):
       M = self.coeff.Eval(K, T, ip)
       return K.Assign(M.real)

class PyImagMatrixCoefficient(mfem.MatrixPyCoefficient):
   def __init__(self, coeff):
       self.coeff = coeff
       assert coeff.height == coeff.width, "not supported"       
       mfem.MatrixPyCoefficient.__init__(self, coeff.height)
       
   def Eval(self, K, T, ip):
       M = self.coeff.Eval(K, T, ip)
       return K.Assign(M.imag)
   
## CC (ComplexCoefficient)   
class CCBase(RealImagCoefficientGen):
   def __init__(self, coeff):
       self.coeff = coeff

   @abc.abstractproperty
   def kind(self):
       pass
   
   @abstractmethod      
   def Eval(self, *args, **kwargs):
       pass
   
   def get_real_coefficient(self):
       if self.kind == 'scalar':
          return PyRealCoefficient(self)
       elif self.kind == 'vector':
          return PyRealVectorCoefficient(self)           
       elif self.kind == 'matrix':
          return PyRealMatrixCoefficient(self)
      
   def get_imag_coefficient(self):
       if self.kind == 'scalar':
          return PyImagCoefficient(self)
       elif self.kind == 'vector':
          return PyImagVectorCoefficient(self)           
       elif self.kind == 'matrix':
          return PyImagMatrixCoefficient(self)

   def get_realimag_coefficient(self, real):
       if real:
           return self.get_real_coefficient()
       else:
           return self.get_imag_coefficient()
   @property
   def vdim(self):
       return self.coeff.vdim
   @property
   def width(self):
       return self.coeff.width
   @property
   def height(self):
       return self.coeff.height

class CC_Scalar(CCBase):
   @property
   def kind(self):
       return 'scalar'    
    
class CC_Matrix(CCBase):
   @property
   def kind(self):
       return 'matrix'
    
class CC_Vector(CCBase):
   @property
   def kind(self):
       return 'vector'
    
    
class PyComplexScalarInvCoefficient(CC_Scalar):
   def Eval(self, T, ip):
       v = self.coeff.Eval(T, ip)
       v = 1./v
       return v

class PyComplexProductCoefficient(CC_Scalar):
   def __init__(self, coeff, scale = 1.0):
       self.scale = scale
       CC_Scalar.__init__(self, coeff)
       
   def Eval(self, T, ip):
       v = self.coeff.Eval(T, ip)
       v *= self.scale
       return v

class PyComplexMatrixProductCoefficient(CC_Matrix):
   def __init__(self, coeff, scale = 1.0):
       self.scale = scale
       CC_Matrix.__init__(self, coeff)
       
   def Eval(self, K, T, ip):
       v = self.coeff.Eval(K, T, ip)
       v *= self.scale
       return v
   
class PyComplexMatrixInvCoefficient(CC_Matrix):
   def Eval(self, K, T, ip):
       M = self.coeff.Eval(K, T, ip)
       M = np.linalg.inv(M)       
       return M
   
class PyComplexMatrixSumCoefficient(CC_Matrix):
    def __init__(self, coeff1,  coeff2):
        CC_Matrix.__init__(self, coeff1)
        self.coeff2 = coeff2
    
    def Eval(self, K, T, ip):
        M1 = self.coeff.Eval(K, T, ip)
        M2 = self.coeff2.Eval(K, T, ip)        
        return M1 + M2
   
class PyComplexMatrixSliceScalarCoefficient(CC_Scalar):
    def __init__(self, coeff,  slice1, slice2):
        CC_Scalar.__init__(self, coeff)
        self.K = mfem.DenseMatrix(coeff.width, coeff.height)
        self.slice1 = slice1
        self.slice2 = slice2
        
    def Eval(self, T, ip):
        M = self.coeff.Eval(self.K, T, ip)
        return M[self.slice1, self.slice2]
    
class PyComplexMatrixSliceVectorCoefficient(CC_Vector):
    def __init__(self, coeff,  slice1, slice2):
        CC_Vector.__init__(self, coeff)
        self.K = mfem.DenseMatrix(coeff.width, coeff.height)
        self.slice1 = slice1
        self.slice2 = slice2
        if len(slice1) == 1 and len(slice2) > 1:
            self._vdim = len(slice2)
        elif len(slice2) == 1 and len(slice1) > 1:            
            self._vdim = len(slice1)
        else:
            assert False, "SliceVector output should be a vector"+str(slice1)+"/" + str(slice2)
            
    def Eval(self, V, T, ip):
        M = self.coeff.Eval(self.K, T, ip)
        return M[self.slice1, :][:, self.slice2].flatten()

    @property
    def vdim(self):
        return self._vdim
    
class PyComplexMatrixSliceMatrixCoefficient(CC_Matrix):
    def __init__(self, coeff,  slice1, slice2):
        CC_Matrix.__init__(self, coeff)
        self.slice1 = slice1
        self.slice2 = slice2
        self._height = len(slice1)
        self._width  = len(slice2)
        
    def Eval(self, K, T, ip):
        M = self.coeff.Eval(K, T, ip)
        return M[self.slice1, :][:, self.slice2]

    @property
    def width(self):
        return self._width
    @property
    def height(self):
        return self._height
    


def PyComplexMatrixSliceCoefficient(coeff, slice1, slice2):
    if len(slice1) == 1 and len(slice2) == 1:
        return PyComplexMatrixSliceScalarCoefficient(coeff, slice1[0], slice2[0])
    
    elif len(slice1) == 1:
        return PyComplexMatrixSliceVectorCoefficient(coeff, slice1, slice2)
    
    elif len(slice2) == 1:
        return PyComplexMatrixSliceVectorCoefficient(coeff, slice1, slice2)
    
    else:
        return PyComplexMatrixSliceMatrixCoefficient(coeff, slice1, slice2)
       
