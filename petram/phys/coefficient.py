'''

   coefficient generation funcitons

'''
import numpy as np

from petram.phys.phys_model  import PhysCoefficient
from petram.phys.phys_model  import VectorPhysCoefficient
from petram.phys.phys_model  import MatrixPhysCoefficient
from petram.phys.phys_model  import PhysConstant, PhysVectorConstant, PhysMatrixConstant

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

def MCoeff(dim, exprs, ind_vars, l, g, **kwargs):
    if isinstance(exprs, str): exprs = [exprs]
    if isinstance(exprs, NativeCoefficientGenBase): exprs = [exprs]
    
    class MCoeff(MatrixPhysCoefficient):
       def __init__(self, *args, **kwargs):
           self.conj = kwargs.pop('conj', False)
           self.scale = kwargs.pop('scale', 1.0)
           
           super(MCoeff, self).__init__(*args, **kwargs)
       def EvalValue(self, x):
           val = super(MCoeff, self).EvalValue(x)
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
        return MCoeff(dim, exprs, ind_vars, l, g, **kwargs)
    else:
        e = exprs

        if isinstance(e[0], NativeCoefficientGenBase):
            return call_nativegen(e[0], l, g, real, conj, scale)

        e = np.array(e, copy=False).reshape(dim, dim)
        e = e * scale
        if np.iscomplexobj(e):
            if conj:  e = np.conj(e)
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
     
def SCoeff(exprs, ind_vars, l, g, **kwargs):
    if isinstance(exprs, str): exprs = [exprs]
    if isinstance(exprs, NativeCoefficientGenBase): exprs = [exprs]
        
    class SCoeff(PhysCoefficient):
       def __init__(self, exprs, ind_vars, l, g, **kwargs):
           #print("SCoeff, args", args[:1])       
           self.component = kwargs.pop('component', None)
           self.conj = kwargs.pop('conj', False)
           self.scale = kwargs.pop('scale', 1.0)
           super(SCoeff, self).__init__(exprs, ind_vars, l, g, **kwargs)

       def EvalValue(self, x):
           val = super(SCoeff, self).EvalValue(x)
           if self.component is None:
               if self.conj: val=np.conj(val)
               v =  val
           else:
               if len(val.shape) == 0: val = [val]
               if self.conj: val=np.conj(val)[self.component]
               v =  val[self.component]
           v = v * self.scale
           if np.iscomplexobj(v):
               if self.real:
                  return v.real
               else:
                  return v.imag
           elif not self.real:
               return 0.0
           else:
               return v
               
    component = kwargs.get('component', None)
    conj = kwargs.get('conj', False)
    real = kwargs.get('real', True)
    scale = kwargs.get('scale', 1.0)

    #print("scalar exprs", exprs)    

    if any([isinstance(ee, str) for ee in exprs]):
        return SCoeff(exprs, ind_vars, l, g, **kwargs)
    else:
        # conj is ignored..(this doesn't no meaning...)
        #print("exprs",exprs)
        if component is None:
            v = exprs[0]         ## exprs[0]
        else:
            v = exprs[component] ## weakform10 didn't work with-> exprs[0][component]
          
        if isinstance(v, NativeCoefficientGenBase):
            return call_nativegen(v, l, g, real, conj, scale)                                 
                 
        v = v * scale 
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
