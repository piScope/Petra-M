'''

   Weakform : interface to use MFEM integrator

'''
import os
import numpy as np
import wx

from petram.phys.phys_model import Phys
from petram.phys.phys_model  import PhysCoefficient
from petram.phys.phys_model  import VectorPhysCoefficient
from petram.phys.phys_model  import MatrixPhysCoefficient
from petram.model import Domain, Bdry, Edge, Point, Pair

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('WeakForm')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem
   
from petram.phys.vtable import VtableElement, Vtable   

def get_integrators(filename):
    import petram
    fid = open(os.path.join(os.path.dirname(petram.__file__), 'data', filename), 'r')
    lines = fid.readlines()
    fid.close()
    
    lines = [l for l in lines if not l.startswith('#')]
    lines = [l.split("|")[1:] for l in lines]
    names = [l[0].strip() for l in lines]
    domains = [[x.strip() for x in l[1].split(',')] for l in lines]
    ranges  = [[x.strip() for x in l[2].split(',')] for l in lines]
    def x(txt):
        return [x for x in txt if x=="M" or x=="D" or x=="V" or x=="S"]    
    coeffs  = [x(l[3])  for l in lines]
    dims    = [[int(x.strip()[0]) for x in l[6].split(',')]  for l in lines]

    return zip(names, domains, ranges, coeffs, dims)
 
bilinintegs = get_integrators('BilinearOps')
linintegs = get_integrators('LinearOps')
 
class MCoeff(MatrixPhysCoefficient):
    def __init__(self, *args, **kwargs):
        self.conj = kwargs.pop('conj', False)
        super(MCoeff, self).__init__(*args, **kwargs)
    def EvalValue(self, x):
        val = super(MCoeff, self).EvalValue(x)
        if self.conj: val=np.conj(val)                
        return val
     
class DCoeff(MatrixPhysCoefficient):
    def __init__(self, *args, **kwargs):
        self.conj = kwargs.pop('conj', False)       
        self.space_dim = args[0]
        super(DCoeff, self).__init__(*args, **kwargs)
    def EvalValue(self, x):
        val = super(DCoeff, self).EvalValue(x)
        val = np.diag(val)
        if self.conj: val=np.conj(val)                
        return val.flatten()
    
class VCoeff(VectorPhysCoefficient):
    def __init__(self, *args, **kwargs):
        self.conj = kwargs.pop('conj', False)
        super(VCoeff, self).__init__(*args, **kwargs)
    def EvalValue(self, x):
        val = super(VCoeff, self).EvalValue(x)
        if self.conj: val=np.conj(val)        
        return val

class SCoeff(PhysCoefficient):
    def __init__(self, *args, **kwargs):
        self.conj = kwargs.pop('conj', False)       
        super(SCoeff, self).__init__(*args, **kwargs)
    def EvalValue(self, x):
        val = super(SCoeff, self).EvalValue(x)
        if self.conj: val=np.conj(val)
        return val
     
data = [("coeff_lambda", VtableElement("coeff_lambda", type='array',
         guilabel = "lambda", default = 0.0, tip = "coefficient",))]
     
class WeakIntegration(Phys):
    vt_coeff = Vtable(data)
    def attribute_set(self, v):
        v['use_src_proj'] = False
        v['use_dst_proj'] = False
        v['coeff_type'] = 'S'
        v['integrator'] = 'MassIntegrator'
        v['test_idx'] = 0     #(index)
        self.vt_coeff.attribute_set(v)
        v = Phys.attribute_set(self, v)
        return v
     
    def get_panel1_value(self):
        dep_vars = self.get_root_phys().dep_vars                    
        return [dep_vars[self.test_idx],
                self.coeff_type,
                self.vt_coeff.get_panel_value(self)[0],
                self.integrator,
                self.use_src_proj,
                self.use_dst_proj,]
              
    def panel1_tip(self):
        pass

    def panel1_param(self):
        p = ["coeff. type", "S", 4,
             {"style":wx.CB_READONLY, "choices": ["Scalar", "Vector", "Diagonal", "Matrix"]}]

        names = [x[0] for x in self.itg_choice()]
        p2 = ["integrator", names[0], 4,
              {"style":wx.CB_READONLY, "choices": names}]
        
        dep_vars = self.get_root_phys().dep_vars             
        panels = self.vt_coeff.panel_param(self)
        ll = [["test space (Rows)", dep_vars[0], 4,
               {"style":wx.CB_READONLY, "choices": dep_vars}],
               p,
               panels[0],
               p2,
              ["use src proj.",  self.use_src_proj,   3, {"text":""}],
              ["use dst proj.",  self.use_dst_proj,   3, {"text":""}],  ]
        return ll
     
    def is_complex(self):
        return self.get_root_phys().is_complex_valued
              
    def import_panel1_value(self, v):
        dep_vars = self.get_root_phys().dep_vars                    
        self.test_idx = dep_vars.index(str(v[0]))
        self.coeff_type = str(v[1])
        self.vt_coeff.import_panel_value(self, (v[2],))
        self.integrator =  str(v[3])
        self.use_src_proj = v[4]
        self.use_dst_proj = v[5]

    def preprocess_params(self, engine):
        self.vt_coeff.preprocess_params(self)
        super(WeakIntegration, self).preprocess_params(engine)
        
    def add_contribution(self, engine, a, real = True, is_trans=False, is_conj=False):
        c = self.vt_coeff.make_value_or_expression(self)    
        if real:       
            dprint1("Add "+self.integrator+ " contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add "+self.integrator+ " contribution(imag)" + str(self._sel_index))
        dprint1("c", c)
        dim = self.get_root_phys().geom_dim
        cotype = self.coeff_type[0]
        if cotype == 'S':
             c_coeff = SCoeff(c[0],  self.get_root_phys().ind_vars,
                              self._local_ns, self._global_ns,
                              real = real, conj=is_conj)
        elif cotype == 'V':
             c_coeff = VCoeff(dim, c[0],  self.get_root_phys().ind_vars,
                              self._local_ns, self._global_ns,
                              real = real, conj=is_conj)
        elif cotype == 'M':
             c_coeff = MCoeff(dim, c[0],  self.get_root_phys().ind_vars,
                              self._local_ns, self._global_ns,
                              real = real, conj=is_conj)
        elif cotype == 'D':
             c_coeff = DCoeff(dim, c[0],  self.get_root_phys().ind_vars,
                              self._local_ns, self._global_ns,
                              real = real, conj=is_conj)
        integrator = getattr(mfem, self.integrator)
        if isinstance(self, Bdry):
            adder = a.AddBdrIntegrator
        elif isinstance(self, Domain):
            adder = a.AddDomainIntegrator
        else:
            assert False, "this class is not supported in weakform"
        self.add_integrator(engine, 'c', c_coeff,
                            adder, integrator, transpose=is_trans)
        
    def add_bf_contribution(self, engine, a, real = True, kfes=0):
        self.add_contribution(engine, a, real = real)
    def add_lf_contribution(self, engine, b, real = True, kfes=0):
        self.add_contribution(engine, b, real = real)
    def add_mix_contribution2(self, engine, mbf, r, c,  is_trans, is_conj,
                              real = True):
        self.add_contribution(engine, mbf, real=real, is_trans=is_trans, is_conj=is_conj)
        
    def itg_choice(self):
        return []
     
class WeakLinIntegration(WeakIntegration):
    def has_lf_contribution(self, kfes):
        return True
    def itg_choice(self):
        return linintegs
     
class WeakBilinIntegration(WeakIntegration):
    def itg_choice(self):
        return bilinintegs
   
    def attribute_set(self, v):
        v = super(WeakBilinIntegration, self).attribute_set(v)
        v['paired_var'] = None  #(phys_name, index)
        v['use_symmetric'] = False
        v['use_conj'] = False                        
        return v
    def get_panel1_value(self):
        if self.paired_var is None:
            n = self.get_root_phys().dep_vars[0]
            p = self.get_root_phys().name()
        else:
            mfem_physroot = self.get_root_phys().parent
            var_s = mfem_physroot[self.paired_var[0]].dep_vars
            n  = var_s[self.paired_var[1]]
            p  = self.paired_var[0]

        var = n + " ("+p + ")"             
        v1 = [var]
        v2 = super(WeakBilinIntegration, self).get_panel1_value()
        v3 = [self.use_symmetric, self.use_conj]
        return v1 + v2 + v3
              
    def panel1_tip(self):
        pass
     
    def is_complex(self):
        return self.get_root_phys().is_complex_valued
              
    def import_panel1_value(self, v):
        mfem_physroot = self.get_root_phys().parent
        names, pnames, pindex = mfem_physroot.dependent_values()

        idx = names.index(str(v[0]).split("(")[0].strip())
        self.paired_var = (pnames[idx], pindex[idx])
        super(WeakBilinIntegration, self).import_panel1_value(v[1:-2])       
        self.use_symmetric = v[-2]
        self.use_conj = v[-1]        
        
    def panel1_param(self):
        mfem_physroot = self.get_root_phys().parent
        names, pnames, pindex = mfem_physroot.dependent_values()
        names = [n+" ("+p + ")" for n, p in zip(names, pnames)]
        
        ll1 = [["paired variable", "S", 4,
                {"style":wx.CB_READONLY, "choices": names}]]
        ll2 = super(WeakBilinIntegration, self).panel1_param()
        ll3 = [["make symmetric",  self.use_symmetric,   3, {"text":""}],  
               ["use  conjugate",  self.use_conj,   3, {"text":""}],  ]
        ll = ll1 + ll2 + ll3

        return ll
        
    def has_bf_contribution(self, kfes):
        mfem_physroot = self.get_root_phys().parent
        var_s = mfem_physroot[self.paired_var[0]].dep_vars
        trialname  = var_s[self.paired_var[1]]
        testname = self.get_root_phys().dep_vars[self.test_idx]

        return (trialname == testname)
     
    def has_mixed_contribution(self):
        mfem_physroot = self.get_root_phys().parent
        var_s = mfem_physroot[self.paired_var[0]].dep_vars
        trialname  = var_s[self.paired_var[1]]
        testname = self.get_root_phys().dep_vars[self.test_idx]

        return (trialname != testname)        
       
    def get_mixedbf_loc(self):
        mfem_physroot = self.get_root_phys().parent
        var_s = mfem_physroot[self.paired_var[0]].dep_vars
        trialname  = var_s[self.paired_var[1]]
        testname = self.get_root_phys().dep_vars[self.test_idx]

        loc = []
        is_trans = 1
        loc.append((testname, trialname, is_trans, 1))

        if self.use_symmetric and not self.use_conj:
             loc.append((testname, trialname, -1, -1))
        if self.use_symmetric and self.use_conj:
             loc.append((testname, trialname, -1, 1))
        print("mix_loc", loc)
        return loc
           
    

              
