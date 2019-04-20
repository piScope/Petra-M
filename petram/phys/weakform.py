'''

   Weakform : interface to use MFEM integrator

'''
import os
import numpy as np


from petram.phys.phys_model import Phys
from petram.phys.phys_model  import PhysCoefficient
from petram.phys.phys_model  import VectorPhysCoefficient
from petram.phys.phys_model  import MatrixPhysCoefficient
from petram.phys.phys_model  import PhysConstant, PhysVectorConstant, PhysMatrixConstant

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
    domains = [[x.strip() for x in l[1].split(',') if x.strip()!=''] for l in lines]
    ranges  = [[x.strip() for x in l[2].split(',') if x.strip()!=''] for l in lines]
    for trial, test in zip(domains, ranges):
        if len(test) == 0:
           for x in trial: test.append(x)
    def x(txt):
        xx = [x.strip() for x in txt.split(",")]
        return xx
     
    coeffs  = [x(l[3])  for l in lines]
    dims    = [[int(x.strip()[0]) for x in l[6].split(',')]  for l in lines]

    return zip(names, domains, ranges, coeffs, dims)
 
bilinintegs = get_integrators('BilinearOps')
linintegs = get_integrators('LinearOps')

def MCoeff(*args, **kwargs):
    class MCoeff(MatrixPhysCoefficient):
       def __init__(self, *args, **kwargs):
           self.conj = kwargs.pop('conj', False)
           super(MCoeff, self).__init__(*args, **kwargs)
       def EvalValue(self, x):
           val = super(MCoeff, self).EvalValue(x)
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

    e = args[1]
    if any([isinstance(ee, str) for ee in e]):
        return MCoeff(*args, **kwargs)
    else:
        conj = kwargs.get('conj', False)
        real = kwargs.get('real', True)
        if np.iscomplexobj(e):
            if conj:  e = np.conj(e)
            if real:  e = e.real
            else: e = e.imag
        elif not real:
            e = np.array(e*0.0, dtype=float, copy=False)           
        else:
            e = np.array(e, dtype=float, copy=False)
        return PhysMatrixConstant(e)
     
def DCoeff(*args, **kwargs):
    class DCoeff(MatrixPhysCoefficient):
       def __init__(self, *args, **kwargs):
           self.conj = kwargs.pop('conj', False)       
           self.space_dim = args[0]
           super(DCoeff, self).__init__(*args, **kwargs)

       def EvalValue(self, x):
           from petram.phys.phys_model import Coefficient_Evaluator
           val = Coefficient_Evaluator.EvalValue(self, x)
           val = np.diag(val)
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

    e = args[1]
    if any([isinstance(ee, str) for ee in e]):
        return DCoeff(*args, **kwargs)
    else:
        e = np.diag(e)       
        conj = kwargs.get('conj', False)
        real = kwargs.get('real', True)
        if np.iscomplexobj(e):
            if conj:  e = np.conj(e)
            if real:  e = e.real
            else: e = e.imag
        elif not real:
            e = np.array(e*0.0, dtype=float, copy=False)           
        else:
            e = np.array(e, dtype=float, copy=False)
        return PhysMatrixConstant(e)
     
def VCoeff(*args, **kwargs):    
    class VCoeff(VectorPhysCoefficient):
       def __init__(self, *args, **kwargs):
           #print("VCoeff, args", args[:2])
           self.conj = kwargs.pop('conj', False)
           super(VCoeff, self).__init__(*args, **kwargs)
       def EvalValue(self, x):
           val = super(VCoeff, self).EvalValue(x)
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
           
    e = args[1]
    if any([isinstance(ee, str) for ee in e]):
        return VCoeff(*args, **kwargs)
    else:
        conj = kwargs.get('conj', False)
        real = kwargs.get('real', True)
        if np.iscomplexobj(e):
            if conj:  e = np.conj(e)
            if real:  e = e.real
            else: e = e.imag
        elif not real:
            e = np.array(e*0.0, dtype=float, copy=False)           
        else:
            e = np.array(e, dtype=float, copy=False)
        return PhysVectorConstant(e)
     
def SCoeff(*args, **kwargs):
    class SCoeff(PhysCoefficient):
       def __init__(self, *args, **kwargs):
           #print("SCoeff, args", args[:1])       
           self.component = kwargs.pop('component', None)
           self.conj = kwargs.pop('conj', False)       
           super(SCoeff, self).__init__(*args, **kwargs)

       def EvalValue(self, x):
           val = super(SCoeff, self).EvalValue(x)
           if self.component is None:
               if self.conj: val=np.conj(val)
               v =  val
           else:
               if len(val.shape) == 0: val = [val]
               if self.conj: val=np.conj(val)[self.component]
               v =  val[self.component]
           if np.iscomplexobj(v):
               if self.real:
                  return v.real
               else:
                  return v.imag
           elif not self.real:
               return 0.0
           else:
               return v
               
    e = args[0]
    component = kwargs.get('component', None)
    conj = kwargs.get('conj', False)
    real = kwargs.get('real', True)
    if any([isinstance(ee, str) for ee in e]):
        return SCoeff(*args, **kwargs)
    else:
        # conj is ignored..(this doesn't no meaning...)       
        if component is None:
            v = args[0]
        else:
            v = args[0][component]
        if np.iscomplexobj(v):
            if conj:  v = np.conj(v)
            if real:  v = v.real
            else: v = v.imag
        elif not real:
            v = 0.0
        else:
            pass
        return PhysConstant(float(v))
    
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
                self.integrator,]
                #self.use_src_proj,
                #self.use_dst_proj,]
              
    def panel1_tip(self):
        pass

    def panel1_param(self):
        import wx       
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
               p2,]
              #["use src proj.",  self.use_src_proj,   3, {"text":""}],
              #["use dst proj.",  self.use_dst_proj,   3, {"text":""}],  ]
        return ll
     
    def is_complex(self):
        return self.get_root_phys().is_complex_valued
              
    def import_panel1_value(self, v):
        dep_vars = self.get_root_phys().dep_vars                    
        self.test_idx = dep_vars.index(str(v[0]))
        self.coeff_type = str(v[1])
        self.vt_coeff.import_panel_value(self, (v[2],))
        self.integrator =  str(v[3])
        #self.use_src_proj = v[4]
        #self.use_dst_proj = v[5]

    def preprocess_params(self, engine):
        self.vt_coeff.preprocess_params(self)
        super(WeakIntegration, self).preprocess_params(engine)
        
    def add_contribution(self, engine, a, real = True, is_trans=False, is_conj=False):
        c = self.vt_coeff.make_value_or_expression(self)
        if isinstance(c, str): c = [c]
        if real:       
            dprint1("Add "+self.integrator+ " contribution(real)" + str(self._sel_index), "c", c)
        else:
            dprint1("Add "+self.integrator+ " contribution(imag)" + str(self._sel_index), "c", c)

        cotype = self.coeff_type[0]

        if self.get_root_phys().vdim > 1:
            dim = self.get_root_phys().vdim
        else:
            el_name = self.get_root_phys().element
            dim = self.get_root_phys().geom_dim
            '''
            if el_name.startswith("ND"):
                dim = self.get_root_phys().geom_dim            
            elif el_name.startswith("RT"):
                dim = self.get_root_phys().geom_dim
            else:
                dim = 1  #H1 scalar (this case does not exist..)
            '''
        if cotype == 'S':
             for b in self.itg_choice():
                if b[0] == self.integrator: break
             if not "S*2" in b[3]:
                 c_coeff = SCoeff(c[0],  self.get_root_phys().ind_vars,
                              self._local_ns, self._global_ns,
                              real = real, conj=is_conj)
             else: # so far this is only for an elastic integrator 
                 c_coeff = (SCoeff(c[0],  self.get_root_phys().ind_vars,
                                   self._local_ns, self._global_ns,
                                   real = real, conj=is_conj, component=0),
                            SCoeff(c[0],  self.get_root_phys().ind_vars,
                                   self._local_ns, self._global_ns,
                                   real = real, conj=is_conj, component=1))
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
            #print "Bdry Integrator"
            adder = a.AddBoundaryIntegrator
        elif isinstance(self, Domain):
            #print "Domain Integrator"
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
        return self.test_idx == kfes
     
    def itg_choice(self):
        t = self.get_root_phys().fes_type
        if len(t)>2 and t[2] == "v":
           t = t[:3]
        return [b for b in linintegs if t in b[1]]
     
class WeakBilinIntegration(WeakIntegration):
    def itg_choice(self):
        t = self.get_root_phys().fes_type
        if len(t)>2 and t[2] == "v":
           t = t[:3]
        return [b for b in bilinintegs if t in b[2]]
   
    def attribute_set(self, v):
        v = super(WeakBilinIntegration, self).attribute_set(v)
        v['paired_var'] = None  #(phys_name, index)
        v['use_symmetric'] = False
        v['use_conj'] = False                        
        return v
     
    def get_panel1_value(self):
        if self.paired_var is not None:
            try:
                mfem_physroot = self.get_root_phys().parent
                var_s = mfem_physroot[self.paired_var[0]].dep_vars
                n  = var_s[self.paired_var[1]]
                p  = self.paired_var[0]
            except:
                self.paired_var = None
                
        if self.paired_var is None:
            n = self.get_root_phys().dep_vars[0]
            p = self.get_root_phys().name()
                
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
        import wx
        
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

        return (trialname == testname) and (self.test_idx == kfes)
     
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

        return loc
           
def add_delta_contribution(obj, engine, a, real = True, is_trans=False, is_conj=False):
    self = obj
    c = self.vt_coeff.make_value_or_expression(self)
    if isinstance(c, str): c = [c]
        
    if real:       
       dprint1("Add "+self.integrator+ " delta (real)" + str(self._sel_index), "c", c)
    else:
       dprint1("Add "+self.integrator+ " delta(imag)" + str(self._sel_index), "c", c)

    cotype = self.coeff_type[0]

    if self.get_root_phys().vdim > 1:
        dim = self.get_root_phys().vdim
    else:
        el_name = self.get_root_phys().element
        dim = self.get_root_phys().geom_dim
    sdim = self.get_root_phys().geom_dim        
 
    integrator = getattr(mfem, self.integrator)
    adder = a.AddDomainIntegrator
            
    for pos in self.pos_value:
        args = list(pos[:sdim])
        if cotype == 'S':
             for b in self.itg_choice():
                if b[0] == self.integrator: break
               
             if not "S*2" in b[3]:
                 if isinstance(c[0], str):
                      c_coeff = None                    
                      #c_coeff = SCoeff(c[0],  self.get_root_phys().ind_vars,
                      #                 self._local_ns, self._global_ns,
                      #                 real = real, conj=is_conj)
                      value = eval(c[0], self._global_ns, self._local_ns)
                      dprint1("time dependent delta", value)
                      args.append(float(value))
                 else:
                     c_coeff = None
                     if real:
                         args.append(float(np.array(c[0])[0].real))
                     else:
                         args.append(float(np.array(c[0])[0].imag))
                 if args[-1] != 0:
                     d = mfem.DeltaCoefficient(*args)
                     if c_coeff is not None:
                         assert False, "This option needs update of PyMFEM"
                         d2 = mfem.ProductCoefficient(c_coeff, d)
                         d2._linked_c = (c_coeff, d)
                         adder(integrator(d2))                         
                     else:
                         adder(integrator(d))
                     
             else: # so far this is only for an elastic integrator
                 if real:
                     args.append(float(np.array(c[0])[0].real))
                 else:
                     args.append(float(np.array(c[0])[0].imag))
                 d1 = mfem.DeltaCoefficient(*args)
                 if real:
                     args.append(float(np.array(c[0])[1].real))
                 else:
                     args.append(float(np.array(c[0])[1].imag))
                 d2 = mfem.DeltaCoefficient(*args)
                 adder(integrator(d1, d2))                 

        elif cotype == 'V':
            if real:
                direction = np.array(c[0]).real
            else:
                direction = np.array(c[0]).imag              
            args.append(1.0)
            dir = mfem.Vector(direction)
            d = mfem.VectorDeltaCoefficient(dir, *args)
            adder(integrator(d))            
        else:
            assert False, "M and D are not supported for delta coefficient"
        
def validate_sel(value, obj, w):
    g = obj._global_ns
    try:
        value = eval(value, g)
        pos = np.atleast_2d(value)
        return True
    except:
        return False
   
class WeakLinDeltaIntegration(WeakLinIntegration): 
    def panel2_param(self):
        return [["Position",  "[[0,0,0],]",  0, {'changing_event': True,
                                                'setfocus_event': True,
                                                 'validator': validate_pos,
                                                 'validator_param':self}] ]
    def import_panel2_value(self, v):
        self.sel_index = ['all']
        self.sel_index_txt = '["all"]'                
        self.pos_txt = v[0]
        g = self._global_ns
        try:
            value = eval(v[0], g)
            self.pos_value = np.atleast_2d(value)
        except:
            import traceback
            traceback.print_exc()
            pass
         
    def attribute_set(self, v):
        v = super(WeakLinDeltaIntegration, self).attribute_set(v)
        v['pos_txt'] = "[[ 0,0,0,],]"
        v['pos_value'] = np.array([[ 0,0, 0,],])
        return v
     
    def get_panel2_value(self):
        return (self.pos_txt,)
   
    def add_contribution(self, engine, a, real = True, is_trans=False, is_conj=False):
        g = self._global_ns
        try:
            value = eval(self.pos_txt, g)
            self.pos_value = np.atleast_2d(value)
        except:
            import traceback
            traceback.print_exc()
            raise
        add_delta_contribution(self, engine, a, real = real, is_trans=is_trans,
                               is_conj = is_conj)
        
class WeakBilinDeltaIntegration(WeakBilinIntegration):
    def panel2_param(self):
        return [["Position",  "[[0,0,0],]",  0, {'changing_event': True,
                                                 'setfocus_event': True,
                                                 'validator': validate_pos,
                                                 'validator_param':self}] ]
    def import_panel2_value(self, v):
        self.sel_index = ['all']
        self.sel_index_txt = '["all"]'        
        self.pos_txt = v[0]
        g = self._global_ns
        try:
            value = eval(v[0], g)
            self.pos_value = np.atleast_2d(value)
        except:
            import traceback
            traceback.print_exc()

    def attribute_set(self, v):
        v = super(WeakBilinDeltaIntegration, self).attribute_set(v)
        v['pos_txt'] = "[[ 0,0,0,],]"
        v['pos_value'] = np.array([[ 0,0, 0,],])
        return v
     
    def get_panel2_value(self):
        return (self.pos_txt,)
   
    def add_contribution(self, engine, a, real = True, is_trans=False, is_conj=False):
        g = self._global_ns
        try:
            value = eval(self.pos_txt, g)
            self.pos_value = np.atleast_2d(value)
        except:
            import traceback
            traceback.print_exc()
            raise
        add_delta_contribution(self, engine, a, real = real, is_trans=is_trans,
                               is_conj = is_conj)
              
