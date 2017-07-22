import numpy as np
import parser
from petram.model import Model, Bdry, Domain
from petram.namespace_mixin import NS_mixin
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('Phys')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem

from petram.helper.variables import Variable, eval_code 

 
# not that PyCoefficient return only real number array
class PhysConstant(mfem.ConstantCoefficient):
    def __init__(self, value):
        self.value = value
        mfem.ConstantCoefficient.__init__(self, value)
        
    def __repr__(self):
        return self.__class__.__name__+"("+str(self.value)+")"
      
class PhysCoefficient(mfem.PyCoefficient):
    def __init__(self, expr, ind_vars, l, g, real=True):
       self.l = {}
       self.g = g
       for key in l.keys():
          self.g[key] = l[key]
       self.real = real
       self.variables = []

       if isinstance(expr, str):
           st = parser.expr(expr.strip())
           code= st.compile('<string>')
           names = code.co_names
           for n in names:
               if n in g and isinstance(g[n], Variable):
                   self.variables.append((n, g[n]))
           self.co = code
       else:
           self.co = expr
           
       #compile(f_name+'('+ind_vars+')', '<strign>', 'eval')
       # 'x, y, z' -> 'x', 'y', 'z'
       self.ind_vars = [x.strip() for x in ind_vars.split(',')]
       mfem.PyCoefficient.__init__(self)

    def __repr__(self):
        return self.__class__.__name__+"(PhysCoefficeint)"
        
    def Eval(self, T, ip):
        for n, v in self.variables:
           v.set_point(T, ip, self.g, self.l)
        return super(PhysCoefficient, self).Eval(T, ip)

    def EvalValue(self, x):
        # set x, y, z to local variable so that we can use in
        # the expression.
        for k, name in enumerate(self.ind_vars):
           self.l[name] = x[k]
        for n, v in self.variables:
           #self.l['v'] = v
           #self.l[n] = eval('v()', self.g, self.l)
           #self.l[n] = eval('v()', self.g, {'v': v})                      
           self.l[n] = v()
        return (eval_code(self.co, self.g, self.l))
             
class VectorPhysCoefficient(mfem.VectorPyCoefficient):
    def __init__(self, sdim, exprs, ind_vars, l, g, real=True):
        self.l = {}
        self.g = g
        for key in l.keys():
           self.g[key] = l[key]

        self.real = real
        self.variables = []

        self.co = []
        
        for expr in exprs:
           if isinstance(expr, str):
               st = parser.expr(expr.strip())
               code= st.compile('<string>')
               names = code.co_names
               for n in names:
                   if n in g and isinstance(g[n], Variable):
                       self.variables.append((n, g[n]))
               self.co.append(code)
           else:
               self.co.append(expr)

        # 'x, y, z' -> 'x', 'y', 'z'
        self.ind_vars = [x.strip() for x in ind_vars.split(',')]
        mfem.VectorPyCoefficient.__init__(self, sdim)
        self.exprs = exprs
    def __repr__(self):
        return self.__class__.__name__+"(VectorPhysCoefficeint)"
        
    def Eval(self, V, T, ip):
        for n, v in self.variables:
           v.set_point(T, ip, self.g, self.l)                      
        return super(VectorPhysCoefficient, self).Eval(V, T, ip)
       
    def EvalValue(self, x):
        for k, name in enumerate(self.ind_vars):
           self.l[name] = x[k]

        for n, v in self.variables:
            self.l[n] = v()           
        val = [eval_code(co, self.g, self.l) for co in self.co]
        return np.array(val, copy = False).flatten()

class MatrixPhysCoefficient(mfem.MatrixPyCoefficient):
    def __init__(self, sdim, exprs,  ind_vars, l, g, real=True):
        self.l = {}
        self.g = g
        for key in l.keys():
           self.g[key] = l[key]
        self.real = real
        self.variables = []

        self.co = []
        for expr in exprs:
           if isinstance(expr, str):
               st = parser.expr(expr.strip())
               code= st.compile('<string>')
               names = code.co_names
               for n in names:
                  if n in g and isinstance(g[n], Variable):
                       self.variables.append((n, g[n]))
               self.co.append(code)
           else:
               self.co.append(expr)
               
        # 'x, y, z' -> 'x', 'y', 'z'
        self.ind_vars = [x.strip() for x in ind_vars.split(',')]
        mfem.MatrixPyCoefficient.__init__(self, sdim)
        
    def __repr__(self):
        return self.__class__.__name__+"(MatrixPhysCoefficeint)"
        
    def Eval(self, K, T, ip):
        for n, v in self.variables:
           v.set_point(T, ip, self.g, self.l)           
        return super(MatrixPhysCoefficient, self).Eval(K, T, ip)

    def EvalValue(self, x):
        for k, name in enumerate(self.ind_vars):
           self.l[name] = x[k]
        for n, v in self.variables:           
           self.l[n] = v()

        val = [eval_code(co, self.g, self.l) for co in self.co]
        return np.array(val, copy=False).reshape(self.sdim, self.sdim)
       

from petram.phys.vtable import VtableElement, Vtable

class Phys(Model, NS_mixin):
    hide_ns_menu = True
    hide_nl_panel = False
    dep_var_base = []
    der_var_base = []

    has_essential = False
    is_complex = False
    is_secondary_condition = False   # if true, there should be other
                                     # condtion assigned to the same
                                     # edge/face/domain
    vt   = Vtable(tuple())         
    vt2  = Vtable(tuple())         
    vt3  = Vtable(tuple())
    nlterms = []
                                     
    def __init__(self, *args, **kwargs):
        super(Phys, self).__init__(*args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)
        
    def attribute_set(self, v):
        v = super(Phys, self).attribute_set(v)
        self.vt.attribute_set(v)
        self.vt3.attribute_set(v)

        nl_config = dict()
        for k in self.nlterms: nl_config[k] = []
        v['nl_config'] = (False, nl_config)
        return v
        
    def get_possible_bdry(self):
        return []
    
    def get_possible_domain(self):
        return []                

    def get_possible_edge(self):
        return []                

    def get_possible_pair(self):
        return []        

    def get_possible_paint(self):
        return []

    def get_independent_variables(self):
        p = self.get_root_phys()
        ind_vars = p.ind_vars
        return [x.strip() for x in ind_vars.split(',')]
     
    def is_complex(self):
        return False

    def get_restriction_array(self, engine, idx = None):
        mesh = engine.meshes[self.get_root_phys().mesh_idx]
        intArray = mfem.intArray

        if isinstance(self, Domain):
            size = mesh.attributes.Size()
        else:
            size = mesh.bdr_attributes.Size()
     
        arr = [0]*size
        if idx is None: idx = self._sel_index
        for k in idx: arr[k-1] = 1
        return intArray(arr)

    def restrict_coeff(self, coeff, engine, vec = False, matrix = False, idx=None):
        arr = self.get_restriction_array(engine, idx)
#        arr.Print()
        if vec:
            return mfem.VectorRestrictedCoefficient(coeff, arr)
        elif matrix:
            return mfem.MatrixRestrictedCoefficient(coeff, arr)           
        else:
            return mfem.RestrictedCoefficient(coeff, arr)

    def get_essential_idx(self, idx):
        '''
        return index of essentail bdr for idx's fespace
        '''
        raise NotImplementedError(
             "you must specify this method in subclass")

    def get_root_phys(self):
        p = self
        while isinstance(p.parent, Phys):
           p = p.parent
        return p
        
    def get_exter_NDoF(self, kfes=0):
        return 0
    
    def has_extra_DoF(self, kfes=0):
        return False

    def update_param(self):
        ''' 
        called everytime it assembles either matrix or rhs
        '''
        pass
     
    def postprocess_extra(self, sol, flag, sol_extra):
        '''
        postprocess extra (Lagrange multiplier) segment
        of solutions. 
        
        sol_extra is a dictionary in which this method 
        will add processed data (dataname , value).
        '''   
        raise NotImplementedError(
             "you must specify this method in subclass")
       
    def has_bf_contribution(self, kfes):
        return False
    
    def has_lf_contribution(self, kfes):
        return False
     
    def has_interpolation_contribution(self, kfes):
        return False
     
    def has_mixed_contribution(self):
        return False

    def get_mixedbf_loc(self):
        '''
        r, c, and flag of MixedBilinearForm
        flag -1 :  use conjugate when building block matrix
        '''
        return []

    def add_bf_contribution(self, engine, a, real=True, **kwargs):
        raise NotImplementedError(
             "you must specify this method in subclass")

    def add_lf_contribution(self, engine, b, real=True, **kwargs):        
        raise NotImplementedError(
             "you must specify this method in subclass")

    def add_extra_contribution(self, engine, kfes, **kwargs):        
        ''' 
        return four elements
        M12, M21, M22, rhs
        '''
        raise NotImplementedError(
             "you must specify this method in subclass")

    def add_interplation_contribution(self, engine, **kwargs):        
        ''' 
        P^t A P y = P^t f, x = Py
        return P, nonzero, zero diagonals.
        '''
        raise NotImplementedError(
             "you must specify this method in subclass")
     
    def apply_essential(self, engine, gf, **kwargs):
        raise NotImplementedError(
             "you must specify this method in subclass")

    def get_init_coeff(self, engine, **kwargs):
        return None
     
    def apply_init(self, engine, gf, **kwargs):
        import warnings
        warnings.warn("apply init is not implemented to " + str(self))
        
    def add_mix_contribution(self, engine, a, real=True):
        '''
        return array of crossterms
        [[vertical block elements], [horizontal block elements]]
        array length must be the number of fespaces
        for the physics
        '''
        raise NotImplementedError(
             "you must specify this method in subclass")



    def check_phys_expr(self, value, param, ctrl, **kwargs):
        try:
            self.eval_phys_expr(str(value), param, **kwargs)
            return True
        except:
            import petram.debug
            import traceback
            if petram.debug.debug_default_level > 2:
                traceback.print_exc()
            return False

    def check_phys_expr_int(self, value, param, ctrl):
        return self.check_phys_expr(value, param, ctrl, chk_int = True)

    def check_phys_expr_float(self, value, param, ctrl):
        return self.check_phys_expr(value, param, ctrl, chk_float = True)
     
    def check_phys_expr_complex(self, value, param, ctrl):
        return self.check_phys_expr(value, param, ctrl, chk_complex = True)
     
    def check_phys_array_expr(self, value, param, ctrl, **kwargs):
        try:
            if not 'array' in self._global_ns:
               self._global_ns['array'] = np.array
            self.eval_phys_array_expr(str(value), param, **kwargs)
            return True
        except:
            import petram.debug
            import traceback
            if petram.debug.debug_default_level > 2:
               traceback.print_exc()
            return False
         
    def check_phys_array_expr_int(self, value, param, ctrl):
        return self.check_phys_array_expr(value, param, ctrl, chk_int = True)

    def check_phys_array_expr_float(self, value, param, ctrl):
        return self.check_phys_array_expr(value, param, ctrl, chk_float = True)

    def check_phys_array_expr_complex(self, value, param, ctrl):
        return self.check_phys_array_expr(value, param, ctrl, chk_complex = True)

    def eval_phys_expr(self, value, param,
                       chk_int = False, chk_complex = False, 
                       chk_float = False):
        def dummy():
            pass
        if value.startswith('='):
            return dummy,  value.split('=')[1]
        else:
            x = eval(value, self._global_ns, self._local_ns)
            if chk_int:
                x = int(x)
            elif chk_complex:
                x = complex(x)
            elif chk_float:
                x = float(x)
            else:
                x = x + 0   # at least check if it is number.
            dprint2('Value Evaluation ', param, '=', x)            
            return x, None
         
    def eval_phys_array_expr(self, value, param, chk_complex = False,
                             chk_float = False, chk_int = False):
        def dummy():
            pass
        if value.startswith('='):
            return dummy,  value.split('=')[1]           
        else:
            if not 'array' in self._global_ns:
               self._global_ns['array'] = np.array
            x = eval('array('+value+')', self._global_ns, self._local_ns)
            if chk_int:
                x = x.astype(int)
            elif chk_complex:
                x = x.astype(complex)
            elif chk_float:
                x = x.astype(float)
            else:
                x = x + 0   # at least check if it is number.
            dprint2('Value Evaluation ', param, '=', x)            
            return x, None
         
    # param_panel (defined in NS_mixin) verify if expression can be evaluated
    # phys_param_panel verify if the value is actually float.
    # it forces the number to become float after evaulating the expresison
    # using namespace.     
    def make_phys_param_panel(self, base_name, value, no_func = True,
                              chk_int = False,
                              chk_complex = False,
                              chk_float = False,
                              validator = None):
        if validator is None:
            if chk_int:
                validator = self.check_phys_expr_int
            elif chk_float:
                validator = self.check_phys_expr_float
            elif chk_complex:
                validator = self.check_phys_expr_complex            
            else:
                validator = self.check_phys_expr

        if no_func:
            return  [base_name + "(=)",  value, 0,  
                     {'validator': validator,
                     'validator_param':base_name}]
        else:
            return  [base_name + "(*)",  value, 0,  
                     {'validator':   validator,
                     'validator_param':base_name}]

    def make_matrix_panel(self, base_name, suffix, row = 1, col = 1,
                          chk_int = False,
                          chk_complex = False,
                          chk_float = False,
                          validator = None):
        if validator is None:
           
            if chk_int:
                validator = self.check_phys_expr_int
                validatora= self.check_phys_array_expr_int                      
            elif chk_float:
                validator = self.check_phys_expr_float           
                validatora= self.check_phys_array_expr_float   
            elif chk_complex:
                validator = self.check_phys_expr_complex
                validatora= self.check_phys_array_expr_complex                       
            else:
                validator = self.check_phys_expr
                validatora= self.check_phys_array_expr       

        a = [ {'validator': validator,
               'validator_param':base_name + n} for n in suffix]
        elp1 = [[None, None, 43, {'row': row,
                                  'col': col,
                                 'text_setting': a}],]
        elp2 = [[None, None, 0, {'validator': validatora,
                                 'validator_param': base_name + '_m'},]]

        ll = [None, None, 34, ({'text': base_name + '*  ',
                                'choices': ['Elemental Form', 'Array Form'],
                                'call_fit': False},
                                {'elp': elp1},  
                                {'elp': elp2},),]
        return ll

    def add_variables(self, solvar, n, solr, soli = None):
        '''
        add model variable so that a user can interept simulation 
        results. It is also used in cross-physics interaction.

        '''
        pass
    def add_domain_variables(self, v, n, suffix, ind_vars, solr, soli = None):
        pass
    def add_bdr_variables(self, v, n, suffix, ind_vars, solr, soli = None):
        pass

    def panel1_param(self):
        return self.vt.panel_param(self)
        
    def get_panel1_value(self):
        return self.vt.get_panel_value(self)

    def preprocess_params(self, engine):
        self.vt.preprocess_params(self)
        self.vt3.preprocess_params(self)               
        return

    def import_panel1_value(self, v):
        return self.vt.import_panel_value(self, v)

    def panel1_tip(self):
        return self.vt.panel_tip()

    def panel3_param(self):
        from petram.pi.widget_nl import NonlinearTermPanel
        l = self.vt3.panel_param(self)
        if self.hide_nl_panel or len(self.nlterms)==0:
           return l
        setting = {'UI':NonlinearTermPanel, 'names':self.nlterms}
        l.append([None, None, 99, setting])
        return l
        
    def get_panel3_value(self):
        if self.hide_nl_panel or len(self.nlterms)==0:
            return self.vt3.get_panel_value(self)
        else:
            return self.vt3.get_panel_value(self) + [self.nl_config]
    
    def import_panel3_value(self, v):
        if self.hide_nl_panel or len(self.nlterms)==0:
            self.vt3.import_panel_value(self, v)
        else:
            self.vt3.import_panel_value(self, v[:-1])
            self.nl_config = v[-1]

    def panel3_tip(self):
        if self.hide_nl_panel or len(self.nlterms)==0:
            return self.vt3.panel_tip()
        else:
            return self.vt3.panel_tip() +[None]

    def add_integrator(self, engine, name, coeff, adder, integrator, idx=None, vt=None):
        if coeff is None: return
        if vt is None: vt = self.vt
        if vt[name].ndim == 0:
           coeff = self.restrict_coeff(coeff, engine, idx=idx)
        elif vt[name].ndim == 1:
           coeff = self.restrict_coeff(coeff, engine, vec = True, idx=idx)
        else:
           coeff = self.restrict_coeff(coeff, engine, matrix = True, idx=idx)           
        adder(integrator(coeff))


class PhysModule(Phys):
    hide_ns_menu = False
    def attribute_set(self, v):
        v = super(PhysModule, self).attribute_set(v)
        v["order"] = 1
        v["element"] = 'H1_FECollection'
        v["dim"] = 2
        v["ind_vars"] = 'x, y'
        v["dep_vars_suffix"] = ''
        v["mesh_idx"] = 0
        return v
     
    def panel1_param(self):
        return [["mesh num.",   self.mesh_idx, 400, {}],
                ["element",self.element,  2,   {}],
                ["order",  self.order,    400, {}],]
    def panel1_tip(self):
        return ["index of mesh", "element type", "element order"]
    def get_panel1_value(self):                
        return [self.mesh_idx, self.element, self.order]
    def import_panel1_value(self, v):
        self.mesh_idx = long(v[0])
        self.element = str(v[1])
        self.order = long(v[2])
        return v[3:]
     
    @property
    def dep_vars(self):
        '''
        list of dependent variables, for example.
           [p]   
           [E]      
           [E, psi]
        '''
        return  ['p' + self.dep_vars_suffix]

    def dep_var_label(self, name):
        return name + self.dep_vars_suffix

    def dep_var_base(self, name):
        return name[:-len(self.dep_vars_suffix)]

    def dep_var_index(self, name):
        return self.dep_vars.index(name)

    def check_new_fespace(fespaces, meshes):
        mesh = meshes[self.mesh_name]
        fespacs = fespeces[self]

    def get_fec(self):
        name = self.dep_vars
        return [(name[0], self.element)]
     
    def is_complex(self):
        return False
     
    def soldict_to_solvars(self, soldict, variables):
        keys = soldict.keys()
        depvars = self.dep_vars
        suffix = self.dep_vars_suffix
        ind_vars = [x.strip() for x in self.ind_vars.split(',')]
        
        for k in keys:
            n = k.split('_')[0]
            if n in depvars:
               sol = soldict[k]
               solr = sol[0]
               soli = sol[1] if len(sol) > 1 else None
               self.add_variables(variables, n, solr, soli)
               
               # collect all definition from children
               for mm in self.walk():
                  if not mm.enabled: continue
                  if mm is self: continue
                  mm.add_domain_variables(variables, n, suffix, ind_vars,
                                    solr, soli)
                  mm.add_bdr_variables(variables, n, suffix, ind_vars,
                                    solr, soli)


            
       
         


        
