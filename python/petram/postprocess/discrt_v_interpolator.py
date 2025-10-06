'''
   discrete variable interpolator

   Interface to DiscreteInterpolators in Postprocessing
     


'''
import numpy as np
import traceback
import warnings
import weakref

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('Integration(PP)')

from petram.phys.vtable import VtableElement, Vtable, Vtable_mixin

from petram.postprocess.pp_model import PostProcessBase

from petram.phys.weakform import get_integrators
bilinintegs = get_integrators('BilinearOps')
linintegs = get_integrators('LinearOps')

def add_variable_vector(obj, v, names, gfr, gfi):
    from petram.helper.variables import add_components

    for phys_name in obj.root()['Phys']:
        if not obj.root()['Phys'][phys_name].enabled:
            continue
        if obj.root()['Phys'][phys_name].emesh_idx == gfr._emesh_idx:
            ind_vars = obj.root()['Phys'][phys_name].ind_vars
            break

    ind_vars = [x.strip() for x in ind_vars.split(',') if x.strip() != '']

    add_components(v, names[0], "", ind_vars, gfr, gfi)

def add_variable_scalar(obj, v, names, gfr, gfi):
    from petram.helper.variables import add_scalar

    for phys_name in obj.root()['Phys']:
        if not obj.root()['Phys'][phys_name].enabled:
            continue
        if obj.root()['Phys'][phys_name].emesh_idx == gfr._emesh_idx:
            ind_vars = obj.root()['Phys'][phys_name].ind_vars
            break

    ind_vars = [x.strip() for x in ind_vars.split(',') if x.strip() != '']

    add_scalar(v, names[0], "", ind_vars, gfr, gfi)

class DSInterpolator(PostProcessBase, Vtable_mixin):
    has_2nd_panel = False
    
    @property
    def geom_dim(self):  # dim of geometry
        return self.root()['Mesh'].sdim
    
    def attribute_set(self, v):
        v = super(DSInterpolator, self).attribute_set(v)
        v["interpolation_name"] = 'interp'        
        v['variables'] = ''
        v['sel_index'] = ['all']
        v['sel_index_txt'] = 'all'
        v['order'] = 0        
        return v

    def panel1_param(self):
        ll = [["name", "", 0, {}],
              ["variable", self.variables, 0, {}],
              ["order", self.order, 400, {}],]        
        return ll
    
    def get_panel1_value(self):
        return [self.interpolation_name,
                self.variables,
                self.order]
    
    def import_panel1_value(self, v):
        self.interpolation_name = str(v[0])
        self.variables = str(v[1])
        self.order = int(v[2])

    def panel1_tip(self):
        pass

    def update_dom_selection(self, all_sel=None):
        pass


    def add_variables(self, v, names, gfr, gfi):
        assert False, "need to be implemented in subclass"
        
    def soldict_to_solvars(self, soldict, variables):
        suffix = ""
        names = [self.interpolation_name.strip()]
        fname = self.interpolation_name.strip()

        for k in soldict:
            n = '_'.join(k.split('_')[:-1])
            if n == fname:
               sol = soldict[k]
               solr = sol[0]
               soli = sol[1] if len(sol) > 1 else None
               self.add_variables(variables, names, solr, soli)


class Grad(DSInterpolator):
    @classmethod
    def fancy_menu_name(self):
        return "Grad"

    @classmethod
    def fancy_tree_name(self):
        return 'Grad'

    def attribute_set(self, v):
        v = super(Grad, self).attribute_set(v)
        v['interpolation_name'] = 'grad'        
        return v

    def add_variables(self, v, names, gfr, gfi):
        add_variable_vector(self, v, names, gfr, gfi)

    def run_postprocess(self, engine):
        dprint1("running postprocess: " + self.name())

        name = self.variables.strip()
        if name not in engine.model._variables:
            print(list(engine.model._variables))
            assert False, name + " is not defined"

        var1 = engine.model._variables[name]

        emesh_idx1 = var1.get_emesh_idx()
        emesh_idx = emesh_idx1[0]

        fes1 = engine.fespaces[name]
        element = 'ND_FECollection'

        _is_new, fes2 = engine.get_or_allocate_fecfes(self.interpolation_name,
                                                          emesh_idx,
                                                          element,
                                                          self.order,
                                                          1,)

        gfr = engine.new_gf(fes2)
        if var1.complex:
            gfi = engine.new_gf(fes2)
        else:
            gfi = None

        from petram.helper.operators import Gradient
        opr = Gradient()
        opr._engine = weakref.ref(engine)
        M = opr.assemble(trial=fes1, test=fes2)

        V1 = engine.variable2vector(var1)
        value = M.dot(V1)
        dprint1("operator size", M.shape)

        if var1.complex:
            engine.X2x(value.real, gfr)
            engine.X2x(value.imag, gfi)
        else:
            engine.X2x(value.real, gfr)
            
        from petram.helper.variables import Variables
        
        v = Variables()

        names = [self.interpolation_name.strip()]
        self.add_variables(v, names, gfr, gfi)

        engine.add_PP_to_NS(v)
        engine.save_solfile_fespace(self.interpolation_name, emesh_idx, gfr, gfi)

class Curl(DSInterpolator):
    @classmethod    
    def fancy_menu_name(self):
        return "Curl"
    
    @classmethod    
    def fancy_tree_name(self):
        return 'Curl'
    
    def attribute_set(self, v):
        v = super(Curl, self).attribute_set(v)
        v['interpolation_name'] = 'curl'
        return v

    def add_variables(self, v, names, gfr, gfi):
        fec_name = gfr.FESpace().FEColl().Name()
        if fec_name.startswith('RT'):
            add_variable_vector(self, v, names, gfr, gfi)
        else:
            add_variable_scalar(self, v, names, gfr, gfi)
            
    def run_postprocess(self, engine):
        dprint1("running postprocess: " + self.name())

        name = self.variables.strip()
        if name not in engine.model._variables:
            print(list(engine.model._variables))
            assert False, name + " is not defined"

        var1 = engine.model._variables[name]

        emesh_idx1 = var1.get_emesh_idx()
        emesh_idx = emesh_idx1[0]

        fes1 = engine.fespaces[name]

        dim1 = fes1.GetMesh().Dimension()
        if dim1 == 3:
            element = 'RT_FECollection'
        elif dim1 == 2:
            element = 'L2_FECollection'
        else:
            assert False, "curl supports 2D/3D only"
            
        _is_new, fes2 = engine.get_or_allocate_fecfes(self.interpolation_name,
                                                          emesh_idx,
                                                          element,
                                                          self.order,
                                                          1,)

        gfr = engine.new_gf(fes2)
        if var1.complex:
            gfi = engine.new_gf(fes2)
        else:
            gfi = None

        from petram.helper.operators import Curl
        opr = Curl()
        opr._engine = weakref.ref(engine)
        M = opr.assemble(trial=fes1, test=fes2)

        V1 = engine.variable2vector(var1)
        value = M.dot(V1)

        dprint1("operator size", M.shape)        

        if var1.complex:
            engine.X2x(value.real, gfr)
            engine.X2x(value.imag, gfi)
        else:
            engine.X2x(value.real, gfr)
            
        from petram.helper.variables import Variables
        
        v = Variables()        
        names = [self.interpolation_name.strip()]
        self.add_variables(v, names, gfr, gfi)

        engine.add_PP_to_NS(v)
        engine.save_solfile_fespace(self.interpolation_name, emesh_idx, gfr, gfi)


class Div(DSInterpolator):
    @classmethod    
    def fancy_menu_name(self):
        return "Div"
    
    @classmethod    
    def fancy_tree_name(self):
        return 'Div'
    
    def attribute_set(self, v):
        v = super(Div, self).attribute_set(v)
        v['interpolation_name'] = 'div'                
        return v

    def add_variables(self, v, names, gfr, gfi):
        add_variable_scalar(self, v, names, gfr, gfi)

    def run_postprocess(self, engine):
        dprint1("running postprocess: " + self.name())

        name = self.variables.strip()
        if name not in engine.model._variables:
            print(list(engine.model._variables))
            assert False, name + " is not defined"

        var1 = engine.model._variables[name]

        emesh_idx1 = var1.get_emesh_idx()
        emesh_idx = emesh_idx1[0]

        fes1 = engine.fespaces[name]
        element = 'L2_FECollection'

        _is_new, fes2 = engine.get_or_allocate_fecfes(self.interpolation_name,
                                                          emesh_idx,
                                                          element,
                                                          self.order,
                                                          1,)

        gfr = engine.new_gf(fes2)
        if var1.complex:
            gfi = engine.new_gf(fes2)
        else:
            gfi = None

        from petram.helper.operators import Divergence
        opr = Divergence()
        opr._engine = weakref.ref(engine)
        M = opr.assemble(trial=fes1, test=fes2)

        V1 = engine.variable2vector(var1)
        value = M.dot(V1)

        dprint1("operator size", M.shape)                

        if var1.complex:
            engine.X2x(value.real, gfr)
            engine.X2x(value.imag, gfi)
        else:
            engine.X2x(value.real, gfr)
            
        from petram.helper.variables import Variables
        
        v = Variables()        
        names = [self.interpolation_name.strip()]
        self.add_variables(v, names, gfr, gfi)

        engine.add_PP_to_NS(v)
        engine.save_solfile_fespace(self.interpolation_name, emesh_idx, gfr, gfi)
    
