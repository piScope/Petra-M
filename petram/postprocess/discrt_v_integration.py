'''

   discrete variable integration

     1) integration of discrete varialbe on certain domain/boundary
        in such a way natural to the basis function.
    
        problem dimension (=mesh.Dimension())

        3D domain     H1    volume integration
                      L2    volume integration
                      ND    volume integration of dot prodcuts with coefficients
                      RT    volume integration of dot prodcuts with coefficients

           boundary   H1    area integration
                      L2    (undefined)
                      ND/RT area integration of dot prodcuts with coefficients

        2D domain:    H1/L2 area integration
                      ND/RT area integration of dot prodcuts with coefficients

           boundary:  H1    line integration
                      L2    (undefined)
                      ND    line integration of tangentail component
                      RT    line integration of normal component
        1D domain:    H1    line integration
                      L2    line integration
           boundary:  H1    local value
                      L2    (undefined)


'''
import numpy as np
import traceback
import warnings

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('Integration(PP)')

from petram.phys.vtable import VtableElement, Vtable, Vtable_mixin

from petram.postprocess.pp_model import PostProcessBase

from petram.phys.weakform import get_integrators
bilinintegs = get_integrators('BilinearOps')
linintegs = get_integrators('LinearOps')

data = [("coeff_lambda", VtableElement("coeff_lambda", type='array',
         guilabel = "lambda", default = 0.0, tip = "coefficient",))]

'''
from petram.helper.variables import var_g
ll = var_g.copy()

class DiscrtVIntegration(PostProcessBase, Vtable_mixin):
    
    has_2nd_panel = False
    vt_coeff = Vtable(data)
    
    @classmethod
    def fancy_menu_name(self):
        return 'Integraion(GF)'
    
    def attribute_set(self, v):
        v = super(DiscrtVIntegration, self).attribute_set(v)
        v["integral_name"] = 'intg'
        v["discrt_variable"]   = ""
        v['sel_index'] = ['all']
        v['sel_index_txt'] = 'all'
        v['is_boundary_int'] = False
        v['useFlux'] = False
        v['useNormal'] = False
        self.vt_coeff.attribute_set(v)         
        return v
    
    def panel1_param(self):
        import wx
        pa = self.vt_coeff.panel_param(self)                
        panels  =[["name", "", 0, {}],
                  ["variable", "", 0, {}],
                  pa[0],                  
                  ["sel.",     "", 0, {}],                  
                  ["boundary", False, 3, {"text":""}],
                  ["normal integrator", False, 3, {"text":""}],
                  ["flux integrator", False, 3, {"text":""}],]
        return panels

    def panel1_tip(self):
        return ["name", "discrete variable", "sel.", "boundary"]
     
    def get_panel1_value(self):                
        return [self.integral_name,
                self.discrt_variable,
                self.vt_coeff.get_panel_value(self)[0],                        
                self.sel_index_txt,
                self.is_boundary_int,
                self.useFlux,
                self.useNormal]
    
    def import_panel1_value(self, v):
        self.integral_name = str(v[0])
        self.discrt_variable = str(v[1])
        self.vt_coeff.import_panel_value(self, (v[2],))        
        self.sel_index_txt = str(v[3])
        self.is_boundary_int = bool(v[4])
        self.useFlux = bool(v[5])
        self.useNormal = bool(v[6])

    def run_postprocess(self, engine):
        dprint1("running postprocess: " + self.name())

        var = engine.model._variables[self.discrt_variable]
        
        fes = var.FESpace()
        vdim = fes.GetVDim()
        mesh = fes.GetMesh()
        dim = mesh.Dimension()
        sdim = mesh.SpaceDimension()
        fec = fes.FEColl()
        

        isDomain = not self.is_boundary_int        

        from petram.helper.variables import var_g
        global_ns = var_g.copy()
        for k in engine.model._variables:
            global_ns[k] = engine.model._variables[k]
        local_ns = {}

        if self.sel_index_txt != 'all':
            idx = np.atleast_1d(eval(self.sel_index_txt,
                                     global_ns, local_ns)).astype(int, copy=False)
            if isDomain:
                size = max(np.max(mesh.attributes.ToList()), engine.max_attr)                
            else:
                size = max(np.max(mesh.bdr_attributes.ToList()), engine.max_bdrattr)
            arr = [0]*size
            if idx is None: idx = self._sel_index
            for k in idx: arr[k-1] = 1
            arr = mfem.intArray(arr)
        else:
            arr = None
        
        fec_name = fec.Name()
        if fec_name.startswith("H1"):
            elem = 'H1'
            vlen = vdim                                    
        elif fec_name.startswith("RT"):
            elem = 'RT'
            vlen = sdim
        elif fec_name.startswith("ND"):
            elem = 'ND'
            vlen = sdim            
        elif fec_name.startswith("L2"):
            elem = 'L2'
            vlen = vdim                        
        else:
            assert False, "Unsupported element"

        self.vt_coeff.preprocess_params(self)
        c = self.vt_coeff.make_value_or_expression(self)
        ind_vars = self.root()['Phys'].values()[0].ind_vars

        if isComplex:
            realflag = [True, False]

        else:
            realflag = [True, ]

        lfs = []
        for real in realflag:
            lf1 = engine.new_lf(fes)
            if vlen > 1:
                coeff = VCoeff(vlen, c[0], ind_vars,
                                local_ns, global_ns, real = real)
                if arr is not None:                
                    coeff = mfem.VectorRestrictedCoefficient(coeff, arr)
            else:
                coeff = SCoeff(c[0], ind_vars,
                                local_ns, global_ns, real = real)
                if arr is not None:                                
                    coeff = mfem.RestrictedCoefficient(coeff, arr)                
        

                
            if dim == 3:
                if isDomain:
                    if elem == 'H1' or elem == 'L2':
                        if vlen == 1:
                            intg = mfem.DomainLFIntegrator(coff)
                        else:
                            intg = mfem.VectorDomainLFIntegrator(coff)
                    elif elem == 'RT' or elem == 'ND':
                        intg = mfem.VectorFEDomainLFIntegrator(coff)                        
                else:
                    if elem == 'H1' or elem == 'L2':
                        if vlen == 1:
                            if isNormal:
                                intg = mfem.BoundaryNormalLFIntegrator(coff)
                            else:
                                intg = mfem.BoundaryLFIntegrator(coff)
                        else:
                            if isFlux:
                                intg = mfem.VectorBoundaryFluxLFIntegrator(coff)
                            else:
                                intg = mfem.VectorBoundaryLFIntegrator(coff)
                    elif elem == 'RT':
                        integ = VectorFEBoundaryFluxLFIntegraton(coff)
                    elif elem == 'ND':
                        integ = VectorFEBoundaryTangentLFIntegraton(coff)
                    else:
                        assert False, "Unsupported integration" 
                        intg = mfem.VectorFEDomainLFIntegrator(coff)                        
            elif dim == 2:
                if isDomain:
                    if elem == 'H1' or elem == 'L2':
                        if vlen == 1:
                            intg = mfem.DomainLFIntegrator(coff)
                        else:
                            intg = mfem.VectorDomainLFIntegrator(coff)
                    elif elem == 'RT' or elem == 'ND':
                        intg = mfem.VectorFEDomainLFIntegrator(coff)                        
                else:
                    if elem == 'H1' or elem == 'L2':
                        if vlen == 1:
                            if isNormal:
                                intg = mfem.BoundaryNormalLFIntegrator(coff)
                            else:
                                intg = mfem.BoundaryLFIntegrator(coff)
                        else:
                            if isFlux:
                                intg = mfem.VectorBoundaryFluxLFIntegrator(coff)
                            else:
                                intg = mfem.VectorBoundaryLFIntegrator(coff)
                    elif elem == 'RT':
                        integ = VectorFEBoundaryFluxLFIntegraton(coff)
                    elif elem == 'ND':
                        integ = VectorFEBoundaryTangentLFIntegraton(coff)
                    else:
                        assert False, "Unsupported integration" 
                        intg = mfem.VectorFEDomainLFIntegrator(coff)                        
            else:
                if isDomain:
                    if elem == 'H1' or elem == 'L2':
                        if vlen == 1:
                            intg = mfem.DomainLFIntegrator(coff)
                        else:
                            intg = mfem.VectorDomainLFIntegrator(coff)
                else:
                    if elem == 'H1':                                    
                        if vlen == 1:
                            if isNormal:
                                intg = mfem.BoundaryNormalLFIntegrator(coff)
                            else:
                                intg = mfem.BoundaryLFIntegrator(coff)
                        else:
                            if isFlux:
                                intg = mfem.VectorBoundaryFluxLFIntegrator(coff)
                            else:
                                intg = mfem.VectorBoundaryLFIntegrator(coff)
                    else:
                        assert False, "Unsupported integration"
                        
        lf1.AddDomainIntegrator(intg)
        lf1.Assemble()
        
        from mfem.common.chypre import LF2PyVec, PyVec2PyMat, MfemVec2PyVec
        v1 = MfemVec2PyVec(engine.b2B(lf1), None)
'''
data = [("coeff_lambda", VtableElement("coeff_lambda", type='array',
         guilabel = "lambda", default = '1.0', tip = "coefficient",))]

class WeakformIntegrator(PostProcessBase, Vtable_mixin):
    has_2nd_panel = True
    vt_coeff = Vtable(data)
    
    @property
    def geom_dim(self):  # dim of geometry
        return self.root()['Mesh'].sdim
    
    def attribute_set(self, v):
        v = super(WeakformIntegrator, self).attribute_set(v)
        v["integration_name"] = 'integ'        
        v['coeff_type'] = 'S'
        v['integrator'] = 'MassIntegrator'
        v['variables']   = ''
        v["sdim"] = 2
        v['sel_index'] = ['all']
        v['sel_index_txt'] = 'all'
        self.vt_coeff.attribute_set(v)
        return v

    def panel1_param(self):
        import wx       
        p = ["coeff. type", "S", 4,
             {"style":wx.CB_READONLY, "choices": ["Scalar", "Vector", "Diagonal", "Matrix"]}]

        names = [x[0] for x in self.itg_choice()]
        names = ['Auto'] + names
        p2 = ["integrator", names[0], 4,
              {"style":wx.CB_READONLY, "choices": names}]
        
        panels = self.vt_coeff.panel_param(self)
        ll = [["name", "", 0, {}],
              ["variable", self.variables, 0, {}],
               p,
               panels[0],
               p2,]

        return ll
    
    def get_panel1_value(self):
        return [self.integration_name,
                self.variables, 
                self.coeff_type,
                self.vt_coeff.get_panel_value(self)[0],
                self.integrator,]
              
    def import_panel1_value(self, v):
        self.integration_name = str(v[0])
        self.variables = str(v[1])
        self.coeff_type = str(v[2])
        self.vt_coeff.import_panel_value(self, (v[3],))
        self.integrator =  str(v[4])

    def panel1_tip(self):
        pass

    def panel2_param(self):
        import wx
        
        if self.geom_dim == 3:
           choice = ("Volume", "Surface", "Edge")
        elif self.geom_dim == 2:
           choice = ("Surface", "Edge")
        elif self.geom_dim == 1:
           choice = ("Edge", )

        p = ["Type", choice[0], 4,
             {"style":wx.CB_READONLY, "choices": choice}]
        return [p, ["index",  'all',  0,   {'changing_event':True,
                                            'setfocus_event':True}, ]]
              
    def get_panel2_value(self):
        choice = ["Point", "Edge", "Surface", "Volume",]
        return choice[self.sdim], self.sel_index_txt
     
    def import_panel2_value(self, v):
        if str(v[0]) == "Volume":
           self.sdim = 3
        elif str(v[0]) == "Surface":
           self.sdim = 2
        elif str(v[0]) == "Edge":
           self.sdim = 1                      
        else:
           self.sdim = 1                                 
        self.sel_index_txt = str(v[1])
           
        from petram.model import convert_sel_txt
        try:
            g = self._global_ns            
            arr = convert_sel_txt(self.sel_index_txt, g)
            self.sel_index = arr            
        except:
            import traceback
            traceback.print_exc()
            assert False, "failed to convert "+self.sel_index_txt
            
    def panel2_tip(self):
        pass
    
                

    def add_integrator(self, form, engine, integrator, cdim, emesh_idx, isDomain, real):
        
        self.vt_coeff.preprocess_params(self)
        c = self.vt_coeff.make_value_or_expression(self)
        
        from petram.helper.variables import var_g
        global_ns = self._global_ns.copy()
        for k in engine.model._variables:
            global_ns[k] = engine.model._variables[k]
        local_ns = {}

        phys = self.root()['Phys'].values()
        for p in phys:
            if p.enabled:
                ind_vars = p.ind_vars
                break
        else:
            assert False, "no phys is enabled"

        from petram.helper.phys_module_util import restricted_integrator

        itg = restricted_integrator(engine, integrator, self.sel_index, 
                                    c[0], self.coeff_type[0], cdim, 
                                    emesh_idx,
                                    isDomain,
                                    ind_vars, local_ns, global_ns, real)

        if isDomain:
            adder = form.AddDomainIntegrator
        else:
            adder = form.AddBoundaryIntegrator
        adder(itg)
    
class LinearformIntegrator(WeakformIntegrator):
    def itg_choice(self):
        return linintegs
    
    @classmethod    
    def fancy_menu_name(self):
        return "LinearForm"
    
    @classmethod    
    def fancy_tree_name(self):
        return 'Integration_LF'
    
    def run_postprocess(self, engine):
        dprint1("running postprocess: " + self.name())

        
        name = self.variables.strip()
        if name not in engine.model._variables:
            assert False, name + " is not defined"
            
        var1 = engine.model._variables[name]
        emesh_idx1 = var1.get_emesh_idx()
        emesh_idx = emesh_idx1[0]
            
        fes1 = engine.fespaces[name]
        info1 = engine.get_fes_info(fes1)
        if info1 is None:
            assert False, "fes info is not found"

        if info1['sdim']  == self.sdim:
            isDomain = True
        elif info1['sdim']  == self.sdim+1:
            isDomain = False
        else:
            warnings.warn("Can not perform integration (skipping)", RuntimeWarning)
            return

        isComplex = var1.complex
        new_lf = engine.new_lf

        cdim = 1
        if (info1['element'].startswith('ND') or
            info1['element'].startswith('RT')):
            cdim = info1['sdim']
        else:
            cdim = info1['vdim']

        from petram.helper.phys_module_util import default_lf_integrator
        if self.integrator == 'Auto':
            integrator = default_lf_integrator(info1, isDomain)
        else:
            integrator = self.integrator
            
        lfr = new_lf(fes1)
        self.add_integrator(lfr, engine, integrator, cdim, emesh_idx, isDomain, True)
        lfr.Assemble()

        if isComplex:
            lfi = new_lf(fes1)
            self.add_integrator(lfi, engine, integrator, cdim, emesh_idx, isDomain, False)
            lfi.Assemble()
        else:
            lfi = None

        from mfem.common.chypre import LF2PyVec
        V = LF2PyVec(lfr, lfi, horizontal = True)
   
        V1 = engine.variable2vector(var1)
        #print(V1.shape)
        #print(V.shape)
        value = np.array(V.dot(V1), copy=False)
        #value = np.array(V1.dot(V2), copy=False)
        dprint1("Integrated Value :" + self.integration_name + ":" + str(value))
        engine.store_pp_extra(self.integration_name, value)
    
class BilinearformIntegrator(WeakformIntegrator):
    def itg_choice(self):
        return bilinintegs
    
    @classmethod    
    def fancy_menu_name(self):
        return "BilinearForm"
    
    @classmethod    
    def fancy_tree_name(self):
        return 'Integration_BF'

    def attribute_set(self, v):
        v = super(BilinearformIntegrator, self).attribute_set(v)
        v["use_conj"] = False
        return v
        
    def panel1_param(self):
        ll = super(BilinearformIntegrator, self).panel1_param()
        ll[1][0] = "variables"
        ll.append(["use conj (ex. AB^)", self.use_conj, 3, {"text":""}])
        return ll
    
    def get_panel1_value(self):
        v = super(BilinearformIntegrator, self).get_panel1_value()
        v.append(self.use_conj)
        return v
              
    def import_panel1_value(self, v):
        super(BilinearformIntegrator, self).import_panel1_value(v[:-1])
        self.use_conj = bool(v[-1])

    def run_postprocess(self, engine):
        dprint1("running postprocess: " + self.name())

        
        names = [x.strip() for x in self.variables.split(',')]
        if len(names) != 2:
            assert False, "wrong setting. specify two variables: " + self.variables
        if names[0] not in engine.model._variables:
            assert False, names[0] + " is not defined"
        if names[1] not in engine.model._variables:            
            assert False, names[1] + " is not defined"
            
        var1 = engine.model._variables[names[0]]
        var2 = engine.model._variables[names[1]]
        
        emesh_idx1 = var1.get_emesh_idx()
        emesh_idx2 = var2.get_emesh_idx()

        if (len(emesh_idx1) != 1 or
            len(emesh_idx2) != 1 or
            emesh_idx1[0] != emesh_idx2[0]):
            assert False, "can not perform integration between different extended-mesh"
        emesh_idx = emesh_idx1[0]
            

        fes1 = engine.fespaces[names[0]]
        fes2 = engine.fespaces[names[1]]
        info1 = engine.get_fes_info(fes1)
        info2 = engine.get_fes_info(fes2)        
        if info1 is None or info2 is None:
            assert False, "fes info is not found"

        if info1['sdim']  == self.sdim:
            isDomain = True
        elif info1['sdim']  == self.sdim+1:
            isDomain = False
        else:
            warnings.warn("Can not perform integration (skipping)", RuntimeWarning)
            return

        isComplex = var1.complex or var2.complex
        if fes1 != fes2:
            new_bf = engine.new_mixed_bf
        else:
            new_bf = engine.new_bf

        cdim = 1
        if (info1['element'].startswith('ND') or
            info1['element'].startswith('RT')):
            cdim = info1['sdim']
        else:
            cdim = info1['vdim']

        from petram.helper.phys_module_util import default_bf_integrator
        if self.integrator == 'Auto':
            integrator = default_bf_integrator(info1, info2, isDomain)            
            '''
            if fes1 == fes2:
                if (info1['element'].startswith('ND') or
                    info1['element'].startswith('RT')):
                    integrator = 'VectorFEMassIntegrator'
                elif info1['element'].startswith('DG'):
                    assert False, "auto selection is not supported for GD"
                elif info1['vdim'] > 1:
                    integrator = 'VectorMassIntegrator'
                elif not isDomain:
                    integrator = 'BoundaryMassIntegrator'
                else:
                    integrator = 'MassIntegrator'
            else:
                if (((info1['element'].startswith('ND') or
                      info1['element'].startswith('RT'))and
                     (info2['element'].startswith('ND') or
                      info2['element'].startswith('RT')))):
                    integrator = 'MixedVectorMassIntegrator'
                elif info1['element'].startswith('DG'):
                    assert False, "auto selection is not supported for GD"
                elif (info1['vdim'] == 1 and info1['vdim'] == info2['vdim']):
                    integrator = 'MixedScalarMassIntegrator'                    
                else:
                    assert False, "No proper integrator is found"
            '''
        else:
            integrator = self.integrator
            
        bfr = new_bf(fes2, fes1)
        self.add_integrator(bfr, engine, integrator, cdim,  emesh_idx, isDomain, True)
        bfr.Assemble()

        if isComplex:
            bfi = new_bf(fes2, fes1)
            self.add_integrator(bfi, engine, integrator, cdim, emesh_idx, isDomain, False)
            bfi.Assemble()
        else:
            bfi = None

        from mfem.common.chypre import BF2PyMat
        
        M = BF2PyMat(bfr, bfi, finalize = True)
   
        V1 = engine.variable2vector(var1, horizontal=True)
        V2 = engine.variable2vector(var2)        
        if self.use_conj:
            try:
                V2 = V2.conj()
            except:
                V2[1] *= -1
        #print(M.shape)
        #print(V1.shape)
        #print(V2.shape)
        value = np.array(V1.dot(M.dot(V2)), copy=False)
        #value = np.array(V1.dot(V2), copy=False)
        dprint1("Integrated Value :" + self.integration_name + ":" + str(value))        
        engine.store_pp_extra(self.integration_name, value)
        

