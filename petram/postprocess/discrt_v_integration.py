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
import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('Integration(PP)')

from petram.phys.vtable import VtableElement, Vtable, Vtable_mixin

from petram.postprocess.pp_model import PostProcessBase

from petram.phys.weakform import get_integrators
bilinintegs = get_integrators('BilinearOps')
linintegs = get_integrators('LinearOps')

data = [("coeff_lambda", VtableElement("coeff_lambda", type='array',
         guilabel = "lambda", default = 0.0, tip = "coefficient",))]


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
        p2 = ["integrator", names[0], 4,
              {"style":wx.CB_READONLY, "choices": names}]
        
        panels = self.vt_coeff.panel_param(self)
        ll = [["Variable", self.variables, 0, {}],
               p,
               panels[0],
               p2,]

        return ll
    
    def get_panel1_value(self):
        return [self.variables, 
                self.coeff_type,
                self.vt_coeff.get_panel_value(self)[0],
                self.integrator,]
              
    def import_panel1_value(self, v):
        self.variables = str(v[0])
        self.coeff_type = str(v[1])
        self.vt_coeff.import_panel_value(self, (v[2],))
        self.integrator =  str(v[3])

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
    
class LinearformIntegrator(WeakformIntegrator):
    def itg_choice(self):
        return linintegs
    
    @classmethod    
    def fancy_menu_name(self):
        return "Using LinearForm"
    
    @classmethod    
    def fancy_tree_name(self):
        return 'Integration_LF'
    
    def run_postprocess(self, engine):
        dprint1("running postprocess: " + self.name())
        import warnings
        warnings.warn("LinearIntegrator is not implemented", RuntimeWarning)
        return 

        var = engine.model._variables[self.discrt_variable]
    
class BilinearformIntegrator(WeakformIntegrator):
    def itg_choice(self):
        return bilinintegs
    
    @classmethod    
    def fancy_menu_name(self):
        return "Using BilinearForm"
    
    @classmethod    
    def fancy_tree_name(self):
        return 'Integration_BF'

    def attribute_set(self, v):
        v = super(BilinearformIntegrator, self).attribute_set(v)
        v["use_conj"] = False
        return v
        
    def panel1_param(self):
        ll = super(BilinearformIntegrator, self).panel1_param()
        ll[0][0] = "Variables"
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
        
        import warnings
        warnings.warn("BilinearIntegrator is not implemented", RuntimeWarning)
        return 
        var = engine.model._variables[self.discrt_variable]

        assert False, "Not yet implemented"
