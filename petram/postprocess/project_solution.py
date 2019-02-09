import traceback

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('Projection(PP)')

from petram.phys.vtable import VtableElement, Vtable, Vtable_mixin


from petram.postprocess.pp_model import PostProcessBase

data = [("coeff_lambda", VtableElement("coeff_lambda", type='array',
         guilabel = "lambda", default = 0.0, tip = "coefficient",))]

class DerivedValue(PostProcessBase, Vtable_mixin):
    has_2nd_panel = True
    vt_coeff = Vtable(data)
    
    def attribute_set(self, v):
        v = super(DerivedValue, self).attribute_set(v)
        v["projection_name"] = 'derived'
        v["element"] = 'H1_FECollection'
        v["is_complex_valued"] = False
        v["order"]   = 1
        v["sdim"] = 2
        v['sel_index'] = ['all']
        v['sel_index_txt'] = 'all'
        self.vt_coeff.attribute_set(v)        
        return v

    @property
    def vdim(self):
        name = [x.strip() for x in self.projection_name.split(',')]

        if (self.element.startswith('ND') or
            self.element.startswith('RT')):
            sdim = self.geom_dim
        else:
            sdim = 1
        return len(name)*sdim

    @property
    def geom_dim(self):  # dim of geometry
        return self.root()['Mesh'].sdim
    
    def panel1_param(self):
        import wx
        pa = self.vt_coeff.panel_param(self)        
        panels  =[["name", "", 0, {}],
                  pa[0],
                  ["element", "H1_FECollection", 4,
                                 {"style":wx.CB_READONLY, 
                                   "choices": ["H1_FECollection",
                                               "L2_FECollection",
                                               "ND_FECollection",
                                               "RT_FECollection",
                                               "DG_FECollection"]}],
                  ["order",     self.order,     400,  {}],
                  ["complex", self.is_complex_valued, 3, {"text":""}],]

        return panels

    def panel1_tip(self):
        return ["name", "expression", "element type", "element order", "complex"]
     
    def get_panel1_value(self):                
        return [self.projection_name,
                self.vt_coeff.get_panel_value(self)[0],
                self.element, self.order, self.is_complex_valued]
    
    def import_panel1_value(self, v):
        self.projection_name = str(v[0])
        self.vt_coeff.import_panel_value(self, (v[1],))        
        self.element = str(v[2])
        self.order = int(v[3])
        self.is_complex_valued = bool(v[4])

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
            g = self.root()['General']._global_ns            
            arr = convert_sel_txt(self.sel_index_txt, g)
            self.sel_index = arr            
        except:
            import traceback
            traceback.print_exc()
            assert False, "failed to convert "+self.sel_index_txt

    def get_emesh_idx(self, engine):
        
        from petram.mesh.mesh_extension import MeshExtInfo
        
        info = MeshExtInfo(dim = self.sdim, base = 0)
        if self.sel_index[0] != 'all':        
           info.set_selection(self.sel_index)
        idx = engine.emesh_data.add_info(info)
        
        if len(engine.emeshes) <= idx:
            assert False, "Extended Mesh was not generated for this postprocessing"
        return idx
            
    def run_postprocess(self, engine):
        dprint1("running postprocess: " + self.name())

        names = [x.strip() for x in self.projection_name.split(',')]
        
        from petram.helper.variables import var_g
        global_ns = var_g.copy()
        for k in engine.model._variables:
            global_ns[k] = engine.model._variables[k]
        local_ns = {}
        
        isVector =  self.vdim > 1

        emesh_idx = self.get_emesh_idx(engine)
        mesh = engine.emeshes[emesh_idx]

        from petram.engine import mfem
        
        fec = getattr(mfem, self.element)
        sdim= self.geom_dim

        fec = fec(self.order, sdim)

        if (self.element.startswith('ND') or
            self.element.startswith('RT')):
            vdim = self.vdim/self.geom_dim
        else:
            vdim = self.vdim 
        fes = engine.new_fespace(mesh, fec, vdim)

        engine.add_fec_fes(''.join(names), fec, fes)

        gfr = engine.new_gf(fes)
        if self.is_complex_valued:
            gfi = engine.new_gf(fes)
        else:
            gfi = None

        from petram.phys.weakform import VCoeff, SCoeff
        self._global_ns = global_ns
        self.vt_coeff.preprocess_params(self)
        c = self.vt_coeff.make_value_or_expression(self)

        ind_vars = self.root()['Phys'].values()[0].ind_vars


        def project_coeff(gf, vdim, c, ind_vars, real):
            if vdim > 1:
                 coeff = VCoeff(vdim, c[0], ind_vars,
                                local_ns, global_ns, real = real)
            else:
                 coeff = SCoeff(c[0], ind_vars,
                                local_ns, global_ns, real = real)
            gf.ProjectCoefficient(coeff)
            
        project_coeff(gfr, vdim, c, ind_vars, True)
        if gfi is not None:
            project_coeff(gfi, vdim, c, ind_vars, False)

        from petram.helper.variables import Variables
        v = Variables()
            
        self.add_variables(v, names, gfr, gfi, mesh)

        engine.add_PP_to_NS(v)
        engine.save_solfile_fespace(''.join(names), emesh_idx, gfr, gfi)
        
    def add_variables(self, v, names, gfr, gfi, mesh):
        ind_vars = self.root()['Phys'].values()[0].ind_vars
        
        isVector = False; isNormal=False
        if (self.element.startswith('RT') and mesh.Dimension == 3):
            isVector = True
        if (self.element.startswith('RT') and mesh.Dimension < 3):
            isNormal = True
        if self.element.startswith('ND'): 
            isVector = True

        from petram.helper.variables import add_scalar
        from petram.helper.variables import add_components
        from petram.helper.variables import GFScalarVariable        
           
        if isVector:
            add_components(v, names[0], "", ind_vars, gfr, gfr)
        elif isNormal:
            add_scalar(v, names[0]+"n", "", ind_vars, gfr, gfi)                
        else:
            if self.vdim == 1:
                add_scalar(v, names[0], "", ind_vars, gfr, gfi)
            else:
                for k, n in enumerate(names):
                    v[n] = GFScalarVariable(gfr, gfi, comp=k+1)
       
        
    def soldict_to_solvars(self, mesh, soldict, variables):

        keys = soldict.keys()
        suffix = ""
        names = [x.strip() for x in self.projection_name.split(',')]
        fname = ''.join(names)
        
        for k in keys:
            n = '_'.join(k.split('_')[:-1])
            if n == fname:
               sol = soldict[k]
               solr = sol[0]
               soli = sol[1] if len(sol) > 1 else None
               self.add_variables(variables, names, solr, soli, mesh)
               
            
class DerivedBdrValue(PostProcessBase, Vtable_mixin):
    pass


    

