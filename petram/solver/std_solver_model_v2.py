import os
import numpy as np

from petram.model import Model
from .solver_model import Solver
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('StdSolver')
rprint = debug.regular_print('StdSolver')

class StdSolver(Solver):
    can_delete = True
    has_2nd_panel = False

    def attribute_set(self, v):
        v['clear_wdir'] = False
        v['init_only'] = False   
        v['assemble_real'] = False
        v['save_parmesh'] = False        
        v['phys_model']   = ''
        v['init_setting']   = ''        
        super(StdSolver, self).attribute_set(v)
        return v
    
    def panel1_param(self):
        return [["Initial value setting",   self.init_setting,  0, {},],
                ["physics model",   self.phys_model,  0, {},],
                ["clear working directory",
                 self.clear_wdir,  3, {"text":""}],
                ["initialize solution only",
                 self.init_only,  3, {"text":""}], 
                ["convert to real matrix (complex prob.)",
                 self.assemble_real,  3, {"text":""}],
                ["save parallel mesh",
                 self.save_parmesh,  3, {"text":""}],]

    def get_panel1_value(self):
        return (self.init_setting,
                self.phys_model,
                self.clear_wdir,
                self.init_only,               
                self.assemble_real,
                self.save_parmesh)    
    
    def import_panel1_value(self, v):
        self.init_setting = str(v[0])        
        self.phys_model = str(v[1])
        self.clear_wdir = v[2]
        self.init_only = v[3]        
        self.assemble_real = v[4]
        self.save_parmesh = v[5]        

    def get_editor_menus(self):
        return []
#        return [("Assemble",  self.OnAssemble, None),
#                ("Update RHS",  self.OnUpdateRHS, None),
#                ("Run Solve Step",  self.OnRunSolve, None),]

    def get_phys(self):
        names = self.phys_model.split(',')
        names = [n.strip() for n in names if n.strip() != '']        
        return [self.root()['Phys'][n] for n in names]
    def get_init_setting(self):
        names = self.init_setting.split(',')
        names = [n.strip() for n in names if n.strip() != '']        
        return [self.root()['InitialValue'][n] for n in names]
    
    '''
    This interactive are mostly for debug purpose
    '''
    def OnAssemble(self, evt):
        '''
        assemble linear system interactively (local matrix)
        '''
        dlg = evt.GetEventObject()       
        viewer = dlg.GetParent()
        engine = viewer.engine

        self.assemble(engine)
        self.generate_linear_system(engine)
        evt.Skip()

    def OnUpdateRHS(self, evt):
        dlg = evt.GetEventObject()       
        viewer = dlg.GetParent()
        engine = viewer.engine
        phys = self.get_phys()[0]

        r_B, i_B, extra, r_x, i_x = engine.assemble_rhs(phys, self.is_complex)
        B = engine.generate_rhs(r_B, i_B, extra, r_x, i_x, self.P, format = self.ls_type)
        self.B = [B]
        evt.Skip()

    def OnRunSolve(self, evt):
        dlg = evt.GetEventObject()       
        viewer = dlg.GetParent()
        engine = viewer.engine

        self.call_solver(engine)
        self.postprocess(engine)

    def get_possible_child(self):
        choice = []
        try:
            from petram.solver.mumps_model import MUMPS
            choice.append(MUMPS)
        except ImportError:
            pass

        try:
            from petram.solver.gmres_model import GMRES
            choice.append(GMRES)
        except ImportError:
            pass

        try:
            from petram.solver.strumpack_model import SpSparse
            choice.append(SpSparse)
        except ImportError:
            pass
        return choice
    
    def init_sol(self, engine):
        phys_targets = self.get_phys()
        
        engine.collect_dependent_vars(phys_targets)        
        for p in phys_targets:
            engine.run_mesh_extension(p)
            
        inits = self.get_init_setting()
        if len(inits) == 0:
            # in this case alloate all fespace and initialize all
            # to zero
            engine.run_alloc_sol(phys_targets)
            engine.run_apply_init(phys_targets, 0)
        else:
            for init in inits:
                init.run(engine)
        engine.run_apply_essential(phys_targets)
        return 


    def assemble(self, engine):
        phys_targets = self.get_phys()
        engine.run_verify_setting(phys_targets, self)
        matvecs, matvecs_c = engine.run_assemble(phys_targets, nterms=1)
        return matvecs, matvecs_c

    def generate_linear_system(self, engine, matvecs, matvecs_c):
        phys_target = self.get_phys()
        solver = self.get_active_solver()

        blocks = engine.generate_linear_system(phys_target,
                                            matvecs, matvecs_c)
        # P: projection,  M:matrix, B: rhs, S: extra_flag
        self.M, B, self.Me = blocks
        self.B = [B]

    def store_rhs(self, engine):
        phys_targets = self.get_phys()
        vecs, vecs_c = engine.run_assemble_rhs(phys_targets)
        blocks = engine.generate_rhs(phys_targets, vecs, vecs_c)
        self.B.append(blocks[1])

    def call_solver(self, engine):
        solver = self.get_active_solver()        
        phys_targets = self.get_phys()        
        phys_real = all([not p.is_complex() for p in phys_targets])
        ls_type = solver.linear_system_type(self.assemble_real,
                                            phys_real)
        '''
        ls_type: coo  (matrix in coo format : DMUMP or ZMUMPS)
                 coo_real  (matrix in coo format converted from complex 
                            matrix : DMUMPS)
                 # below is a plan...
                 blk (matrix made mfem:block operator)
                 blk_real (matrix made mfem:block operator for complex
                             problem)
                          (unknowns are in the order of  R_fes1, R_fes2,... I_fes1, Ifes2...)
                 blk_interleave (unknowns are in the order of  R_fes1, I_fes1, R_fes2, I_fes2,...)
                 None(not supported)
        '''
        if debug.debug_memory:
            dprint1("Block Matrix before shring :\n",  self.M)
            dprint1(debug.format_memory_usage())                
        M_block, B_blocks, P = engine.eliminate_and_shrink(self.M,
                                                           self.B, self.Me)
        
        if debug.debug_memory:
            dprint1("Block Matrix after shrink :\n",  M_block)
            dprint1(debug.format_memory_usage())        
        M, B = engine.finalize_linearsystem(M_block, B_blocks,
                                            not phys_real,
                                            format = ls_type)
        solall = solver.solve(engine, M, B)
        #solall = np.zeros((M.shape[0], len(B_blocks))) # this will make fake data to skip solve step
        
        #if ls_type.endswith('_real'):
        if not phys_real and self.assemble_real:
            solall = solver.real_to_complex(solall, M)
        PT = P.transpose()

        return solall, PT

    def store_sol(self, engine, matvecs, solall, PT, ksol = 0):
        phys_targets = self.get_phys()

        sol = PT.reformat_central_mat(solall, ksol)
        sol = PT.dot(sol)

        dprint1(sol)
        l = len(self.B)

        sol, sol_extra = engine.split_sol_array(phys_targets, sol)


        # sol_extra = engine.gather_extra(sol_extra)                

        phys_target = self.get_phys()
        engine.recover_sol(phys_target, matvecs, sol)
        extra_data = engine.process_extra(phys_target, sol_extra)

        return extra_data
            
    def free_matrix(self):
        self.P = None
        self.M = None
        self.B = None

    def save_solution(self, engine, extra_data, 
                      skip_mesh = False, 
                      mesh_only = False):
        phys_target = self.get_phys()
        engine.save_sol_to_file(phys_target, 
                                skip_mesh = skip_mesh,
                                mesh_only = mesh_only,
                                save_parmesh = self.save_parmesh)
        if mesh_only: return
        engine.save_extra_to_file(extra_data)
        engine.is_initialzied = False
        
    def run(self, engine):
        phys_target = self.get_phys()
        if self.clear_wdir:
            engine.remove_solfiles()
        if not engine.isInitialized: self.init_sol(engine)
        if self.init_only:
            extra_data = None
        else:
            matvecs, matvecs_c = self.assemble(engine)
            self.generate_linear_system(engine, matvecs, matvecs_c)
            solall, PT = self.call_solver(engine)
            extra_data = self.store_sol(engine, matvecs, solall, PT, 0)
            dprint1("Extra Data", extra_data)
            
        engine.remove_solfiles()
        dprint1("writing sol files")
        self.save_solution(engine, extra_data)

        print(debug.format_memory_usage())
           



