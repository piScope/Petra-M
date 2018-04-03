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
        v['use_profiler'] = False
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
                 self.save_parmesh,  3, {"text":""}],
                ["use cProfiler",
                 self.use_profiler,  3, {"text":""}],]

    def get_panel1_value(self):
        return (self.init_setting,
                self.phys_model,
                self.clear_wdir,
                self.init_only,               
                self.assemble_real,
                self.save_parmesh,
                self.use_profiler)        
    
    def import_panel1_value(self, v):
        self.init_setting = str(v[0])        
        self.phys_model = str(v[1])
        self.clear_wdir = v[2]
        self.init_only = v[3]        
        self.assemble_real = v[4]
        self.save_parmesh = v[5]
        self.use_profiler = v[6]                

    def get_editor_menus(self):
        return []
#        return [("Assemble",  self.OnAssemble, None),
#                ("Update RHS",  self.OnUpdateRHS, None),
#                ("Run Solve Step",  self.OnRunSolve, None),]

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
        phys_target = self.get_phys()
        num_matrix= engine.run_set_matrix_weight(phys_target, self)
        
        engine.set_formblocks(phys_target, num_matrix)

        for p in phys_target:
            engine.run_mesh_extension(p)
            
        engine.run_alloc_sol(phys_target)
        
        inits = self.get_init_setting()
        if len(inits) == 0:
            # in this case alloate all fespace and initialize all
            # to zero
            engine.run_apply_init(phys_target, 0)
        else:
            for init in inits:
                init.run(engine)
        engine.run_apply_essential(phys_target)
        return 

    def get_matrix_weight(self, timestep_config, timestep_weight):
        return [1, 0, 0]

    def compute_A_rhs(self, M, B, X):
        '''
        M[0] x = B
        '''
        RHS = B
        return M[0], RHS

    def assemble(self, engine):
        phys_target = self.get_phys()
        engine.run_verify_setting(phys_target, self)
        engine.run_assemble_mat(phys_target)
        engine.run_assemble_rhs(phys_target)
        blocks = engine.run_assemble_blocks(self)
        A, X, RHS, Ae, B = blocks

        self.A   = A
        self.RHS = [RHS]
        return blocks # A, X, RHS, Ae, B

    '''
    def generate_linear_system(self, engine, blocks)
        phys_target = self.get_phys()
        solver = self.get_active_solver()

        blocks = engine.generate_linear_system(phys_target, blocks)
        
        # P: projection,  M:matrix, B: rhs, S: extra_flag
        self.M, B, self.Me = blocks
        self.B = [B]
    '''
    def store_rhs(self, engine):
        phys_target = self.get_phys()
        vecs, vecs_c = engine.run_assemble_rhs(phys_target)
        blocks = engine.generate_rhs(phys_target, vecs, vecs_c)
        self.B.append(blocks[1])

    def call_solver(self, engine, blocks):
        A, X, RHS, Ae, B = blocks
        
        solver = self.get_active_solver()        
        phys_target = self.get_phys()        
        phys_real = all([not p.is_complex() for p in phys_target])
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
        #if debug.debug_memory:
        #    dprint1("Block Matrix before shring :\n",  self.M)
        #    dprint1(debug.format_memory_usage())                
        #M_block, B_blocks, P = engine.eliminate_and_shrink(self.M,
        #                                                   self.B, self.Me)
        
        if debug.debug_memory:
            dprint1("Block Matrix after shrink :\n",  M_block)
            dprint1(debug.format_memory_usage())

        dprint1("A", self.A, self.A.format_nnz())
        dprint1("RHS", self.RHS)
        
        AA = engine.finalize_matrix(self.A, not phys_real, format = ls_type)
        BB = engine.finalize_rhs(self.RHS, not phys_real, format = ls_type)

        solall = solver.solve(engine, AA, BB)
        #solall = np.zeros((M.shape[0], len(B_blocks))) # this will make fake data to skip solve step
        
        #if ls_type.endswith('_real'):
        if not phys_real and self.assemble_real:
            solall = solver.real_to_complex(solall, self.A)
        #PT = P.transpose()

        return solall

    def store_sol(self, engine, solall, X, ksol = 0):
        sol = self.A.reformat_central_mat(solall, ksol)
        l = len(self.RHS)

        sol, sol_extra = engine.split_sol_array(sol)


        # sol_extra = engine.gather_extra(sol_extra)                

        phys_target = self.get_phys()
        engine.recover_sol(sol)
        extra_data = engine.process_extra(sol_extra)

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
        if self.use_profiler:
            import cProfile, pstats, StringIO
            pr = cProfile.Profile()
            pr.enable()        
        phys_target = self.get_phys()
        if self.clear_wdir:
            engine.remove_solfiles()
        if not engine.isInitialized: self.init_sol(engine)
        if self.init_only:
            extra_data = None
        else:
            blocks = self.assemble(engine)
            #self.generate_linear_system(blocks)
            solall = self.call_solver(engine, blocks)
            extra_data = self.store_sol(engine, solall, blocks[1][0], 0)
            dprint1("Extra Data", extra_data)
            
        engine.remove_solfiles()
        dprint1("writing sol files")
        self.save_solution(engine, extra_data)
        
        if self.use_profiler:
            pr.disable()
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print s.getvalue()
            
        print(debug.format_memory_usage())
           



