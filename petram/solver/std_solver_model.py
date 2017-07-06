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
        v['init_only'] = False   
        v['assemble_real'] = False
        v['phys_model']   = ''
        super(StdSolver, self).attribute_set(v)
        return v
    
    def panel1_param(self):
        return [["physics model",   self.phys_model,  0, {},],
                ["assemble complex \nas real problem",
                 self.assemble_real,  3, {"text":""}],
                ["initialize solution only",
                 self.init_only,  3, {"text":""}],    ]     

    def get_panel1_value(self):
        return (self.phys_model,
                self.assemble_real,
                self.init_only)    

    def get_editor_menus(self):
        return []
#        return [("Assemble",  self.OnAssemble, None),
#                ("Update RHS",  self.OnUpdateRHS, None),
#                ("Run Solve Step",  self.OnRunSolve, None),]

    def get_phys(self):
        names = self.phys_model.split(',')
        names = [n.strip() for n in names]        
        return [self.root()['Phys'][n] for n in names]
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

    def import_panel1_value(self, v):
        self.phys_model = str(v[0])
        self.assemble_real = v[1]

    def get_possible_child(self):
        from petram.solver.mumps_model import MUMPS
        from petram.solver.gmres_model import GMRES
        return [MUMPS, GMRES]
    
    def init_sol(self, engine):
        phys_targets = self.get_phys()
        engine.run_init_sol(phys_targets)
        return 

    def assemble(self, engine):
        phys_targets = self.get_phys()
        matvecs, matvecs_c = engine.run_assemble(phys_targets)
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
                 block (matrix made mfem:block operator)
                 block_real (matrix made mfem:block operator for complex
                             problem)
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
        
        if ls_type.endswith('_real'):
            s = sallall.shape[1]
            solall = sol[:, :s/2] + 1j*solall[:,s/2:]


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
                                mesh_only = mesh_only)
        if mesh_only: return
        engine.save_extra_to_file(extra_data)
        engine.is_initialzied = False
        
    def run(self, engine):
        phys_target = self.get_phys()
        if not engine.isInitialized: self.init_sol(engine)
        if self.init_only:
            return
        matvecs, matvecs_c = self.assemble(engine)
        self.generate_linear_system(engine, matvecs, matvecs_c)
        solall, PT = self.call_solver(engine)
        extra_data = self.store_sol(engine, matvecs, solall, PT, 0)
        dprint1("Extra Data", extra_data)
        self.save_solution(engine, extra_data)

        rprint(debug.format_memory_usage())
           



