from petram.model import Model
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('Solver')

class Solver(Model):
    def get_phys(self):
        names = self.phys_model.split(',')
        names = [n.strip() for n in names if n.strip() != '']        
        return [self.root()['Phys'][n] for n in names]
    def get_init_setting(self):
        names = self.init_setting.split(',')
        names = [n.strip() for n in names if n.strip() != '']        
        return [self.root()['InitialValue'][n] for n in names]
    
    def assemble(self, engine):
        raise NotImplementedError(
             "you must specify this method in subclass")
        
    def run(self, engine):
        raise NotImplementedError(
             "you must specify this method in subclass")
    
    def postprocess(self, engine):
        raise NotImplementedError(
             "you must specify this method in subclass")

    def set_parameters(self, names, params):
        raise NotImplementedError(
             "you must specify this method in subclass")
    
    def get_matrix_weight(self, timestep_config, timestep_weight):
        raise NotImplementedError(
             "you must specify this method in subclass")
        
    def compute_A_rhs(self, M, B, X):
        '''
        called from an engine to compute linear system from matrices/solutions.
        '''
        raise NotImplementedError(
             "you must specify this method in subclass")
        
    def get_active_solver(self, mm = None):
        for x in self.iter_enabled(): return x

    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('phys', self)


class LinearSolver(Solver):
    def get_phys(self):
        return self.parent.get_phys()
    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('phys', self)

    def create_solver_instance(self, datatype='D'):
        # datatype = S, D, C, Z
        raise NotImplementedError(
             "you must specify this method in subclass")
    def SetOperator(self, opr, dist=False):
        # opr : operator (matrix)
        # dist: disributed matrix or not
        raise NotImplementedError(
             "you must specify this method in subclass")
    def Mult(self, b):
        raise NotImplementedError(
             "you must specify this method in subclass")

class TimeDependentSolver(object):
    def Step(self, b, t, dt):
        pass


class AdaptiveTimeDependentSolver(object):
    def __init__(self, solver_gui):
        Solvers = {}
        self.gui = solver_gui
        
    def Step(self, b, t, dt):
        pass        
    

class TimeSteping(object):
    def __init__(self, ls_type):
        self.ls_type = ls_type
        
    def SetOperator(self, AA):
        self.solver.SetOperator(AA, dist = engine.is_matrix_distributed)

    def EvalBB(self, Ae, X, RHS):
        RHS = self.compute_rhs(B, sol, self.time_step)    
        RHS = engine.eliminateBC(Ae, X[1], RHS)
        BB = engine.finalize_rhs([RHS], not phys_real,
                                      format = self.ls_type)

        
    def Step(engine, solver, t, dt):
        dprint1("A", A)
        dprint1("RHS", RHS)
        dprint1("time", t)
        AA = engine.finalize_matrix(A, not phys_real, format = ls_type)
        BB = engine.finalize_rhs([RHS], not phys_real, format = ls_type)

        datatype = 'Z' if (AA.dtype == 'complex') else 'D'
        Solver = solver.create_solver_instance(datatype)


        counter = 0
        while True:
            #solall = solver.solve(engine, AA, BB)
            dprint1("before multi")
            dprint1(debug.format_memory_usage())

            
            solall = Solver.Mult(BB, case_base=engine.case_base)
            engine.case_base += BB.shape[1]
            
            if not phys_real and self.assemble_real:
                solall = solver.real_to_complex(solell, self.A)
            t = t + dt
            counter += 1
            dprint1("TimeStep ("+str(counter)+ "), t="+str(t))
            if t >= et: break
            #if counter > 5: break
            
            dprint1("before reformat")
            dprint1(debug.format_memory_usage())
            
            sol = A.reformat_central_mat(solall, 0)

            dprint1("after reformat")
            dprint1(debug.format_memory_usage())
            
            if checkpoint[icheckpoint] < t:
                 dprint1("writing checkpoint t=" + str(t) + "("+str(icheckpoint)+")")
                 
                 extra_data = self.store_sol(engine, sol, blocks[1][0], 0)
                 path = os.path.join(od, 'checkpoint_' + str(icheckpoint))
                 engine.mkdir(path) 
                 os.chdir(path)
                 engine.cleancwd() 
                 self.save_solution(engine, extra_data)
                 
                 icheckpoint = icheckpoint+1
            os.chdir(od)
            dprint1("before compute_rhs")                                    
            dprint1(debug.format_memory_usage())                
            
            RHS = self.compute_rhs(B, sol, self.time_step)

            dprint1("before eliminateBC")                                    
            dprint1(debug.format_memory_usage())
            
            RHS = engine.eliminateBC(Ae, X[1], RHS)
            BB = engine.finalize_rhs([RHS], not phys_real, format = ls_type)
            dprint1("end reformat")                                    
            dprint1(debug.format_memory_usage())                

        #PT = P.transpose()
        sol = A.reformat_central_mat(solall, 0)
        
        return sol
           
