from petram.model import Model
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('Solver')

'''

    Solver : Model Tree Object for solvers such as TimeDependent Solver

    SolverInstance: an actual solver logic comes here
       SolverInstance : base class for standard solver
       TimeDependentSolverInstance : an actual solver logic comes here


'''
class SolverBase(Model):
    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('phys', self)

class SolveStep(SolverBase):
    def get_possible_child(self):
        ret = self.parent.get_possible_child()
        # last element is SolveStep, 
        return ret[:-1]
    
    def get_phys(self):
        phys_root = self.root()['Phys']        
        ret = []
        for k in self.keys():
            for x in self[k].get_target_phys():
                if not x in ret: ret.append(x)
        return ret
    
    def get_active_solvers(self):
        return [x for x in self.iter_enabled()]

    def run(self, engine, is_first = True):
        solvers = self.get_active_solvers()
        
        is_first = True
        for solver in solvers:
             solver.run(engine, is_first=is_first)        
             is_first = False
             engine.add_FESvariable_to_NS(self.get_phys()) 
             engine.store_x()
        
        
class Solver(SolverBase):
    def attribute_set(self, v):
        v['clear_wdir'] = False
        v['init_only'] = False   
        v['assemble_real'] = False
        v['save_parmesh'] = False        
        v['phys_model']   = ''
        v['init_setting']   = ''
        v['use_profiler'] = False
        v['probe'] = ''
        super(Solver, self).attribute_set(v)
        return v
    
    def get_phys(self):
        return self.parent.get_phys()

    def get_target_phys(self):
        names = self.phys_model.split(',')
        names = [n.strip() for n in names if n.strip() != '']        
        return [self.root()['Phys'][n] for n in names]
    
    def is_complex(self):
        phys = self.get_phys()
        is_complex = any([p.is_complex() for p in phys])
        if self.assemble_real: return False        
        #if is_complex: return True
        return is_complex
    
    def is_converted_from_complex(self):
        phys = self.get_phys()
        is_complex = any([p.is_complex() for p in phys])
        if is_complex and self.assemble_real: return True
        return False
        
    def get_init_setting(self):
        names = self.init_setting.split(',')
        names = [n.strip() for n in names if n.strip() != '']        
        return [self.root()['InitialValue'][n] for n in names]
    
    def get_active_solver(self, mm = None):
        for x in self.iter_enabled():
            if isinstance(x, LinearSolverModel):
               return x

    def get_num_matrix(self, phys_target=None):
        solver_root = self.get_solve_root()
        num = []
        for k in solver_root.keys():
            mm = solver_root[k]
            if not mm.enabled: continue
            num.append(self.root()['Phys'].get_num_matrix(mm.get_matrix_weight,
                                           phys_target))
        return max(num)
    
    def prepare_form_sol_variables(self, engine):
        phys_target = self.get_phys()
        num_matrix= self.get_num_matrix(phys_target)
        
        engine.set_formblocks(phys_target, num_matrix)
        
        for p in phys_target:
            engine.run_mesh_extension(p)
            
        engine.run_alloc_sol(phys_target)
#        engine.run_fill_X_block()

    def get_solve_root(self):
        obj = self
        solver_root = self.root()['Solver']

        while (not isinstance(obj, SolveStep) and
               obj is not solver_root):
            obj = obj.parent
        return obj
    
    def get_matrix_weight(self):
        raise NotImplementedError(
             "you must specify this method in subclass")
    
    def run(self, engine, is_first = True):        
        raise NotImplementedError(
             "you must specify this method in subclass")
    

class SolverInstance(object):
    '''
    Solver instance is where the logic of solving linear system
    (time stepping, adaptation, non-linear...) is written.

    It is not a model object. SolverModel will generate this
    instance to do the actual solve step.
    '''
    def __init__(self, gui, engine):
        self.gui = gui
        self.engine = engine
        self.sol = None
        self.linearsolver_model= None # LinearSolverModel
        self.linearsolver = None      # Actual LinearSolver        
        self.probe = []
        self.linearsolver_model = None
        self.phys_real = True
        self.ls_type = ''
        
        
        self.set_linearsolver_model()
        
    def get_phys(self):
        return self.gui.get_phys()

    def get_target_phys(self):
        return self.gui.get_target_phys()
    
    @property
    def blocks(self):
        return self.engine.assembled_blocks
        
    def get_init_setting(self):

        names = self.gui.init_setting.split(',')
        names = [n.strip() for n in names if n.strip() != '']
        
        root = self.engine.model        
        return [root['InitialValue'][n] for n in names]

    def set_fes_mask(self):
        # mask defines which FESspace will be solved by
        # a linear solver.
        all_phys = self.get_phys()        
        phys_target = self.get_target_phys()
        mask = self.engine.get_block_mask(all_phys, phys_target)
        self.fes_mask = mask

    def save_solution(self, ksol = 0, skip_mesh = False, 
                      mesh_only = False, save_parmesh=False):
                      
        engine = self.engine
        phys_target = self.get_phys()

        sol, sol_extra = engine.split_sol_array(self.sol)

        engine.recover_sol(sol)
        extra_data = engine.process_extra(sol_extra)


        engine.save_sol_to_file(phys_target, 
                                skip_mesh = skip_mesh,
                                mesh_only = mesh_only,
                                save_parmesh = save_parmesh)
        if mesh_only: return
        engine.save_extra_to_file(extra_data)
        #engine.is_initialzied = False
        
        
    def save_probe(self):
        for p in self.probe:
            p.write_file()
        
    def set_linearsolver_model(self):
        solver = self.gui.get_active_solver()      
        phys_target = self.get_phys()
        
        self.linearsolver_model = solver
        self.phys_real = all([not p.is_complex() for p in phys_target])        
        self.ls_type = solver.linear_system_type(self.gui.assemble_real,
                                                 self.phys_real)

    def configure_probes(self, probe_txt):
        from petram.sol.probe import Probe
        dprint1("configure probes: "+probe_txt)
        if probe_txt.strip() != '':
            probe_names = [x.strip() for x in probe_txt.split(',')]
            probe_idx =  [self.engine.dep_var_offset(n) for n in probe_names]
            for n, i in zip(probe_names, probe_idx):
                self.probe.append(Probe(n, i))

    def allocate_linearsolver(self, is_complex):
        if self.ls_type.startswith('coo'):
            datatype = 'Z' if (is_complex) else 'D'
        else:
            datatype = 'D'
            
        linearsolver  = self.linearsolver_model.allocate_solver(datatype)
        return linearsolver
        
    def solve(self):
        raise NotImplementedError(
             "you must specify this method in subclass")

    def compute_rhs(self, M, B, X):
        raise NotImplementedError(
             "you must specify this method in subclass")

    def compute_A(self, M, B, X, mask_M, mask_B):
        # must return A and isAnew        
        raise NotImplementedError(
             "you must specify this method in subclass")
    
class TimeDependentSolverInstance(SolverInstance):
    def __init__(self, gui, engine):
        self.st = 0.0
        self.et = 1.0
        self.checkpoint = [0, 0.5, 1.0]
        self._time = 0.0
        self.child_instance = []
        SolverInstance.__init__(self, gui, engine)

    @property
    def time(self):
        return self._time
    
    @time.setter
    def time(self, value):
        self._time = value
        self.engine.model['General']._global_ns['t']=value
        
    def set_start(self, st):
        self.st = st
        self.time = st
        
    def set_end(self, et):
        self.et = et

    def set_timestep(self, time_step):
        self.time_step = time_step
        
    def set_checkpoint(self, checkpoint):
        self.checkpoint = checkpoint

    def add_child_instance(self, instance):
        self.child_instance.append(instance)
        
    def solve(self):
        assert False, "time dependent solver does not have solve method. call step"
        
    def step(self):
        raise NotImplementedError(
             "you must specify this method in subclass")

'''

    LinearSolverModel : Model Tree Object for linear solver
    LinearSolver : an interface to actual solver


'''    
class LinearSolverModel(SolverBase):
    is_iterative = True    
    def get_phys(self):
        return self.parent.get_phys()

    def linear_system_type(self, assemble_real, phys_real):
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
        raise NotImplementedError(
             "you must specify this method in subclass")
    def allocate_solver(self, datatype='D'):
        # datatype = S, D, C, Z
        raise NotImplementedError(
             "you must specify this method in subclass")

    def real_to_complex(self, solall, M=None):
        # method called when real value solver is used for complex value problem
        raise NotImplementedError(
             "you must specify this method in subclass")
        
class LinearSolver(object):
    '''
    LinearSolver is an interface to linear solvers such as MUMPS.
    It is generated by SolverInstan.
    '''
    is_iterative = True        
    def __init__(self, gui):
        self.gui = gui

    def SetOperator(self, opr, dist=False, name=None):
        # opr : operator (matrix)
        # dist: disributed matrix or not
        # name: name of variables in block operator
        raise NotImplementedError(
             "you must specify this method in subclass")
    def Mult(self, b, case_base=0):
        raise NotImplementedError(
             "you must specify this method in subclass")

    
