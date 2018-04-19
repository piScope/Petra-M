import os
import numpy as np

from petram.model import Model
from .solver_model import Solver

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints("TimeDomainSolver")
rprint = debug.regular_print('StdSolver')

class TimeDomain(Solver):
    can_delete = True
    has_2nd_panel = False

    def attribute_set(self, v):
        v['st_et_nt'] = [0, 1, 0.1]
        v['time_step'] = 0.01
        v['ts_method'] = "Backward Eular"
        v['abe_minstep']= 0.01
        v['abe_maxstep']= 1.0
        
        super(TimeDomain, self).attribute_set(v)
        return v
    
    def panel1_param(self):
        elp_be =  [["dt", "", 0, {}],]
        elp_abe =  [["min. dt", "", 0, {}],
                    ["max. dt", "", 0, {}],]                
        return [["Initial value setting",   self.init_setting,  0, {},],
                ["physics model",   self.phys_model,  0, {},],
                ["start/end/delta time ",  "",  0, {},],
                ["probes",   self.probe,  0, {},],                                
                [None, None, 34, ({'text':'method','choices': ["Backward Eular", "Adaptive BE"], 'call_fit':False},
                                  {'elp':elp_be},
                                  {'elp':elp_abe},)],                                 
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
        st_et_nt = ", ".join([str(x) for x in self.st_et_nt])
        return (self.init_setting,
                self.phys_model,
                st_et_nt,
                self.probe,                                
                [self.ts_method,
                 [str(self.time_step),],
                 [str(self.abe_minstep), str(self.abe_maxstep),],
                ],
                self.clear_wdir,
                self.init_only,               
                self.assemble_real,
                self.save_parmesh,
                self.use_profiler,)

    
    def import_panel1_value(self, v):
        self.init_setting = str(v[0])        
        self.phys_model = str(v[1])
        tmp = str(v[2]).split(',')
        st_et_nt = [tmp[0], tmp[1], ",".join(tmp[2:])]
        self.st_et_nt = [eval(x) for x in st_et_nt]
        self.probe = str(v[3])
        self.clear_wdir = v[5]
        self.init_only = v[7]        
        self.assemble_real = v[7]
        self.save_parmesh = v[8]
        self.use_profiler = v[9]
        
        self.ts_method = str(v[4][0])
        self.time_step= float(v[4][1][0])
        self.abe_minstep= float(v[4][2][0])
        self.abe_maxstep= float(v[4][2][1])                

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
    
    def get_matrix_weight(self, timestep_config, timestep_weight):
        dt = float(self.time_step)
        lns = self.engine.model['General']._global_ns.copy()
        lns['dt'] = dt

        wt = [eval(x, lns) for x in timestep_weight]
        return wt

    @debug.use_profiler
    def run(self, engine):
        if self.clear_wdir:
            engine.remove_solfiles()

        st, et, nt = self.st_et_nt
        
        if self.ts_method == 'Backward Eular':
            instance = FirstOrderBackwardEuler(self, engine)
            instance.set_timestep(self.time_step)
        elif self.ts_method == "Adaptive BE":
            instance = FirstOrderBackwardEulerAT(self, engine)
            instance.set_timestep(self.abe_minstep)
            instance.set_maxtimestep(self.abe_maxstep)
        else:
            assert False, "unknown stepping method: "+ self.ts_method
            
        instance.set_start(st)
        instance.set_end(et)
        instance.set_checkpoint(np.linspace(st, et, nt))


        finished = instance.init(self.init_only)
        
        instance.configure_probes(self.probe)

        while not finished:
            finished = instance.step()

        instance.save_solution(ksol = 0,
                               skip_mesh = False, 
                               mesh_only = False,
                               save_parmesh=self.save_parmesh)
        instance.save_probe()        
        print(debug.format_memory_usage())


from petram.solver.solver_model import TimeDependentSolverInstance

class FirstOrderBackwardEuler(TimeDependentSolverInstance):
    '''
    Fixed time step solver
    '''
    def __init__(self, gui, engine):
        TimeDependentSolverInstance.__init__(self, gui, engine)
        self.pre_assembled = False
        self.assembled = False
        self.counter = 0
        self.icheckpoint = 0
        
    def init(self, init_only=False):
        self.time = self.st
        if self.time == self.et: return True

        self.counter = 0
        self.icheckpoint = 0
        engine = self.engine
                      
        phys_target = self.get_phys()
        num_matrix= self.gui.get_num_matrix(phys_target)
        
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
        
        self.pre_assemble()
        self.assemble()
        A, X, RHS, Ae, B, M, depvars = self.blocks        
        self.sol = X[0]
        
        if init_only:
            self.write_checkpoint_solution()
            return True
        else:
            return False
                      
    def pre_assemble(self):
        engine = self.engine
        phys_target = self.get_phys()
        engine.run_verify_setting(phys_target, self.gui)
        engine.run_assemble_mat(phys_target)
        engine.run_assemble_rhs(phys_target)
        self.pre_assembled = True


    def compute_A(self, M, B, X):
        '''
        M/dt u_1 + K u_1 = M/dt u_0 + b
        '''
        #print "M, B, X", M, B, X
        one_dt = 1/float(self.time_step)
        MM = M[1]*one_dt
        A = M[0]+ M[1]*one_dt
        return A
    
    def compute_rhs(self, M, B, X):
        one_dt = 1/float(self.time_step)
        MM = M[1]*one_dt
        RHS = MM.dot(X[-1]) + B
        return RHS
        
    def assemble(self):
        self.blocks = self.engine.run_assemble_blocks(self.compute_A,
                                                      self.compute_rhs,
                                                      inplace=False)                

        #A, X, RHS, Ae, B, M, depvars = blocks
        self.assembled = True
        
    def step(self):
        engine = self.engine

        if not self.pre_assembled:
            assert False, "pre_assmeble must have been called"
            
        if self.counter == 0:
            A, X, RHS, Ae, B, M, depvars = self.blocks
            AA = engine.finalize_matrix(A, not self.phys_real, format = self.ls_type)
            BB = engine.finalize_rhs([RHS], not self.phys_real, format = self.ls_type)
            self.write_checkpoint_solution()
            self.icheckpoint += 1
        else:
            A, X, RHS, Ae, B, M, depvars = self.blocks                    
            RHS = self.compute_rhs(M, B, [self.sol])
            dprint1("before eliminateBC")                                    
            dprint1(debug.format_memory_usage())
            RHS = engine.eliminateBC(Ae, X[1], RHS)
            RHS = engine.apply_interp(RHS=RHS)            
            BB = engine.finalize_rhs([RHS], not self.phys_real, format = self.ls_type)

        if self.linearsolver is None:
            if self.ls_type.startswith('coo'):
                datatype = 'Z' if (AA.dtype == 'complex') else 'D'
            else:
                datatype = 'D'
            self.linearsolver  = self.linearsolver_model.allocate_solver(datatype)
            self.linearsolver.SetOperator(AA, dist = engine.is_matrix_distributed,
                                          name = depvars)

        solall = self.linearsolver.Mult(BB, case_base=engine.case_base)
        engine.case_base += BB.shape[1]
            
        if not self.phys_real and self.assemble_real:
            assert False, "this has to be debugged (convertion from real to complex)"
            solall = self.linearsolver_model.real_to_complex(solell, A)
            
        self.time = self.time + self.time_step
        self.counter += 1
        
        self.sol = A.reformat_central_mat(solall, 0)

        for p in self.probe:
            p.append_sol(self.sol, self.time)
                
        if self.checkpoint[self.icheckpoint] < self.time:
            self.write_checkpoint_solution()
            self.icheckpoint += 1
            
        dprint1("TimeStep ("+str(self.counter)+ "), t="+str(self.time)+"...done.")
        dprint1(debug.format_memory_usage())        
        return self.time >= self.et

    def write_checkpoint_solution(self):
        dprint1("writing checkpoint t=" + str(self.time) +
                "("+str(self.icheckpoint)+")")        
        od = os.getcwd()
        path = os.path.join(od, 'checkpoint_' + str(self.icheckpoint))
        self.engine.mkdir(path) 
        os.chdir(path)
        self.engine.cleancwd() 
        self.save_solution()
        os.chdir(od)        

class FirstOrderBackwardEulerAT(FirstOrderBackwardEuler):
    def __init__(self, gui, engine):
        FirstOrderBackwardEuler.__init__(self, gui, engine)
        self.linearsolver  = {}
        self.blocks1 = {}
        self.sol1 = None
        self.sol2 = None
        self._time_step1 = 0
        self._time_step2 = -1
        self.maxstep = 0

    def set_maxtimestep(self, dt):
        self.max_timestep = dt
    @property
    def time_step1(self):
        return (self.time_step_base)* 2**self._time_step1
    @property
    def time_step2(self):
        return (self.time_step_base)* 2**self._time_step2
    
    def set_timestep(self, time_step):
        self.time_step = time_step
        self.time_step_base = time_step
        
    def assemble(self,idt=None):
        flag = False
        if idt is None:
            idt = 0
            flag = True
        self.blocks1[idt] = self.engine.run_assemble_blocks(self.compute_A,
                                                            self.compute_rhs,
                                                            inplace=False)        
        if flag:
            self.blocks = self.blocks1[0]
        else:
            self.blocks = None
            
    def step(self):
        dprint1("Entering step", self.time_step1)
        def get_A_BB(mode, sol, recompute_rhs=False):
            if sol is None: sol = self.sol
            idt = self._time_step1 if mode == 0 else self._time_step2
            dt  = self.time_step1 if mode == 0 else self.time_step2
            self.time_step = dt            
            if not idt in self.blocks1:
                self.assemble(idt = idt)
                A, X, RHS, Ae, B, M, depvars = self.blocks1[idt]
                BB = engine.finalize_rhs([RHS], not self.phys_real,
                                         format = self.ls_type, verbose=False)
            else:
                A, X, RHS, Ae, B, M, depvars = self.blocks1[idt]
                if self.counter != 0 or recompute_rhs:
                    # recompute RHS
                    RHS = self.compute_rhs(M, B, [sol])
                    RHS = engine.eliminateBC(Ae, X[1], RHS)
                    RHS = engine.apply_interp(RHS=RHS)
                BB = engine.finalize_rhs([RHS], not self.phys_real,
                                         format = self.ls_type, verbose=False)
            if not idt in self.linearsolver:
                AA = engine.finalize_matrix(A, not self.phys_real,
                                            format = self.ls_type, verbose=False)
                if self.ls_type.startswith('coo'):
                    datatype = 'Z' if (AA.dtype == 'complex') else 'D'
                else:
                    datatype = 'D'
                self.linearsolver[idt]  = self.linearsolver_model.allocate_solver(datatype)
                self.linearsolver[idt].SetOperator(AA,
                                                   dist = engine.is_matrix_distributed,
                                                   name = depvars)
                
            return A, BB
            
        engine = self.engine

        if not self.pre_assembled:
            assert False, "pre_assmeble must have been called"

        A, BB = get_A_BB(0, self.sol1)
        solall = self.linearsolver[self._time_step1].Mult(BB)
        sol1 = A.reformat_central_mat(solall, 0)
        print "check sample1 (0)", [p.current_value(sol1) for p in self.probe]
        
        A, BB = get_A_BB(1, self.sol2)
        solall2 = self.linearsolver[self._time_step2].Mult(BB)
        sol2 = A.reformat_central_mat(solall2, 0)
        print "check sample2 (1)", [p.current_value(sol2) for p in self.probe]
        
        A, BB = get_A_BB(1, sol2, recompute_rhs=True)
        solall2 = self.linearsolver[self._time_step2].Mult(BB)
        sol2 = A.reformat_central_mat(solall2, 0)
        print "check sample2 (1)", [p.current_value(sol2) for p in self.probe]        

        sample1 = np.hstack([p.current_value(sol1) for p in self.probe]).flatten()
        sample2 = np.hstack([p.current_value(sol2) for p in self.probe]).flatten()
        
        from petram.mfem_config import use_parallel
        if use_parallel:
             from petram.helper.mpi_recipes import allgather, allgather_vector
             sample1 = allgather_vector(np.atleast_1d(sample1))
             sample2 = allgather_vector(np.atleast_1d(sample2))
             
        delta=np.mean(np.abs(sample1-sample2)/np.abs(sample1+sample2)*2)
    
        threshold = 0.01
        if delta > threshold:
            dprint1("delta is large ", delta, sample1, sample2)
            if self._time_step1 > 0:
                self._time_step1 -= 1
                self._time_step2 -= 1
                dprint1("next try....", self.time_step1, self.time_step2)
                return False                
            else:
                dprint1("delta may be too large, but restricted by min time step")                
        else:
            dprint1("delta good  ", delta, sample1, sample2)
        
        if not self.phys_real and self.assemble_real:
            assert False, "this has to be debugged (convertion from real to complex)"
            solall = self.linearsolver_model.real_to_complex(solell, A)
            
        self.time = self.time + self.time_step1
        self.counter += 1
        
        self.sol = A.reformat_central_mat(solall, 0)
        self.sol1 = self.sol
        self.sol2 = self.sol        

        for p in self.probe:
            p.append_sol(self.sol, self.time)
                
        if self.checkpoint[self.icheckpoint] < self.time:
            self.write_checkpoint_solution()
            self.icheckpoint += 1

        dprint1("TimeStep ("+str(self.counter)+ "), t="+str(self.time)+"...done.")
        if delta < threshold/4:
            dt_test = (self.time_step_base)* 2**(self._time_step1 + 1)
            if self.max_timestep >  dt_test:
                 self._time_step1 += 1
                 self._time_step2 += 1
                 dprint1("next try....", self.time_step1, self.time_step2)
            else:
                 dprint1("delta is small, but restricted by max time step")
        return self.time >= self.et
                      



