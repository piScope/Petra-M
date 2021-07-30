from __future__ import print_function

import os
import numpy as np

from petram.model import Model
from .solver_model import Solver

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints("TimeDomainSolver")
rprint = debug.regular_print('TimeDependentSolver')


from petram.solver.std_solver_model import StdSolver
class DerivedValue(StdSolver):
    def allocate_instance(self, engine):
        from petram.solver.std_solver_model import StandardSolver        
        instance = StandardSolver(self, engine)
        instance.set_blk_mask()        
        return instance
    
class TimeStep():
    def __init__(self, data):
        self.data = list(np.atleast_1d(data))
        
    def __call__(self, i):
        if i >= len(self.data):
            return self.data[-1]
        else:
            return self.data[i]        
    
class TimeDomain(Solver):
    can_delete = True
    has_2nd_panel = False

    def attribute_set(self, v):
        v['st_et_nt'] = [0, 1, 5]
        v['time_step'] = 0.01
        v['time_step_cnk'] = 0.01
        v['time_step_fe'] = 0.01                
        v['ts_method'] = "Backward Euler"
        v['abe_minstep']= 0.01
        v['abe_maxstep']= 1.0
        v['use_dwc_cp']   = False   # check point
        v['dwc_cp_name']   = ''              
        v['dwc_cp_arg']   = ''      
        v['use_dwc_ts']   = False   # every time step
        v['dwc_ts_name']   = ''                      
        v['dwc_ts_arg']   = ''      
        
        super(TimeDomain, self).attribute_set(v)
        return v
    
    def panel1_param(self):
        elp_be =  [["dt", "", 0, {}],]
        elp_abe =  [["min. dt", "", 0, {}],
                    ["max. dt", "", 0, {}],]
        ret_cp = [["dwc",   self.dwc_cp_name,   0, {}],
                  ["args.",   self.dwc_cp_arg,   0, {}],]
        value_cp = [self.dwc_cp_name, self.dwc_cp_arg]
        ret_ts = [["dwc",   self.dwc_ts_name,   0, {}],
                  ["args.",   self.dwc_ts_arg,   0, {}],]
        value_ts = [self.dwc_ts_name, self.dwc_ts_arg]
        
        return [#["Initial value setting",   self.init_setting,  0, {},],
                ["physics model",   self.phys_model,  0, {},],
                ["start/end/#step",  "",  0, {},],
                ["probes",   self.probe,  0, {},],                                
                [None, None, 34, ({'text':'method','choices': ["Backward Euler",
                                                               "CrankNicolson",
                                                               "Forward Euler",
                                                               "Adaptive BE"], 'call_fit':False},
                                  {'elp':elp_be},
                                  {'elp':elp_be},
                                  {'elp':elp_be},                                                                    
                                  {'elp':elp_abe},)],
                [None, [False, value_cp], 27, [{'text':'Use DWC (check point)'},
                                              {'elp': ret_cp}]],
                [None, [False, value_ts], 27, [{'text':'Use DWC (time stepping)'},
                                              {'elp': ret_ts}]],
                [None,
                 self.clear_wdir,  3, {"text":"clear working directory"}],
                [None,
                 self.init_only,  3, {"text":"initialize solution only"}], 
                [None,
                 self.assemble_real,  3, {"text":"convert to real matrix (complex prob.)"}],
                [None,
                 self.save_parmesh,  3, {"text":"save parallel mesh"}],
                [None,
                 self.use_profiler,  3, {"text":"use profiler"}],]

    def get_panel1_value(self):
        st_et_nt = ", ".join([str(x) for x in self.st_et_nt])
        return (#self.init_setting,
                self.phys_model,
                st_et_nt,
                self.probe,                                
                [self.ts_method,
                 [str(self.time_step),],
                 [str(self.time_step_cnk),],
                 [str(self.time_step_fe),],                                  
                 [str(self.abe_minstep), str(self.abe_maxstep),],
                ],
                [self.use_dwc_cp, [self.dwc_cp_name, self.dwc_cp_arg,]],
                [self.use_dwc_ts, [self.dwc_ts_name, self.dwc_ts_arg,]],            
                self.clear_wdir,
                self.init_only,               
                self.assemble_real,
                self.save_parmesh,
                self.use_profiler,)

    
    def import_panel1_value(self, v):
        #self.init_setting = str(v[0])        
        self.phys_model = str(v[0])
        tmp = str(v[1]).split(',')
        st_et_nt = [tmp[0], tmp[1], ",".join(tmp[2:])]
        self.st_et_nt = [eval(x) for x in st_et_nt]
        self.probe = str(v[2])
        self.clear_wdir = v[6]
        self.init_only = v[7]        
        self.assemble_real = v[8]
        self.save_parmesh = v[9]
        self.use_profiler = v[10]
        
        self.ts_method = str(v[3][0])
        self.time_step = str(v[3][1][0])
        self.time_step_cnk = str(v[3][2][0])
        self.time_step_fe = str(v[3][3][0])                
        self.abe_minstep  = float(v[3][4][0])
        self.abe_maxstep  = float(v[3][4][1])
        self.use_dwc_cp   = v[4][0]
        self.dwc_cp_name  = v[4][1][0]                           
        self.dwc_cp_arg   = v[4][1][1]         
        self.use_dwc_ts   = v[5][0]
        self.dwc_ts_name  = v[5][1][0]
        self.dwc_ts_arg   = v[5][1][1]                           
        

    def get_possible_child(self):
        choice = []
        try:
            from petram.solver.mumps_model import MUMPS
            choice.append(MUMPS)
        except ImportError:
            pass

        try:
            from petram.solver.iterative_model import Iterative
            choice.append(Iterative)
        except ImportError:
            pass

        try:
            from petram.solver.strumpack_model import SpSparse
            choice.append(SpSparse)
        except ImportError:
            pass
        choice.append(DerivedValue)
        return choice
    
    def get_matrix_weight(self, timestep_config):
        #timestep_weight):
        #dt = float(self.time_step)
        #lns = self.root()['General']._global_ns.copy()
        #lns['dt'] = dt
        wt = [1 if x else 0 for x in timestep_config]
        return wt

    def get_child_solver(self):
        return self.derived_value_solver()
    
    def derived_value_solver(self):
        return [self[key] for key in self
                if isinstance(self[key],DerivedValue) and self[key].enabled]
    
    @debug.use_profiler
    def run(self, engine, is_first=True):
        if self.clear_wdir:
            engine.remove_solfiles()

        fid = engine.open_file('checkpoint.'+self.parent.name()+'_'+self.name()+'.txt', 'w')
        st, et, nt = self.st_et_nt
        
        if self.ts_method == 'Backward Euler' or self.ts_method == 'Backward Eular':
            instance = FirstOrderBackwardEuler(self, engine)
            time_step = self.eval_text_in_global(self.time_step)
            dprint1("time step configuration: " + str(self.time_step) + ':' +  str(time_step))            
            instance.set_timestep(TimeStep(time_step))
            
        elif self.ts_method == "CrankNicolson":
            instance = CrankNicolson(self, engine)
            time_step = self.eval_text_in_global(self.time_step_cnk)
            dprint1("time step configuration: " + str(self.time_step_cnk) + ':' +  str(time_step))            
            instance.set_timestep(TimeStep(time_step))
            
        elif self.ts_method == "Forward Euler":
            instance = FirstOrderForwardEuler(self, engine)            
            time_step = self.eval_text_in_global(self.time_step_fe)
            dprint1("time step configuration: " + str(self.time_step_fe) + ':' +  str(time_step))            
            instance.set_timestep(TimeStep(time_step))
            
        elif self.ts_method == "Adaptive BE":
            instance = FirstOrderBackwardEulerAT(self, engine)
            instance.set_timestep(self.abe_minstep)
            instance.set_maxtimestep(self.abe_maxstep)
        else:
            assert False, "unknown stepping method: "+ self.ts_method
            
        instance.set_start(st)
        instance.set_end(et)
        instance.set_checkpoint(np.linspace(st, et, nt))

        engine.sol = engine.assembled_blocks[1][0]
        instance.sol = engine.sol
        instance.time = st
        
        if self.init_only:
            instance.write_checkpoint_solution()
        
        else:
            if is_first:
                instance.pre_assemble()                                
                instance.assemble()
                
            #instance.solve()            
            #if is_first:
            #self.prepare_form_sol_variables(engine)
            #finished = instance.init(self.init_only)
            
            instance.set_blk_mask()                        
            instance.configure_probes(self.probe)
        
            for solver in self.derived_value_solver():
                child = solver.allocate_instance(engine)
                instance.add_child_instance(child)
                
            finished = False
            if fid is not None:
                fid.write(str(0)+':'+str(instance.time)+"\n")
            while not finished:
                finished, cp_written = instance.step(is_first)
                is_first=False
                
                if self.use_dwc_ts:
                    engine.call_dwc(self.get_phys_range(),
                                    method="timestep",
                                    callername = self.name(),
                                    dwcname = self.dwc_ts_name,
                                    args = self.dwc_ts_arg,
                                    time = instance.time)
                if self.use_dwc_cp and cp_written:                
                    engine.call_dwc(self.get_phys_range(),
                                    method="checkpoint",
                                    callername = self.name(),
                                    dwcname = self.dwc_cp_name, 
                                    args = self.dwc_cp_arg,
                                    time = instance.time,
                                    icheckpoint = instance.icheckpoint-1)
                if cp_written:
                    instance.save_probe()
                    if fid is not None:
                        fid.write(str(instance.icheckpoint-1)+':'+str(instance.time)+"\n")
                        fid.flush()
                        
        instance.save_solution(ksol = 0,
                               skip_mesh = False, 
                               mesh_only = False,
                               save_parmesh=self.save_parmesh)
        instance.save_probe()
        if fid is not None: fid.close()
        
        return is_first       

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
        self._dt_used_in_assemble = 0.0        
    @property
    def time_step(self):
        return self._time_step(self.counter)

    def set_blk_mask(self):
        super(FirstOrderBackwardEuler, self).set_blk_mask()
        phys_target = self.get_target_phys()
        time_deriv_vars = []
        for phys in phys_target:
           dep_vars0 = phys.dep_vars0
           dep_vars = phys.dep_vars
           for x in dep_vars:
               if not x in dep_vars0: time_deriv_vars.append(x)
        self.time_deriv_vars = time_deriv_vars
        
        #self.fes_dt_mask = [False]*len(self.fes_mask)
        #for name in time_deriv_vars:
        #     offset = self.engine.dep_var_offset(name)
        #     self.fes_dt_mask[offset] = True

        dprint1("time deivatives to be computed", time_deriv_vars)
        
    def pre_assemble(self, update=False):
        engine = self.engine
        phys_target = self.get_phys()
        phys_range  = self.get_phys_range()
        
        if not update:
            engine.run_verify_setting(phys_target, self.gui)
            
        M_Updated = engine.run_assemble_mat(phys_target, phys_range,
                                             update=update)
        B_Updated = engine.run_assemble_b(phys_target, update=update)
        self.pre_assembled = True

        return M_Updated, B_Updated


    def compute_A(self, M, B, X, mask_M, mask_B):
        '''
        M/dt u_1 + K u_1 = M/dt u_0 + b
        '''
        one_dt = 1./float(self.time_step)
        A = M[0]+ M[1]*one_dt
        dprint1("A", A)
        return A, np.any(mask_M)

    def compute_rhs(self, M, B, X):
        one_dt = 1./float(self.time_step)
        MM = M[1]*one_dt
        RHS = MM.dot(self.engine.sol) + B
        dprint1("RHS", RHS)
        return RHS

    def assemble(self, update=False):
        blocks, M_changed = self.engine.run_assemble_blocks(self.compute_A,
                                        self.compute_rhs,
                                        inplace=False,
                                        update=update)

        #A, X, RHS, Ae, B, M, depvars = blocks
        self.assembled = True
        self._dt_used_in_assemble = self.time_step
        return M_changed
        
    def step(self, is_first):
        engine = self.engine
        mask = self.blk_mask
        engine.copy_block_mask(mask)
        
        #if not self.pre_assembled:
        #    assert False, "pre_assmeble must have been called"
            
        if (self.counter == 0 and is_first):
            M_changed = True
        else:
            if self._dt_used_in_assemble != self.time_step:
                engine.set_update_flag('UpdateAll')
            else:
                engine.set_update_flag('TimeDependent')    
            engine.run_apply_essential(self.get_phys(), self.get_phys_range(),
                                       update=True)
            engine.run_fill_X_block(update=True)        
            M_updated, B_updated = self.pre_assemble(update=True)
            M_changed = self.assemble(update=True)

        A, X, RHS, Ae, B, M, depvars = self.blocks
        if M_changed or self.counter == 0:
            AA = engine.finalize_matrix(A, mask,
                                        not self.phys_real, format = self.ls_type,
                                        verbose=False)
        BB = engine.finalize_rhs([RHS], A, X[-1], mask,
                                     not self.phys_real, format = self.ls_type,
                                     verbose=False)
        if self.counter == 0:
            self.sol = engine.sol
            self.write_checkpoint_solution()
            self.icheckpoint += 1

        depvars = [x for i, x in enumerate(depvars) if mask[0][i]]
        if self.linearsolver is None:
            is_complex = self.gui.is_complex()
            self.linearsolver  = self.linearsolver_model.allocate_solver(is_complex, engine)
            M_changed = True
            
        if M_changed:
            self.linearsolver.SetOperator(AA, dist = engine.is_matrix_distributed,
                                          name = depvars)
            
        if self.linearsolver.is_iterative:
            XX = engine.finalize_x(X[-1], RHS, mask, not self.phys_real,
                                   format = self.ls_type)
        else:
            XX = None
        solall = self.linearsolver.Mult(BB, x=XX, case_base=engine.case_base)
        engine.case_base += len(BB)
            
        if not self.phys_real and self.assemble_real:
            assert False, "this has to be debugged (convertion from real to complex)"
            solall = self.linearsolver_model.real_to_complex(solell, A)

        A.reformat_central_mat(solall, 0, X[0], mask)

        # this apply interpolation operator 
        sol, sol_extra = engine.split_sol_array(X[0])

        for name in self.time_deriv_vars:
            offset1 = engine.dep_var_offset(name)       # vt
            offset2 = engine.dep_var_offset(name[:-1])  # v
            X[0][offset1, 0] = (X[0][offset2, 0]-X[-1][offset2, 0])*(1./self.time_step)

        for child in self.child_instance:
            # for now update_operator is True only for the first run.
            child.solve(update_operator = (self.counter == 0))

        self.time = self.time + self.time_step
        
        self.counter += 1
        for p in self.probe:
            p.append_sol(X[0], self.time)

        
        # swap X[0] and X[-1] for next computing
        tmp = X[0]; X[0] = X[-1]; X[-1]=tmp
        self.sol = X[-1]
        engine.sol = self.sol

        engine.recover_sol(sol, access_idx=-1)
        ## ToDo. Provide a way to use Lagrange multipler in model
        extra_data = engine.process_extra(sol_extra)

        checkpoint_written = False
        if self.checkpoint[self.icheckpoint] < self.time:
            self.write_checkpoint_solution()
            self.icheckpoint += 1
            checkpoint_written = True
            
        dprint1("TimeStep ("+str(self.counter)+ "), t="+str(self.time)+"...done.")
        dprint1(debug.format_memory_usage())        
        return self.time >= self.et, checkpoint_written

    def write_checkpoint_solution(self):
        dprint1("writing checkpoint t=" + str(self.time) +
                "("+str(self.icheckpoint)+")")        
        od = os.getcwd()
        path = os.path.join(od, 'checkpoint_' + self.gui.parent.name()+'_'+
                            self.gui.name()+'_'+str(self.icheckpoint))
        self.engine.mkdir(path) 
        os.chdir(path)
        self.engine.cleancwd() 
        self.save_solution()
        self.engine.symlink('../model.pmfm', 'model.pmfm')        
        os.chdir(od)
        
class CrankNicolson(FirstOrderBackwardEuler):
    def compute_A(self, M, B, X, mask_M, mask_B):
        '''
        M/dt u_1 + K/2 u_1 = M/dt u_0 - K/2 u_0 + b
        '''
        one_dt = 1./float(self.time_step)
        #MM = M[1]*one_dt
        A = M[0]*0.5+ M[1]*one_dt
        dprint1("A", A)
        return A, np.any(mask_M)

    def compute_rhs(self, M, B, X):
        one_dt = 1./float(self.time_step)
        MM = (-M[0]*0.5 + M[1]*one_dt)
        RHS = MM.dot(self.engine.sol) + B
        dprint1("RHS", RHS)
        return RHS

class FirstOrderForwardEuler(FirstOrderBackwardEuler):
    def compute_A(self, M, B, X, mask_M, mask_B):
        '''
        M/dt u_1  = M/dt u_0 - K u_0 + b
        '''
        one_dt = 1./float(self.time_step)
        #MM = M[1]*one_dt
        A = M[1]*one_dt
        dprint1("A", A)
        return A, np.any(mask_M)

    def compute_rhs(self, M, B, X):
        one_dt = 1./float(self.time_step)
        MM = (-M[0] + M[1]*one_dt)
        RHS = MM.dot(self.engine.sol) + B
        dprint1("RHS", RHS)
        return RHS
    
    
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
                self.linearsolver[idt]  = self.linearsolver_model.allocate_solver(datatype,
                                                                                  engine)
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
        print("check sample1 (0)", [p.current_value(sol1) for p in self.probe])
        
        A, BB = get_A_BB(1, self.sol2)
        solall2 = self.linearsolver[self._time_step2].Mult(BB)
        sol2 = A.reformat_central_mat(solall2, 0)
        print("check sample2 (1)", [p.current_value(sol2) for p in self.probe])
        
        A, BB = get_A_BB(1, sol2, recompute_rhs=True)
        solall2 = self.linearsolver[self._time_step2].Mult(BB)
        sol2 = A.reformat_central_mat(solall2, 0)
        print("check sample2 (1)", [p.current_value(sol2) for p in self.probe])

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
                      



