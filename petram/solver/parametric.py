import os
import traceback
import gc

from petram.model import Model
from petram.namespace_mixin import NS_mixin
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('Parametric')
format_memory_usage = debug.format_memory_usage

assembly_methods = {'Full assemble': 0,
                    'Reuse matrix' : 1}

class Parametric(Model, NS_mixin):
    '''
    parametric sweep of some model paramter
    and run solver
    '''
    can_delete = True
    has_2nd_panel = False
    
    def init_solver(self):
        pass

    def panel1_param(self):
        v = self.get_panel1_value()
        return [
                ["assembly method",  v[0],  4, {"readonly": True,
                      "choices": assembly_methods.keys()}],
                self.make_param_panel('scanner',  v[1]),
                [ "save separate mesh",  v[2],  3, {"text":""}],
                ["inner solver", v[3]  ,2, None],                
                ]
    
    def get_panel1_value(self):
        txt = assembly_methods.keys()[0]
        for k, n in assembly_methods.items():
            if n == self.assembly_method: txt = k

        return (
                str(txt),      
                str(self.scanner),    
                self.save_separate_mesh,
                self.get_inner_solver_names())

    def import_panel1_value(self, v):
        self.assembly_method = assembly_methods[v[0]]
        self.scanner = v[1]
        self.save_separate_mesh = v[2]

    def get_inner_solver_names(self):
        names = [s.name() for s in self.get_inner_solvers()]
        return ', '.join(names)
    
    def get_inner_solvers(self):
        return [self[k] for k in self if self[k].enabled]
        
    def attribute_set(self, v):
        v = super(Parametric, self).attribute_set(v)
        v['assembly_method'] = 0
        v['scanner'] = 'Scan("a", [1,2,3])'
        v['save_separate_mesh'] = True
        return v
    
    def get_possible_child(self):
        from petram.solver.std_solver_model import StdSolver        
        from petram.solver.mumps_model import MUMPS
        from petram.solver.gmres_model import GMRES
        return [StdSolver,]
    
    def get_scanner(self):
        try:
            scanner = self.eval_param_expr(str(self.scanner), 
                                           'scanner')[0]
        except:
            traceback.print_exc()
            return
        return scanner

    def get_default_ns(self):
        from petram.solver.parametric_scanner import Scan
        return {'Scan': Scan} 
    
    def run_method_0(self, solver, engine, kcase, ksolver,
                     isFirst):
        
        solver.init_sol(engine)
        matvecs, matvecs_c = solver.assemble(engine)

        
        solver.generate_linear_system(engine, matvecs, 
                                      matvecs_c)

        solall, PT = solver.call_solver(engine)

        extra_data = solver.store_sol(engine, matvecs, solall, 
                                      PT, 0)
        dprint1("Extra Data (case" + str(kcase)+")", 
                extra_data)
        
        od = os.getcwd()
        path = os.path.join(od, 'case' + str(kcase))

        if isFirst:
            solver.save_solution(engine, extra_data, 
                                 mesh_only = True)
            engine.mkdir(path)
            os.chdir(path)
            engine.cleancwd()
        else:
            os.chdir(path)
        solver.save_solution(engine, extra_data, 
                           skip_mesh = not self.save_separate_mesh)

        os.chdir(od)

    def run_method_1(self, solver, engine, kcase, ksolver,
                     isFirst):

        if isFirst:
            solver.init_sol(engine)            
            matvecs, matvecs_c = solver.assemble(engine)
            solver.generate_linear_system(engine, matvecs, 
                                          matvecs_c)
            self.matvecs[ksolver] = (matvecs, matvecs_c)
        else:
            solver.store_rhs(engine)

    def run(self, engine):
        scanner = self.get_scanner()
        if scanner is None: return
        
        solvers = self.get_inner_solvers()
        phys_models = []
        for s in solvers:
            for p in s.get_phys():
                if not p in phys_models: phys_models.append(p)
        scanner.set_phys_models(phys_models)

        sol_list = []
        self.matvecs = [None]*len(solvers)
        engine.case_base = 0
        for kcase, case in enumerate(scanner):
            isFirst = True
            for ksolver, s in enumerate(solvers):
                if self.assembly_method == 0:
                    self.run_method_0(s, engine, kcase, ksolver,
                                      isFirst)
                    dprint1(format_memory_usage())
                    
                elif self.assembly_method == 1:
                    #s.init_sol(engine)                    
                    self.run_method_1(s, engine, kcase, ksolver, 
                                      kcase == 0)
                else:
                    pass
                isFirst = False
        if self.assembly_method == 0: return        

        od = os.getcwd()
        for ksolver, s in enumerate(solvers):
            solall, PT = s.call_solver(engine)
            matvecs, matvecs_c = self.matvecs[0]
            for kcase in range(scanner.len()):
                extra_data = s.store_sol(engine, matvecs, 
                                         solall, PT, kcase)
                dprint1("Extra Data (case" + str(kcase)+")", 
                         extra_data)
                if kcase == 0:
                    s.save_solution(engine, extra_data, 
                                         mesh_only = True)
                path = os.path.join(od, 'case' + str(kcase))
                if ksolver == 0:
                    engine.mkdir(path) 
                    os.chdir(path)
                    engine.cleancwd() 
                else:
                    os.chdir(path)
                s.save_solution(engine, extra_data, 
                         skip_mesh = not self.save_separate_mesh) 

                dprint1(format_memory_usage())
                #gc.set_debug(gc.DEBUG_LEAK)
                #dprint1(gc.garbage)

        os.chdir(od)

