from __future__ import print_function
import petram.helper.pickle_wrapper as pickle

from itertools import product
import os
import numpy as np
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('Optimizer')
dprint0 = debug.regular_print('ParametricScanner', True)
format_memory_usage = debug.format_memory_usage


class CostFunction():
    def __init__(self, runner, fcost, params):
        self.kcase = 0

        self.runner = runner
        self.fcost = fcost

        self.xvalues = []
        self.params = params
        for param in self.params:
            self.xvalues[param] = []

    def apply_param(self, *args):
        dprint1("Simple Scanner: Target " + str(self.target_phys))

        general = self.target_phys[0].root()["General"]

        for name, value in zip(params, args):
            dprint1("Simple Scanner: Setting " + name + ':' + str(value))
            general.dataset[name] = value

    def __call__(self, *args):
        for x, name in zip(args, self.params):
            self.xvalues[name].append(x)

        self.apply_param(*args)
        self.runner(self.kcase)
        cost = self.fcost()
        self.kcase = self.kcase + 1

        return cost


class ParametricMinimizer():
    '''
    Minimize(cost, param1, range1, param2, range2, **kwargs)

    cost: callable, function to be minimized
    param1, range1: name of parameter (variable in global_ns), and it's range to search

    kwargs : (TBD)
      reltol = 1e-2
      abstol = 1e-2
    '''

    def __init__(self, *args, **kwargs):
        self.fcost = args[0]

        self.params = []
        self.ranges = []

        xx = args[1:].__iter__()
        while True:
            try:
                self.params.append(xx.__next__())
                self.ranges.append(xx.__next__())
            except StopIteration:
                break

        assert len(self.params) != len(
            self.ranges), "invalid input to minimizer"

        self._data_record = {}

        self.runnder = None
        self.costobj = None

    def generate_cost_function(self, runner):
        cost = CostFunction(runner, self.fcost, self.params)
        self.costobj = cost

    def set_data_from_model(self, model):
        '''
        this is called after __init__.
        model is passed. so that it can be set using
        model tree
        '''
        pass

    def set_phys_models(self, targets):
        '''
        set target physics model
        '''
        if (not isinstance(targets, tuple) and
                not isinstance(targets, list)):
            self.target_phys = [targets]
        else:
            self.target_phys = targets

    def save_optimizer_data(self, solver):
        solver_name = solver.fullpath()
        data = self.list_data()
        dprint1("saving parameter", os.getcwd(), notrim=True)
        try:
            from mpi4py import MPI
        except ImportError:
            from petram.helper.dummy_mpi import MPI
        myid = MPI.COMM_WORLD.rank

        if myid == 0:
            fid = open("parametric_data_"+solver_name, "wb")
            dd = {"name": solver_name, "data": data}
            pickle.dump(dd, fid)
            fid.close()

        MPI.COMM_WORLD.Barrier()

    def collect_probe_signals(self, engine, dirs):
        '''
        scanner can implement its own probe collections
        '''
        raise NotImplementedError(
            "subclass needs to be provide this method")

    def set_model(self, data):
        raise NotImplementedError(
            "set model for parametric scanner needs to be given in subclass")

    @property
    def names(self):
        '''
        suposed to return parameternames
        '''
        raise NotImplementedError(
            "set model for parametric scanner needs to be given in subclass")


class SimpleMinimizer(ParametricMinimizer):
    def __init__(self, fcost, x0, *args, **kwargs):
        self.x0 = x0
        self.bounds = kwargs.pop("bounds", None)
        self.method = kwargs.pop("method", "Nelder-Mead")
        self.tol = kwargs.pop('tol', 1e-2)
        self.maxiter = kwargs.pop('maxiter', 100)
        self.verbose = kwargs.pop('verbose', False)

        ParametricMinimizer.__init__(self, fcost, *args, **kwargs)

    def run(self):
        from scipy.optimize import minimize

        res = minimize(self.costobj,
                       self.x0,
                       bounds=self.bounds,
                       method=self.method,
                       tol=self.tol,
                       options={"maxiter": self.maxiter, "disp": self.verobse},)

    def collect_probe_data(self, engine, dirs):
        pass

    def collect_probe_signals(self, engine, dirs):
        pass


Minimizer = SimpleMinimizer
