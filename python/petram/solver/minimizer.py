#
#  Minimizer
#
#     minimizer implement a logic to find model parameters which minimize a
#     cost function. minimizer is used from optimizer
#

import petram.helper.pickle_wrapper as pickle

from itertools import product
import os
import numpy as np
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('Minimizer')
dprint0 = debug.regular_print('Minimizer', True)
format_memory_usage = debug.format_memory_usage

from petram.mfem_config import use_parallel
if use_parallel:
    from petram.helper.mpi_recipes import *
    from mpi4py import MPI
    myid = MPI.COMM_WORLD.rank
else:
    myid = 0

from mfem.common.mpi_debug import nicePrint

class CostFunction():
    '''
    CostFunction : __call__ runs simulation and evaluate cost
    '''
    def __init__(self, runner, fcost, params, model, engine):
        self.kcase = 0

        self.model = model
        self.runner = runner
        self.fcost = fcost
        self.engine = engine

        self.xvalues = dict()
        self.params = params
        for param in self.params:
            self.xvalues[param] = []
        self.costs = []

    def apply_param(self, args):
        general = self.model["General"]

        for name, value in zip(self.params, args):
            dprint1("CostFunction: Setting " + name + ':' + str(value))
            general.dataset[name] = value
            self.xvalues[name].append(np.atleast_1d(value))

    def call_cost(self, x, prbs):
        # create keyword arguments to call cost function
        probes = {}
        probes["prbs"] = prbs
        probes.update(prbs.__dict__)

        cost = self.fcost(x, **probes)

        return cost.item()

    def __call__(self, x):
        dprint1("!!!! Costfunction is called ("+str(self.kcase)+ ")")
        x = np.atleast_1d(x)

        self.apply_param(x)

        prbs = self.runner(self.kcase, self.engine)

        cost = self.call_cost(x, prbs)

        self.costs.append(np.atleast_1d(cost))

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

    def __init__(self, fcost, *args, **kwargs):
        self.fcost = fcost

        if len(args) % 2 == 1:
            self.x0 = np.atleast_1d(args[0])
            xx = args[1:].__iter__()
        else:
            self.x0 = None
            xx = args.__iter__()

        self.params = []
        self.ranges = []


        while True:
            try:
                self.params.append(xx.__next__())
                self.ranges.append(xx.__next__())
            except StopIteration:
                break

        assert len(self.params) == len(
            self.ranges), "invalid input to minimizer"

        self._data_record = {}

        self.runnder = None
        self.costobj = None
        self.model = None


    def generate_cost_function(self, engine, runner):
        cost = CostFunction(runner, self.fcost, self.params,
                            self.model, engine)
        self.costobj = cost

    def get_params(self):
        return self.params

    def set_model(self, model):
        self.model = model

        if self.x0 is None:
            gs = model['General']._global_ns
            data = [(gs[n] if n in gs else 0.0) for n in self.params]
            self.x0 = np.asarray(data)

    def collect_probe_signals(self, engine, dirs):
        '''
        scanner can implement its own probe collections
        '''
        raise NotImplementedError(
            "subclass needs to be provide this method")

    @property
    def names(self):
        '''
        suposed to return parameternames
        '''
        raise NotImplementedError(
            "set model for parametric scanner needs to be given in subclass")


class SimpleMinimizer(ParametricMinimizer):
    def __init__(self, fcost, *args, **kwargs):
        #self.method = kwargs.pop("method", "Nelder-Mead")
        self.method = kwargs.pop("method", "Powell")
        self.tol = kwargs.pop('tol', 1e-2)
        self.maxiter = kwargs.pop('maxiter', 100)
        self.maxfev = kwargs.pop('maxfev', 100)
        self.verbose = kwargs.pop('verbose', False)

        ParametricMinimizer.__init__(self, fcost, *args, **kwargs)

    def run(self):
        from scipy.optimize import minimize, minimize_scalar

        verbose = self.verbose if myid == 0 else False

        print("!!!!!", self.x0)
        if len(self.x0) > 1:
            res = minimize(self.costobj,
                           self.x0,
                           bounds=self.ranges,
                           method=self.method,
                           options={"maxfev": self.maxiter,
                                    "maxiter": self.maxiter,
                                    'xtol': self.tol,
                                    'ftol':self.tol,
                                    "disp": verbose,})

        else:
            res = minimize_scalar(self.costobj,
                                  self.x0,
                                  bounds=self.ranges[0],
                                  method="Bounded",
                                  tol = self.tol,
                                  options={"maxiter": self.maxiter,
                                           "disp": verbose,})

        if myid == 0:
            print(res)


Minimizer = SimpleMinimizer

def default_cost(*x, **kwargs):
    '''
    sample cost function, which causes NotImplemented error.

    cost function should accept
       *x : control parameter.
       **kwargs : all probe signals from previous simulaiton.

       return value should be scalar value.

    '''
    assert False, "Cost function is not implemented. Provide your own in global_ns"
