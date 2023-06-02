from __future__ import print_function
import petram.helper.pickle_wrapper as pickle

from itertools import product
import os
import numpy as np
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('ParametricScanner')
dprint0 = debug.regular_print('ParametricScanner', True)
format_memory_usage = debug.format_memory_usage


class DefaultParametricScanner(object):
    def __init__(self, data=None):
        if data is None:
            data = []
        self.idx = 0
        self.target = None
        self.set_data(data)

    def set_data_from_model(self, model):
        '''
        this is called after __init__.
        model is passed. so that it can be set using   
        model tree
        '''
        pass

    def set_data(self, data):
        self._data = data
        self.max = len(data)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):

        if self.idx == self.max:
            raise StopIteration

        data = self._data[self.idx]
        dprint0("==== Entering next parameter:", data, "(" +
                str(self.idx+1) + "/" + str(self.max) + ")")
        dprint1(format_memory_usage())

        self.apply_param(data)

        self.idx = self.idx + 1
        return self.idx

    def list_data(self):
        return list(self._data)

    def set_phys_models(self, targets):
        '''
        set target physics model
        '''
        if (not isinstance(targets, tuple) and
                not isinstance(targets, list)):
            self.target_phys = [targets]
        else:
            self.target_phys = targets

    def next(self):
        return self.__next__()

    def __len__(self):
        return self.max

    def len(self):
        return self.max

    def save_scanner_data(self, solver):
        solver_name = solver.fullpath()
        data = self.list_data()
        dprint1("saving parameter", os.getcwd())
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


class SimpleScanner(DefaultParametricScanner):
    '''
    Scan("freq", [3e9, 4e9, 5e9])
    # parameters are expanded to run all combination
    Scan("freq", [3e9, 4e9, 5e9], "phase", [0, 90])
    Scan("freq", [3e9, 4e9, 5e9], "phase", [0, 90, 180], product = False)
    # 1D scan
    Scan("freq", "phase", start = (3e9, 0), stop = (5e9, 180), nstep = 3)
    # 2D scan
    Scan("freq", "phase", start = (3e9, 0), stop = (5e9, 180), nstep = (3,4)) 
    '''

    def __init__(self, *args, **kwargs):
        use_product = kwargs.pop('product', True)

        if len(kwargs) != 0:
            self._names = args
            starts = np.atleast_1d(kwargs['start'])
            stops = np.atleast_1d(kwargs['stop'])
            steps = np.atleast_1d(kwargs['nstep'])
            if len(steps) == 1:
                use_product = False

            data = []
            for k in range(len(self._names)):
                s = starts[k]
                e = stops[k]
                n = steps[0] if len(steps) == 1 else steps[k]
                data.append(np.linspace(s, e, n))

        else:
            o = iter(args)
            names = []
            data = []
            while True:
                try:
                    names.append(o.__next__())
                    data.append(o.__next__())
                except StopIteration:
                    break
            self._names = names

        if use_product:
            data = product(*data)
        else:
            data = zip(*data)

        DefaultParametricScanner.__init__(self, data=list(data))

    def apply_param(self, data):
        names = self._names

        dprint1("Simple Scanner: Target " + str(self.target_phys))
        for k, name in enumerate(names):
            id_list = []
            for phys in self.target_phys:
                if id(phys._global_ns) in id_list:
                    continue

                dprint1("Simple Scanner: Setting " + name + ':' + str(data[k]) +
                        " setting in " + str(phys))
                phys._global_ns[name] = data[k]
                id_list.append(id(phys._global_ns))

    @property
    def names(self):
        '''
        suposed to return parameternames
        '''
        return self._names


Scan = SimpleScanner
