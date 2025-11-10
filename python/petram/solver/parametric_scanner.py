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
        self._current_data = None
        self._stored_data = []

    def get_probes(self):
        return []

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

    @property
    def current_data(self):
        return self._current_data

    def store_current_data(self):
        self._stored_data.append(self.current_data)

    def __next__(self):

        if self.idx == self.max:
            raise StopIteration

        data = self._data[self.idx]
        self._current_data = data
        self.store_current_data()

        dprint0("")
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

    def format_data(self, data):
        #
        # default format data
        #   controls what appears in subdir menu.
        #
        return str(data)

    def write_scanner_data(self, solver):
        #
        #  called at the end of parametric scan to save parameters
        #  to "parametric_data_(solvername).txt". This is used in subdir
        #  menus in dialog
        #
        solver_name = solver.name()
        dprint1("saving parameter", os.getcwd(), notrim=True)
        from petram.mfem_config import use_parallel

        if use_parallel:
            from mpi4py import MPI
        else:
            from petram.helper.dummy_mpi import MPI

        try:
            myid = MPI.COMM_WORLD.rank
            if myid == 0:
                fid = open("cases."+solver_name, "w")
                for k, x in enumerate(self._stored_data):
                    txt = str(k) + " : " + self.format_data(x)
                    fid.write(txt+"\n")
                fid.close()
            MPI.COMM_WORLD.Barrier()
        except:
            if use_parallel:
                MPI.COMM_WORLD.Abort()
            else:
                raise

    def save_namedparam_probes(self):
        #
        #  if scanner is using named parameter, overwrite this to
        #  save the parameter as probe signal (see SimpleScanner below)
        #
        pass

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

        self.probes = {}
        DefaultParametricScanner.__init__(self, data=list(data))

    def format_data(self, data):
        txts = []
        for k, name in enumerate(self.names):
            txts.append(name + '=' + str(data[k]))
        return ", ".join(txts)

    def apply_param(self, data):
        names = self._names
        general = self.target_phys[0].root()["General"]

        for k, name in enumerate(names):
            dprint1("Simple Scanner: Setting " + name + ':' + str(data[k]))
            general.dataset[name] = data[k]

    def save_namedparam_probes(self):
        try:
            from mpi4py import MPI
        except ImportError:
            from petram.helper.dummy_mpi import MPI
        myid = MPI.COMM_WORLD.rank

        if myid != 0:
            return

        from petram.sol.probe import Probe

        data = self.current_data
        probes = {n: Probe(n, ) for n in self._names}
        for k, name in enumerate(self._names):
            probes[name].append_value(data[k])
            probes[name].write_file(nosmyid=True)

    @property
    def names(self):
        '''
        suposed to return parameternames
        '''
        return self._names

    def get_probes(self):
        return self._names

    def collect_probe_data(self, engine, dirs):
        pass

    def collect_probe_signals(self, engine, dirs):
        pass
        # import warnings
        # warnings.warn(
        #    "collect_probe_signals is not used for SimpleScanner", UserWarning)


Scan = SimpleScanner
