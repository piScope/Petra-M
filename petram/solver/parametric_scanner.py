from __future__ import print_function

from itertools import product
import numpy as np
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('ParametricScanner')
dprint0 = debug.regular_print('ParametricScanner', True)
format_memory_usage = debug.format_memory_usage

class DefaultParametricScanner(object):
    def __init__(self, data = None):
        if data is None: data = []
        self._data = data
        self.idx = 0
        self.max = len(data)
        self.target = None

    def __iter__(self): return self

    def __next__(self):

        if self.idx == self.max:
            raise StopIteration

        data = self._data[self.idx]
        dprint0("Entering next parameter:", data, "(" +
                str(self.idx+1)+ "/" + str(self.max) + ")")
        dprint1(format_memory_usage())

        self.apply_param(data)
        self.idx = self.idx +1
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
    Scan("freq", [3e9, 4e9, 5e9], "phase", [0, 90])      # parameters are expanded to run all combination
    Scan("freq", [3e9, 4e9, 5e9], "phase", [0, 90, 180], product = False)
    Scan("freq", "phase", start = (3e9, 0), stop = (5e9, 180), nstep = 3)  # 1D scan 
    Scan("freq", "phase", start = (3e9, 0), stop = (5e9, 180), nstep = (3,4)) # 2D scan 

    '''
    def __init__(self, *args, **kwargs):
        use_product = kwargs.pop('product', True)
        
        if len(kwargs) != 0:
            self._names = args
            starts = np.atleast_1d(kwargs['start'])
            stops = np.atleast_1d(kwargs['stop'])
            steps = np.atleast_1d(kwargs['nstep'])
            if len(steps) == 1: use_product = False
            
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
            
        DefaultParametricScanner.__init__(self, data = list(data))

    def apply_param(self, data):
        names = self._names
                            
        dprint1("Simple Scanner: Target " + str(self.target_phys))
        for k, name in enumerate(names):
            for phys in self.target_phys:
                 dprint1("Simple Scanner: Setting " + name + ':' + str(data[k]))
                 phys._global_ns[name] = data[k]

    @property
    def names(self):
        '''
        suposed to return parameternames
        '''
        return self._names

                 
Scan = SimpleScanner
