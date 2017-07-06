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

        dprint0("Entering next parameter:", self._data[self.idx], "(" +
                str(self.idx+1)+ "/" + str(self.max) + ")")
        dprint1(format_memory_usage())

        self.apply_param(self._data[self.idx])        
        self.idx = self.idx +1
        return self.idx

    def set_phys_models(self, targets):
        '''
        set target physics model
        '''
        print targets
        if (not isinstance(targets, tuple) and
            not isinstance(targets, list)):
            self.target_phys = [targets]
        else:
            self.target_phys = targets
    def next(self):
        return self.__next__()

    def len(self):
        return self.max

    def set_model(self, data):     
        raise NotImplementedError(
             "set model for parametric scanner needs to be given in subclass")


class SimpleScanner(DefaultParametricScanner):
    def __init__(self, name = '', data = None, start = None,
                 stop = None, num = None):
        self.name = name
        if (start is not None and stop is not None and num is not None):
            data = np.linspace(start, stop, num)
        dprint1(data)
        DefaultParametricScanner.__init__(self, data = data)


    def apply_param(self, data):
        if (not isinstance(self.name, tuple) and 
            not isinstance(self.name, list)):
            names = [self.name]
            data = [data]
        else:
            names = self.name
            #data = self.data
        
        dprint1("Simple Scanner: Target " + str(self.target_phys))
        for k, name in enumerate(names):
            for phys in self.target_phys:
                 dprint1("Simple Scanner: Setting " + name + ':' + str(data[k]))
                 phys._global_ns[name] = data[k]

Scan = SimpleScanner
