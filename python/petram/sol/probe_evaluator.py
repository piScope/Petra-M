'''
   BdrNodalEvaluator:
      a thing to evaluate solution on a boundary
'''
import numpy as np
import weakref
import six
import os

from weakref import WeakKeyDictionary as WKD
from weakref import WeakValueDictionary as WVD

from petram.sol.evaluator_agent import EvaluatorAgent


class ProbeEvaluator(EvaluatorAgent):
    def __init__(self, battrs):
        super(ProbeEvaluator, self).__init__()
        self.battrs = battrs

    def preprocess_geometry(self,  *args, **kargs):
        pass

    def eval_probe(self, expr, xexpr, probe_files, phys=None):
        from petram.helper.variables import Variable, var_g
        from petram.sol.probe import collect_probesignals

        path = probe_files[0]
        path = os.path.expanduser(path)
        path = os.path.join(path, *probe_files[1])

        prbs = collect_probesignals(path)

        code = compile(expr, '<string>', 'eval')
        names = list(code.co_names)

        if len(xexpr.strip()) != 0:
            xcode = compile(xexpr, '<string>', 'eval')
            names.extend(xcode.co_names)
        else:
            xcode = None

        if phys is not None:   # this option is not used anymore (?)
            g = phys._global_ns.copy()
        else:
            g = {}

        g.update(var_g)

        g["prbs"] = prbs
        g.update(prbs.__dict__)

        val = np.asarray(eval(code, g, {}))
        if xcode is None:
            xval = None
        else:
            xval = np.asarray(eval(xcode, g, {}))

        return xval, val
