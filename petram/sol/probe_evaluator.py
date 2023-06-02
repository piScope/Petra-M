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

    def eval_probe(self, expr, xexpr, probe_files, phys):
        from petram.helper.variables import Variable, var_g
        from petram.sol.probe import load_probes

        #print("probe_files", probe_files)
        path = probe_files[0]
        path = os.path.expanduser(path)        
        probes = probe_files[1]

        code= compile(expr, '<string>', 'eval')
        names = list(code.co_names)

        if len(xexpr.strip()) != 0:
            xcode = compile(xexpr, '<string>', 'eval')
            names.extend(xcode.co_names)
        else:
            xcode = None

        g = phys._global_ns.copy()
        for key in var_g.keys():
            g[key] = var_g[key]

        default_xname = ''
        for n in names:
            if n in probes:
                xdata, ydata = load_probes(path, probes[n])
                g[n] = ydata
                for nn in xdata:
                    g[nn] = xdata[nn]
                    default_xname = nn
                    

        val = np.array(eval(code, g, {}), copy=False)
        if xcode is None:
            xval = g[default_xname]
        else:
            xval = np.array(eval(xcode, g, {}), copy=False)

        return xval, val

