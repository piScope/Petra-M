'''
   BdrNodalEvaluator:
      a thing to evaluate solution on a boundary
'''
import numpy as np
import parser
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

    def eval_probe(self, expr, probe_files, phys):
        from petram.helper.variables import Variable, var_g
        from petram.sol.probe import load_probes
        
        path = probe_files[0]
        path = os.path.expanduser(path)        
        probes = probe_files[1]

        st = parser.expr(expr)
        code= st.compile('<string>')
        names = code.co_names

        g = phys._global_ns.copy()
        for key in var_g.keys():
            g[key] = var_g[key]

        for n in names:
            if n in probes:
                xdata, ydata = load_probes(path, probes[n])
                g[n] = ydata

        val = np.array(eval(code, g, {}), copy=False)
        return xdata, val

