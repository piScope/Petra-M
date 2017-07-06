import time
import numpy as np
import parser
import weakref
import traceback
import six
from weakref import WeakKeyDictionary as WKD
from weakref import WeakValueDictionary as WVD


from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

import multiprocessing as mp
from petram.sol.evaluators import Evaluator, EvaluatorCommon

class EvaluatorSingle(EvaluatorCommon):
    '''
    define a thing which takes expression involving Vriables
    and evaualte it
    '''
    def __init__(self):
        self.mfem_model = None
        self.solfiles = None
        self.solvars = WKD()
        self.agents = {}
        self.physpath = ''
        self.init_done = False
        
    def set_solfiles(self, solfiles):
        self.solfiles = weakref.ref(solfiles)
        
    def set_phys_path(self, phys_path):
        self.phys_path = phys_path

    def validate_evaluator(self, name, attr, solfiles, **kwargs):
        #print("validate evaulator", self.solfiles(), solfiles)
        redo_geom = False
        if (self.solfiles is None or
            self.solfiles() is not solfiles):
            self.set_solfiles(solfiles)
            print("new_solfiles")
            redo_geom = True
        if not super(EvaluatorCommon, self).validate_evaluator(name, attr, **kwargs):
            redo_geom = True
        if not self.init_done: redo_geom = True
        if not redo_geom: return
 
        solvars = self.load_solfiles()
        self.make_agents(self._agent_params[0],
                         attr, **kwargs)
        for key in six.iterkeys(self.agents):
            evaluators = self.agents[key]
            for o in evaluators:
                o.preprocess_geometry([key], **kwargs)

        self.init_done = True
                
    def eval(self, expr, merge_flag1, merge_flag2, **kwargs):
        if self.phys_path == '': return None, None
        
        phys = self.mfem_model()[self.phys_path]
        solvars = self.load_solfiles()
        if solvars is None: return None, None
        
        data = []
        attrs = []
        for key in six.iterkeys(self.agents): # scan over battr
            vdata = []
            cdata = []
            data.append([])
            attrs.append(key)                                  
            evaluators = self.agents[key]
            for o, solvar in zip(evaluators, solvars): # scan over sol files
                v, c = o.eval(expr, solvar, phys, **kwargs)
                if v is None: continue
                vdata.append(v)
                cdata.append(c)
                data[-1].append((v, c))

            if merge_flag1:
                if len(vdata) != 0:
                    data[-1]  = [(np.vstack(vdata), np.vstack(cdata))]
                else:
                    return None, None # for now,, let's do this
                
        ## normally this is good option.
        if merge_flag1 and not merge_flag2:
            vdata = np.vstack([x[0][0] for x in data])
            cdata = np.vstack([x[0][1] for x in data])
            data = [(vdata, cdata)]
        elif merge_flag1:
            data0 = []
            for x in data: data0.extend(x)
            data = data0
        else:
            data0 = []
            for x in data: data0.extend(x)
            data = data0
        return data, attrs                                  
