import time
import numpy as np
import parser
import weakref
import traceback
from weakref import WeakKeyDictionary as WKD
from weakref import WeakValueDictionary as WVD


from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

import multiprocessing as mp
from petram.sol.evaluators import Evaluator, EvaluatorCommon
from petram.sol.evaluator_mp import EvaluatorMPChild, EvaluatorMP

class EvaluatorServer(object):
    '''
    this is an hold of EvaluatorMP 
    '''
    def __init__(self, nproc = 2):
        print("new evaluator server", nproc)
        self.init_done = False        
        self.tasks = BroadCastQueue(nproc)
        self.results= mp.JoinableQueue() 
        self.workers = [None]*nproc
        self.solfiles = None
        
        for i in range(nproc):
            w = EvaluatorMPChild(self.tasks[i], self.results, i, nproc)
            self.workers[i] = w
            time.sleep(0.1)
        for w in self.workers: w.start()
        
    def __del__(self):
        self.terminate_all()

    def set_model(self, model):
        import tempfile, shutil
        tmpdir = tempfile.mkdtemp()
        model_path = os.path.join(tmpdir, 'model.pmfm')
        model.save_to_file(model_path,
                           meshfile_relativepath = False)
        self.tasks.put((3, model_path), join = True)
        shutil.rmtree(tmpdir)
        
    def set_solfiles(self, solfiles):
        self.solfiles = weakref.ref(solfiles)        
        self.tasks.put((2, solfiles))

    def make_agents(self, name, params, **kwargs):
        super(EvaluatorMP, self).make_agents(name, params)
        self.tasks.put((1, name, params, kwargs))
        
    def load_solfiles(self, mfem_mode = None):
        self.tasks.put((4, ), join = True)
        
    def set_phys_path(self, phys_path):        
        self.tasks.put((5, phys_path))
        
    def validate_evaluator(self, name, attr, solfiles, **kwargs):
        redo_geom = False
        if (self.solfiles is None or
            self.solfiles() is not solfiles):
            redo_geom = True
        if not super(EvaluatorMP, self).validate_evaluator(name, attr, **kwargs):
            redo_geom = True
        if not self.init_done: redo_geom = True
        if not redo_geom: return

        self.make_agents(self._agent_params[0],
                         attr, **kwargs)
        self.tasks.put((6, attr, kwargs))        
        self.init_done = True
        
    def eval(self, expr, merge_flag1, merge_flag2, **kwargs):
        self.tasks.put((7, expr, kwargs), join = True)
        print("waiting for answer'")
        res = [self.results.get() for x in range(len(self.workers))]
        results = [x[0] for x in res if x[0] is not None]
        attrs = [x[1] for x in res if x[0] is not None]
        attrs = attrs[0]
        
        data = [None]*len(attrs)

        for kk, x in enumerate(results):
            for k, y in enumerate(x):
                if len(y) == 0: continue
                if data[k] is None: data[k] = y
                else: data[k].extend(y)

        if merge_flag1:
            data0 = [None]*len(attrs)
            for k, x in enumerate(data):
                vdata, cdata = zip(*x)
                data0[k] = [(np.vstack(vdata), np.vstack(cdata))]
            data = data0
            
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
                

    def terminate_all(self):
        print('terminating all')      
        #num_alive = 0
        #for w in self.workers:
        #    if w.is_alive(): num_alive = num_alive + 1
        #for x in range(num_alive):
        self.tasks.put([-1])
        self.tasks.join()
        print('joined')


class EvaluatorClient(Evaluator):
    def __init__(self, nproc = 2, host = 'localhost',
                       soldir = ''):
        self.init_done = False        
        self.soldir = soldir
        self.solfiles = None
    

