import time
import numpy as np
import parser
import weakref
import traceback
import six
import os
import sys
import tempfile
from weakref import WeakKeyDictionary as WKD
from weakref import WeakValueDictionary as WVD


from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

import multiprocessing as mp
from petram.sol.evaluators import Evaluator, EvaluatorCommon

def data_partition(m, num_proc, myid):
    min_nrows  = m / num_proc
    extra_rows = m % num_proc
    start_row  = min_nrows * myid + (extra_rows if extra_rows < myid else myid)
    end_row    = start_row + min_nrows + (1 if extra_rows > myid else 0)
    nrows   = end_row - start_row
    return start_row, end_row

class BroadCastQueue(object):
   def __init__(self, num):
       self.queue = [None]*num
       self.num = num
       for i in range(num):
           self.queue[i] = mp.JoinableQueue()
           
   def put(self, value, join = False):
       for i in range(self.num):
           self.queue[i].put(value)
       if join:
           for i in range(self.num):
                self.queue[i].join()
   def join(self):         
       for i in range(self.num):
           self.queue[i].join()
           
   def close(self):         
       for i in range(self.num):
           self.queue[i].close()

   def __getitem__(self, idx):
       return self.queue[idx]

   
class EvaluatorMPChild(EvaluatorCommon, mp.Process):
    def __init__(self, task_queue, result_queue, myid, rank,
                 logfile = False):

        mp.Process.__init__(self)
        EvaluatorCommon.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.myid = myid
        self.rank = rank
        self.solvars = WKD()        
        self.agents = {}
        self.logfile = logfile

    def run(self, *args, **kargs):
        if self.logfile == 'suppress':
            sys.stdout = open(os.devnull, 'w')
        elif self.logfile == 'log':
            path = os.path.expanduser('~/MPChild.out')
            sys.stdout = open(path, "w", 0)
        else:
            pass
        while True:
            time.sleep(0.01)
            try:
               task = self.task_queue.get(True)
            except EOFError:
                self.result_queue.put((-1, None))                
                self.task_queue.task_done()
                continue
            if task[0] == -1:
                self.task_queue.task_done()
                break
            try:
                #print("got task", task[0], self.myid, self.rank)
                if task[0] == 1: # (1, cls, param) = make_agents
                    if self.solfiles is None: continue                    
                    cls = task[1]
                    params = task[2]
                    kwargs = task[3]
                    self.make_agents(cls, params, **kwargs)

                elif task[0] == 2: # (2, solfiles) = set_solfiles
                    
                    self.set_solfiles(task[1])
                    
                elif task[0] == 3: # (3, mfem_model) = set_model
                    self.set_model(task[1])
                    
                elif task[0] == 4: # (4,)  = load_solfiles
                    if self.solfiles is None: continue                    
                    self.load_solfiles()
                    
                elif task[0] == 5: # (5,)  = phys_path
                    self.phys_path = task[1]
                    
                elif task[0] == 6: # (6, attr)  = process_geom
                    if self.solfiles is None: continue
                    self.call_preprocesss_geometry(task[1], **task[2])
                    
                elif task[0] == 7: # (7, expr)  = eval
                    if self.solfiles is None:
                        value = (None, None)
                    else:
                        value =  self.eval(task[1], **task[2])
            except:
                traceback.print_exc()
                value = (None, None)
            finally:
                self.task_queue.task_done()
                if task[0] == 7:
                    self.result_queue.put(value)
                
        #end of while
        self.task_queue.close()
        self.result_queue.close()
        
    def set_solfiles(self, solfiles):
        st, et = data_partition(len(solfiles.set), self.rank, self.myid)
        s = solfiles[st:et]
        if len(s) > 0:
            self.solfiles_real = s
            self.solfiles = weakref.ref(s)
        else:
            self.solfiles = None
            
    def set_model(self, model_path):
        try:
            from petram.engine import SerialEngine
            s = SerialEngine(modelfile = model_path)
            s.run_config()
            s.run_mesh()
            s.assign_sel_index()        
            self.model_real = s.model
        except:
             print(traceback.format_exc())
        super(EvaluatorMPChild, self).set_model(s.model)

    def call_preprocesss_geometry(self, attr, **kwargs):
        solvars = self.load_solfiles()
        for key in six.iterkeys(self.agents):
            evaluators = self.agents[key]
            for o in evaluators:
                o.preprocess_geometry([key], **kwargs)                
        
    def eval(self, expr, **kwargs):
        phys_path = self.phys_path
        phys = self.mfem_model()[phys_path]
        solvars = self.load_solfiles()
        
        if solvars is None: return None, None

        data = []
        attrs = []
        for key in six.iterkeys(self.agents): # scan over battr
            data.append([])
            attrs.append(key)                                  
            evaluators = self.agents[key]
            for o, solvar in zip(evaluators, solvars): # scan over sol files
                v, c, a = o.eval(expr, solvar, phys, **kwargs)
                if v is None:
                    v = None; c = None; a = None
                data[-1].append((v, c, a))
        #print("eval result", data, attrs)
        return data, attrs
        

class EvaluatorMP(Evaluator):
    def __init__(self, nproc = 2, logfile = False):
        print("new evaluator MP", nproc)
        self.init_done = False        
        self.tasks = BroadCastQueue(nproc)
        self.results= mp.JoinableQueue() 
        self.workers = [None]*nproc
        self.solfiles = None
        self.failed = False        
        for i in range(nproc):
            w = EvaluatorMPChild(self.tasks[i], self.results, i, nproc,
                                 logfile = logfile)
            self.workers[i] = w
            time.sleep(0.1)
        for w in self.workers:
            w.daemon = True
            w.start()
        
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
            print("new solfiles")
            self.set_solfiles(solfiles)
            self.load_solfiles()
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
        for x in range(len(self.workers)):
            self.results.task_done()
            
        results = [x[0] for x in res if x[0] is not None]
        attrs = [x[1] for x in res if x[0] is not None]
        attrs = attrs[0]

        data = [None]*len(attrs)

        for kk, x in enumerate(results):
            for k, y in enumerate(x):
                if data[k] is None: data[k] = y
                else: data[k].extend(y)
        num_files = len(data[0])
        def omit_none(l):
            return [x for x in l if x is not None]

        if merge_flag1:
            #data0 = [None]*len(attrs)
            data0 = []
            offset = 0            
            for k, x in enumerate(data):
                if merge_flag2: offset = 0                
                vdata, cdata, adata = zip(*x)
                if len(omit_none(vdata)) == 0: continue
                for c, a in zip(cdata, adata):
                    if c is not None:                    
                        a += offset
                        offset = len(c) + offset
                data0.append([(np.vstack(omit_none(vdata)),
                              np.hstack(omit_none(cdata)),
                              np.vstack(omit_none(adata)))])
            data = data0

        # eliminate non-existent attribute
        data0 = []; attrs0 = []
        for x, a in zip(data, attrs):
            if x is not None:
                data0.append(x)
                attrs0.append(a)
        data = data0; attrs = attrs0                
        
        if merge_flag1 and not merge_flag2:
            x0 = [x[0][0] for x in data]
            if len(x0) == 0:
                assert False, "No slice data point"
            vdata = np.vstack(x0)
            cdata = np.hstack([x[0][1] for x in data])
            adata = np.vstack([x[0][2] for x in data])
            data = [(vdata, cdata, adata)]
        elif merge_flag1:
            data0 = []
            for x in data: data0.extend(x)
            data = data0
        elif not merge_flag2:
            keys = attrs
            data0 = []
            attr = []
            for idx in range(num_files): # for each file
                vdata = []
                cdata = []
                adata = []
                offset = 0
                for idx0, key in enumerate(keys):
                    d1 = data[idx0][idx]
                    if d1[0] is None: continue
                    vdata.append(d1[0])
                    cdata.append(d1[1])
                    adata.append(d1[2]+offset)
                    offset = offset + d1[1].shape[0]
                if offset == 0: continue
                dd  = (np.vstack(vdata), np.hstack(cdata), np.vstack(adata))
                data0.append(dd)
                attr.append(key)
            attrs = list(set(attr))
            data = data0
        else:
            data0 = []
            for x in data:
                data0.extend([xx for xx in x if xx[0] is not None])
            data = data0
        return data, attrs
    
    def terminate_all(self):
        #print('terminating all')      
        #num_alive = 0
        #for w in self.workers:
        #    if w.is_alive(): num_alive = num_alive + 1
        #for x in range(num_alive):
        self.tasks.put([-1])
        self.tasks.join()
        self.tasks.close()
        self.results.close()
        print('joined')

    

