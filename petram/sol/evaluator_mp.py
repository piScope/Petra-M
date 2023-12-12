import time
import numpy as np
import weakref
import traceback
import six
import os
import sys
import tempfile
from six import StringIO
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
    min_nrows  = m // num_proc
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
                
   def put_single(self, value, join=False):
       self.queue[0].put(value)
       if join:
          self.queue[0].join()
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
                 logfile = False, text_queue = None):

        mp.Process.__init__(self)
        EvaluatorCommon.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.text_queue = text_queue        
        self.myid = myid
        self.rank = rank
        self.agents = {}
        self.logfile = logfile
        print("log file")
        #self.logfile = 'log'
        self.use_stringio = False
        self.solfiles = None
        
        ## enable it when checking performance        
        self.use_profiler = False

    def run(self, *args, **kargs):
        # this can not be in init, sicne it is not pickable
        self.solvars = WKD()
        
        if self.logfile == 'suppress':
            sys.stdout = open(os.devnull, 'w')
        elif self.logfile == 'queue':
            self.use_stringio = True
        elif self.logfile == 'log':
            path = os.path.expanduser('~/MPChild'+ str(os.getpid()) + '.out')
            sys.stdout = open(path, "w")
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
                if self.use_stringio:
                    stringio = StringIO()
                    org_sys_stdout = sys.stdout
                    org_sys_stderr = sys.stderr
                    sys.stdout = stringio
                    sys.stderr = stringio
                    
                if self.use_profiler:
                    import cProfile
                    pr = cProfile.Profile()
                    pr.enable()

                #print("got task", task[0], self.myid, self.rank)
                if task[0] == 1: # (1, cls, param) = make_agents
                    cls = task[1]
                    params = task[2]
                    kwargs = task[3]

                    if self.solfiles is None and cls != 'Probe':
                        continue
                    self.make_agents(cls, params, **kwargs)

                elif task[0] == 2: # (2, solfiles) = set_solfiles
                    self.set_solfiles(task[1])
                    ll = len(self.solfiles) if self.solfiles is not None else 0
                    value = (self.myid, ll, None)

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
                        value = (self.myid, None, None)
                    else:
                        value =  self.eval(task[1], **task[2])

                elif task[0] == 8: # (8, expr)  = eval_probe
                    value = self.eval_probe(task[1], task[2], task[3])
                    value = (self.myid, value[0], value[1])

                elif task[0] == 9: # (8, expr)  = make_probe_agents
                    cls = task[1]
                    params = task[2]
                    kwargs = task[3]
                    self.make_probe_agents(cls, params, **kwargs)
                    
                elif task[0] == 10: # (10, expr)  = eval_pointcloud
                    value = self.eval_pointcloud(task[1], **task[2])
                    
                elif task[0] == 11: # (11, expr)  = eval_integral
                    value = self.eval_integral(task[1], **task[2])

            except:
                traceback.print_exc()
                err = traceback.format_exc()
                value = (self.myid, None, None)
            finally:
                self.task_queue.task_done()

                if self.use_profiler:
                    import  pstats                    
                    pr.disable()
                    ps = pstats.Stats(pr,
                                      stream=sys.stdout).sort_stats('cumulative')
                    ps.print_stats()


                if task[0] in [2, 7, 8, 10, 11]:
                    self.result_queue.put(value)
                    
                if self.use_stringio:
                    output = stringio.getvalue()
                    stringio.getvalue()
                    sys.stdout = org_sys_stdout
                    sys.stderr = org_sys_stderr 
                    stringio.close()
                    self.text_queue.put(output)
                    
        #end of while
        self.task_queue.close()
        self.result_queue.close()
        
    def set_solfiles(self, solfiles):
        st, et = data_partition(len(solfiles.set), self.rank, self.myid)
        s = solfiles[st:et]
        print(s)
        if len(s) > 0:
            self.solfiles_real = s
            self.solfiles = s
        else:
            self.solfiles = None

    def set_model(self, model_path):
        try:
            from petram.engine import SerialEngine
            s = SerialEngine(modelfile = model_path)

            if (os.path.basename(model_path) == 'model_proc.pmfm'):
                s.build_ns()
            else:
                s.run_config()            
                model_path = os.path.join(os.path.dirname(model_path), 'model_proc.pmfm')
                if self.myid == 0:
                    s.model.save_to_file(model_path,
                                         meshfile_relativepath=False)

            self.model_real = s.model
        except:
            print(traceback.format_exc())
            return self.myid, None, traceback.format_exc()
            
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
        
        if solvars is None: return self.myid, None, None

        data = []
        attrs = []
        for key in six.iterkeys(self.agents): # scan over battr
            data.append([])
            attrs.append(key)                                  
            evaluators = self.agents[key]
            for o, solvar in zip(evaluators, solvars): # scan over sol files
                try:
                     v, c, a = o.eval(expr, solvar, phys, **kwargs)
                except:
                     import traceback
                     return self.myid, None, traceback.format_exc()
                     
                if v is None:
                    v = None; c = None; a = None
                data[-1].append((v, c, a))

        return self.myid, data, attrs

    def eval_pointcloud(self, expr, **kwargs):
        if self.phys_path == '': return None, None, None

        #use_pr = True
        #import cProfile
        #if use_pr:
        #    pr = cProfile.Profile()
        #    pr.enable()
        
        phys = self.mfem_model()[self.phys_path]
        solvars = self.load_solfiles()

        if solvars is None: return self.myid, None, None        

        export_type = kwargs.get('export_type', 1)
        
        data = []
        attrs = []
        offset = 0

        key = list(self.agents)[0]
        
        vdata = [] # vertex
        cdata = [] # data
        adata = [] # array idx     
        attrs.append(key)                                  
        evaluators = self.agents[key]
        
        for o, solvar in zip(evaluators, solvars): # scan over sol files
           try:
               v, c, a = o.eval(expr, solvar, phys, **kwargs)
           except:
               import traceback
               return self.myid, None, traceback.format_exc()
               
           if v is None:
               v = None; c = None; a = None
           else:
               vdata.append(v)
               cdata.append(c)
               adata.append(a)

        if len(adata)==0: return None, None, None

        ptx  = vdata[0]
        ddim = cdata[0].shape[1:]  # data dim

        shape = adata[0].shape
        shape_d = sum((shape, ddim), ())
                      
        attrs = np.zeros(shape, dtype=int)-1

        data  = np.zeros(shape_d, dtype=cdata[0].dtype)

        #print("data shape", data.shape)
                
        for v, c, a in zip(vdata, cdata, adata):
            idx = (a != -1)
            if np.sum(idx) == 0: continue
            attrs[idx] = a[idx]
            data[idx] = c

        #if use_pr:
        #    path = os.path.expanduser('~/MPChild_profile'+ str(os.getpid()) + '.out')
        #    pr.dump_stats(path)
        #    pr.disable()
        return ptx, data, attrs

    def eval_integral(self, expr, **kwargs):
        phys_path = self.phys_path
        phys = self.mfem_model()[phys_path]
        solvars = self.load_solfiles()
        
        if solvars is None:
            return self.myid, None, None

        data = 0.0
        for key in six.iterkeys(self.agents): # scan over battr
            evaluators = self.agents[key]
            for o, solvar in zip(evaluators, solvars): # scan over sol files
                try:
                     v = o.eval_integral(expr, solvar, phys, **kwargs)
                except:
                     import traceback
                     return self.myid, None, traceback.format_exc()
                     
                if v is None:
                    v = None
                data = data + v

        return self.myid, data, 0
        
    def eval_probe(self, expr, xexpr, probes):
        if self.phys_path == '': return None, None
        
        phys = self.mfem_model()[self.phys_path]
        evaluator = self.agents[1][0]
        
        return evaluator.eval_probe(expr, xexpr, probes, phys)
        

class EvaluatorMP(Evaluator):
    def __init__(self, nproc = 2, logfile = False):
        super(EvaluatorMP, self).__init__()
        print("new evaluator MP", nproc)
        self.init_done = False        
        self.tasks = BroadCastQueue(nproc)
        self.results= mp.JoinableQueue() 
        self.workers = [None]*nproc
        self.solfiles = None
        self.failed = False
        self.closed = False
        
        if logfile == 'queue':
            self.text_queue = mp.Queue()
        else:
            self.text_queue = None

        for i in range(nproc):
            w = EvaluatorMPChild(self.tasks[i], self.results, i, nproc,
                                 logfile = logfile,
                                 text_queue = self.text_queue)
            self.workers[i] = w
            time.sleep(0.1)
        for w in self.workers:
            w.daemon = True
            w.start()
        
    def __del__(self):
        if not self.closed:
            self.terminate_all()

    def set_model(self, model):
        #print("looking for model file in " + os.getcwd())
        #print(model)
        #print(model.local_sol_path)
        
        tmpdir = model.local_sol_path
        file1 = os.path.join(tmpdir, "model_proc.pmfm")
        file2 = os.path.join(tmpdir, "model.pmfm")
        
        if os.path.exists(file1):
            self.tasks.put((3, file1), join = True)
        elif os.path.exists(file2):
            self.tasks.put((3, file2), join = True)
        else:
            assert False, "No model file in " + os.getcwd()

        self._mfem_model_bk = model

        #import tempfile, shutil
        #tmpdir = tempfile.mkdtemp()
        #model_path = os.path.join(tmpdir, 'model.pmfm')
        #model.save_to_file(model_path,
        #                   meshfile_relativepath = False)
        #self.tasks.put((3, model_path), join = True)

        #shutil.rmtree(tmpdir)
        
    def set_solfiles(self, solfiles):
        self.solfiles = solfiles
        self.tasks.put((2, solfiles))
        res = [self.results.get() for x in range(len(self.workers))]
        for x in range(len(self.workers)):
            self.results.task_done()

        isnotNone = []
        for v, c, a in res: # handle (myid, error, message)
            if c is None and v is not None:
                continue
            isnotNone.append(v)
        if len(isnotNone) == 0:
            assert False, "solution may not exist"

    def make_agents(self, name, params, **kwargs):
        super(EvaluatorMP, self).make_agents(name, params, **kwargs)
        self.tasks.put((1, name, params, kwargs))

    def load_solfiles(self, mfem_mode = None):
        self.tasks.put((4, ), join = True)

    def set_phys_path(self, phys_path):
        self.tasks.put((5, phys_path))
        
    def validate_evaluator(self, name, attr, solfiles, isFirst=False, **kwargs):
        redo_geom = False
        #if  self.solfiles is None or self.solfiles() is not solfiles:
        if (self.solfiles is None or
            self.solfiles.is_different_timestamps(solfiles)):

            if self.solfiles is None:
                print("self.solfiles is None")
            else:
                print("new file time stamp")
                
            print("new solfiles (reloading model from)")
            self.set_model(self._mfem_model_bk)
            self.set_solfiles(solfiles)
            self.load_solfiles()
            redo_geom = True

        else:
            print("same solfiles")            
        if not super(EvaluatorMP, self).validate_evaluator(name, attr, **kwargs):
            redo_geom = True
        if not self.init_done: redo_geom = True
        if not redo_geom: return
        
        if isFirst:
            self.init_done = True            
            return

        self.make_agents(self._agent_params[0],
                         attr, **kwargs)
        #self.tasks.put((6, attr, kwargs))        
        self.init_done = True
        
    def eval(self, expr, merge_flag1, merge_flag2, **kwargs):
        self.tasks.put((7, expr, kwargs), join = True)

        res = [self.results.get() for x in range(len(self.workers))]
        for x in range(len(self.workers)):
            self.results.task_done()

        for v, c, a in res: # handle (myid, error, message)
            if c is None and v is not None and a is not None:
                assert False, a
            
        # sort it so that the data is in the order of workes.rank
        res = [(x[1], x[2]) for x in sorted(res)]

        for x1, x2 in res:
            if x1 is None and x2 is not None:
                assert False, x2
            
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

    def eval_pointcloud(self, expr, **kwargs):
        self.tasks.put((10, expr, kwargs), join = True)

        res = [self.results.get() for x in range(len(self.workers))]
        for x in range(len(self.workers)):
            self.results.task_done()

        res = [x for x in res if x[-1] is not None]

        if len(res) == 0:
            return None, None, None
        
        for v, c, a in res: # handle (myid, error, message)
            if c is None and v is not None and a is not None:
                assert False, a

        ptx, data, attrs = res[0]

        for v, c, a in res[1:]:
            idx = (a != -1)
            if np.sum(idx) == 0: continue
            #print(a)
            #print(attrs)
            attrs[idx] = a[idx]
	    #print(data.shape, c.shape)                                                                                                                         
            data[idx] = c[idx]

        return ptx, data, attrs

    def eval_integral(self, expr, **kwargs):
        self.tasks.put((11, expr, kwargs), join = True)

        res = [self.results.get() for x in range(len(self.workers))]
        for x in range(len(self.workers)):
            self.results.task_done()

        res = [x for x in res if x[-1] is not None]

        if len(res) == 0:
            return None

        v = 0
        for _myid, vv, _extra in res: # handle (myid, error, message)
            if vv is not None:
                v = v + vv
            else:
                assert False, _extra

        return v
            
    def eval_probe(self, expr, xexpr, probes):
        self.tasks.put_single((8, expr, xexpr, probes), join = True)
        res = self.results.get()# for x in range(len(self.workers))]

        self.results.task_done()
        return res

    def make_probe_agents(self, name, params, **kwargs):
        print("make_probe_agents")
        super(EvaluatorMP, self).make_agents(name, params, **kwargs)
        self.tasks.put((9, name, params, kwargs))
    
    def terminate_all(self):
        #print('terminating all')      
        #num_alive = 0
        #for w in self.workers:
        #    if w.is_alive(): num_alive = num_alive + 1
        #for x in range(num_alive):
        if self.closed: return
        self.tasks.put([-1])
        self.tasks.join()
        
        self.tasks.close()
        self.results.close()
        self.results.cancel_join_thread()
        
        if self.text_queue is not None:
            self.text_queue.close()
            self.text_queue.cancel_join_thread()

        self.closed = True
        for w in self.workers:
            w.join()
        
        print('joined')
        
    def terminate_allnow(self):
        return self.terminate_all()

    

