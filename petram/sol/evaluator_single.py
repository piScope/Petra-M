from __future__ import print_function

import time
import numpy as np
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
        #self.solvars = WKD()
        self.solvars = {}
        self.agents = {}
        self.physpath = ''
        self.init_done = False
        self.failed = False
        super(EvaluatorSingle, self).__init__()        
        
    def set_solfiles(self, solfiles):
        self.solfiles = solfiles
        # make sure solvars is empty and weakref does not go away.
        self._soliles = solfiles
        self.solvars = {}        
        
    def set_phys_path(self, phys_path):
        self.phys_path = phys_path

    def validate_evaluator(self, name, attr, solfiles, isFirst=False, **kwargs):
        redo_geom = False
        if (self.solfiles is None or 
            self.solfiles.is_different_timestamps(solfiles)):
            self.set_solfiles(solfiles)
            print("new solfiles")
            redo_geom = True
        else:
            print("same solfiles")            
            
        if not super(EvaluatorCommon, self).validate_evaluator(name, attr, **kwargs):
            print("different_setting")
            redo_geom = True
        if not self.init_done: redo_geom = True
        if not redo_geom: return
        
        if isFirst:
            self.init_done = True            
            return

        solvars = self.load_solfiles()
        self.make_agents(self._agent_params[0],
                         attr, **kwargs)

        #for key in list(self.agents):
        #    evaluators = self.agents[key]
        #    for o in evaluators:
        #        o.preprocess_geometry([key], **kwargs)
        self.init_done = True
                
    def eval(self, expr, merge_flag1, merge_flag2, **kwargs):
        if self.phys_path == '': return None, None
        
        phys = self.mfem_model()[self.phys_path]
        solvars = self.load_solfiles()
        if solvars is None: return None, None

        export_type = kwargs.get('export_type', 1)
        
        data = []
        attrs = []
        offset = 0
        def omit_none(l):
            return [x for x in l if x is not None]
        for key in six.iterkeys(self.agents): # scan over battr
            if merge_flag2: offset = 0
            vdata = [] # vertex
            cdata = [] # data
            adata = [] # array idx     
            data.append([])
            attrs.append(key)                                  
            evaluators = self.agents[key]
            for o, solvar in zip(evaluators, solvars): # scan over sol files
                v, c, a = o.eval(expr, solvar, phys, **kwargs)
                if v is None:
                    v = None; c = None; a = None
                else:
                    if merge_flag1: a = a + offset
                    offset = offset + c.shape[0]
                    vdata.append(v)
                    cdata.append(c)
                    adata.append(a)                
                data[-1].append((v, c, a))
            if merge_flag1:
                if len(vdata) != 0:
                    data[-1]  = [(np.vstack(omit_none(vdata)),
                                  np.hstack(omit_none(cdata)),
                                  np.vstack(omit_none(adata)))]
                else:
                    data = data[:-1]  # remove empty tupple
                    attrs = attrs[:-1]
        if export_type == 2: return data, attrs                     
        ## normally this is good option.
        if merge_flag1 and not merge_flag2:
            vdata = np.vstack([x[0][0] for x in data])
            cdata = np.hstack([x[0][1] for x in data])
            adata = np.vstack([x[0][2] for x in data])                                
            data = [(vdata, cdata, adata)]
        elif merge_flag1:
            data0 = []
            for x in data: data0.extend(x)
            data = data0
        elif not merge_flag2:
            keys = self.agents.keys()
            data0 = []
            attr = []
            for idx, o in enumerate(evaluators): # for each file
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
        if self.phys_path == '': return None, None
        
        phys = self.mfem_model()[self.phys_path]
        solvars = self.load_solfiles()
        if solvars is None: return None, None

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
           v, c, a = o.eval(expr, solvar, phys, **kwargs)
           if v is None:
               v = None; c = None; a = None
           else:
               vdata.append(v)
               cdata.append(c)
               adata.append(a)

        if len(adata)==0: return None, None

        ptx  = vdata[0]
        ddim = cdata[0].shape[1:]  # data dim

        shape = adata[0].shape
        shape_d = sum((shape, ddim), ())
                      
        attrs = np.zeros(shape, dtype=int)-1

        #print(shape_d)
        data  = np.zeros(shape_d, dtype=cdata[0].dtype)

        #print("data shape", data.shape)
                
        for v, c, a in zip(vdata, cdata, adata):
            idx = (a != -1)
            if np.sum(idx) == 0: continue
            attrs[idx] = a[idx]
            data[idx] = c
        return ptx, data, attrs

    def eval_integral(self, expr, **kwargs):
        if self.phys_path == '':
            return None, None

        phys = self.mfem_model()[self.phys_path]
        solvars = self.load_solfiles()
        if solvars is None:
            return None

        key = list(self.agents)[0]
        evaluators = self.agents[key]
        
        data = []
        
        v = 0
        num = 0
        for o, solvar in zip(evaluators, solvars): # scan over sol files
            kwargs['num'] = num           
            vv = o.eval_integral(expr, solvar, phys, **kwargs)
            num = num + 1
            if vv is None:
                pass
            else:
                v = v + vv

        return v

    def eval_probe(self, expr, xexpr, probes):
        if self.phys_path == '':
            return None, None, None

        phys = self.mfem_model()[self.phys_path]

        evaluator = self.agents[1][0]
        xval, val = evaluator.eval_probe(expr, xexpr, probes, phys)
        return (0, xval, val)

    def terminate_allnow(self):
        pass
