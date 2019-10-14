from __future__ import print_function

import os
import six
import numpy as np

class Solfiles(object):
    '''
    solfile list container ( used to refer it weakly)
    '''
    def __init__(self, l):
        if isinstance(l, Solfiles):
            self.set = l.set            
        else:
            self.set = l
    def __len__(self):
        return len(self.set)
    def __getitem__(self, idx):
        return Solfiles(self.set[idx])
    @property
    def path(self):
        return os.path.dirname(self.set[0][0][0])
    
class MeshDict(dict):
    pass
        
class Solsets(object):
    '''
    Solsets: bundle of GridFunctions

      methes: names, meshes, gfr, gfi
    '''
    def __init__(self, solfiles, refine=0):
        def fname2idx(t):
           i = int(os.path.basename(t).split('.')[0].split('_')[-1])
           return i
        solfiles = solfiles.set
        object.__init__(self)
        self.set = []
        import mfem.ser as mfem
        
        for meshes, solf, in solfiles:
            idx = [fname2idx(x) for x in meshes]
            meshes = {i:  mfem.Mesh(str(x), 1, refine) for i, x in zip(idx, meshes)}
            meshes=MeshDict(meshes) # to make dict weakref-able
            ### what is this refine = 0 !?
            for i in idx:
                meshes[i].ReorientTetMesh()
                meshes[i]._emesh_idx = i
            s = {}
            for key in six.iterkeys(solf):
               fr, fi =  solf[key]
               i = fname2idx(fr)
               m = meshes[i]

               solr = (mfem.GridFunction(m, str(fr)) if fr is not None else None)
               soli = (mfem.GridFunction(m, str(fi)) if fi is not None else None)
               if solr is not None: solr._emesh_idx = i
               if soli is not None: soli._emesh_idx = i

               s[key] = (solr, soli)
            self.set.append((meshes, s))

    def __len__(self):
        return len(self.set)

    def __iter__(self):
        return self.set.__iter__()

    @property
    def meshes(self):
        return [x[0] for x in self.set]
    @property
    def names(self):
        ret = []
        for x in self.set:
            ret.extend(x[1].keys())
        return tuple(set(ret))

    def gfr(self, name):
        return [x[1][name][0] for x in self.set]

    def gfi(self, name):
        return [x[1][name][1] for x in self.set]

def find_solfiles(path, idx = None):
    import os

    files = os.listdir(path)
    mfiles = [x for x in files if x.startswith('solmesh')]
    if len(mfiles) == 0:
        return None
    solrfile = [x for x in files if x.startswith('solr')]
    solifile = [x for x in files if x.startswith('soli')]

    if len(mfiles) == 0:
        '''
        mesh file may exist one above...shared among parametric
        scan...
        '''
        files2 = os.listdir(os.path.dirname(path))
        mfiles = [x for x in files2 if x.startswith('solmesh')]
        pathm = os.path.dirname(path)
    else:
        pathm = path

    solfiles = []    
    x = mfiles[0]
    suffix_list = list(set(['' if len(x.split('.')) == 1 else '.'+x.split('.')[-1]
                            for x in mfiles]))
    for s in suffix_list:
        meshes = [x for x in mfiles if (len(x.split('.')) == 1 and s == '') or
                                        x.endswith(s)]
        meshes = [os.path.join(pathm, x) for x in meshes]
        solrs  = [x for x in solrfile if (len(x.split('.')) == 1 and s == '') or
                                        x.endswith(s)]
        solis  = [x for x in solifile if (len(x.split('.')) == 1 and s == '') or
                                        x.endswith(s)]
        names = ['_'.join(x.split('.')[0].split('_')[1:]) for x in solrs]

        sol = {}
        for n in names:
            print('solr_'+ n + s)
            solr = (os.path.join(path, 'solr_'+ n + s)
                  if ('solr_'+ n + s) in solrfile else None)
            soli = (os.path.join(path, 'soli_'+ n + s)
                  if ('soli_'+ n + s) in solifile else None)
            
            if solr is None: continue
            sol[n] = (solr, soli)
        solfiles.append([meshes, sol])                  
    return Solfiles(solfiles)

def read_solsets(path, idx = None, refine=0):
    solfiles = find_solfiles(path, idx)
    return Solsets(solfiles, refine=refine)
read_sol = read_solsets


