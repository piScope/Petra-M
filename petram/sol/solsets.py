import six

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
        
class Solsets(object):
    '''
    Solsets: bundle of GridFunctions

      methes: names, meshes, gfr, gfi
    '''
    def __init__(self, solfiles):
        solfiles = solfiles.set
        object.__init__(self)
        self.set = []
        import mfem.ser as mfem

        for x, solf, in solfiles:
            m = mfem.Mesh(str(x), 1, 0)  ### what is this refine = 0 !?
            m.ReorientTetMesh()
            s = {}
            for key in six.iterkeys(solf):
               fr, fi =  solf[key]
               solr = (mfem.GridFunction(m, str(fr)) if fr is not None else None)
               soli = (mfem.GridFunction(m, str(fi)) if fi is not None else None)
               s[key] = (solr, soli)
            self.set.append((m, s))

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
    
    solsets = []
    x = mfiles[0]
    suffix = '' if len(x.split('.')) == 1 else '.'+x.split('.')[-1]

    files = [x for x in solrfile if x.endswith(suffix)]
    names = [x.split('.')[0] for x in files]    
    names = ['_'.join(x.split('_')[1:3]) for x in names]
    # names = ['E_0'], meaning  E defined on mesh 0

    solfiles = []
    for x in mfiles:       
       suffix = '' if len(x.split('.')) == 1 else '.'+x.split('.')[-1]
       if idx is not None:
          if int(suffix) in idx: continue
       x = os.path.join(pathm, x)
       sol = {}
       for name in names:
          solr = (os.path.join(path, 'solr_'+name +suffix)
                  if ('solr_'+name+suffix) in solrfile else None)
          soli = (os.path.join(path,'soli_'+name+suffix)
                  if ('soli_'+name+suffix) in solifile else None)
          sol[name] = (solr, soli)
       if sol[names[-1]][0] is None: continue # if real sol is None, skip this
       solfiles.append([x, sol])          
    return Solfiles(solfiles)

def read_solsets(path, idx = None):
    solfiles = find_solfiles(path, idx)
    return Solsets(solfiles)
read_sol = read_solsets
'''    
def read_solsets(path, idx = None):
    import os

    files = os.listdir(path)
    mfiles = [x for x in files if x.startswith('solmesh')]
    solrfile = [x for x in files if x.startswith('solr')]
    solifile = [x for x in files if x.startswith('soli')]

    if len(mfiles) == 0:
        files2 = os.listdir(os.path.dirname(path))
        mfiles = [x for x in files2 if x.startswith('solmesh')]
        pathm = os.path.dirname(path)
    else:
        pathm = path
    
    import mfem.ser as mfem

    solsets = []
    x = mfiles[0]
    suffix = '' if len(x.split('.')) == 1 else '.'+x.split('.')[-1]


    files = [x for x in solrfile if x.endswith(suffix)]
    names = [x.split('.')[0] for x in files]    
    names = ['_'.join(x.split('_')[1:3]) for x in names]
    # names = ['E_0'], meaning  E defined on mesh 0

    for x in mfiles:       
       suffix = '' if len(x.split('.')) == 1 else '.'+x.split('.')[-1]
       if idx is not None:
          if int(suffix) in idx: continue
       x = os.path.join(pathm, x)
       m = mfem.Mesh(x, 1, 1)
       m.ReorientTetMesh()
       sol = {}
       for name in names:
          solr = (mfem.GridFunction(m,
                                    os.path.join(path, 'solr_'+name +suffix))
                  if ('solr_'+name+suffix) in solrfile else None)
          soli = (mfem.GridFunction(m,
                                    os.path.join(path,'soli_'+name+suffix))
                  if ('soli_'+name+suffix) in solifile else None)
          sol[name] = (solr, soli)
       solsets.append([m, sol])
    return Solsets(solsets)
'''

