'''
  list up the contents of sol directory
   solr
   soli
   solmesh
   probe_
   checkpoint_
'''
import os
from collections import defaultdict

def gather_soldirinfo(path):

    checkpoints = {}    
    for nn in os.listdir(path):
        if (nn.startswith('checkpoint.') and
            nn.endswith('.txt')):
              fid = open(os.path.join(path, nn))
              lines = [l.strip().split(":") for l in fid.readlines()]
              lines = [(int(l[0]), float(l[1])) for l in lines]
              fid.close()
              solvername = nn.split('.')[1]
              checkpoints[solvername] = dict(lines)

    cp = defaultdict(dict)   ### cp["SolveStep1_TimeStep1"] = (1.0, dirname)
    for nn in os.listdir(path):
        if (nn.startswith('checkpoint_') and os.path.isdir(os.path.join(path, nn))):
            solvername = '_'.join(nn.split('_')[1:-1])
            idx = int(nn.split('_')[-1])
            cp[solvername][(idx, checkpoints[solvername][idx])] = nn
    cp.default_factory=None

    probes = {}
    for nn in os.listdir(path):
        if nn.startswith('probe_'):
            if nn.find('.') == -1:
               signal = '_'.join(nn.split('_')[1:]) 
            else:
                if int(nn.split('.')[1]) != 0: continue
                signal = '_'.join(nn.split('.')[0].split('_')[1:])
            probes[signal] = nn

    cases = []
    for nn in os.listdir(path):
        if nn.startswith('case'):
            cases.append(nn)
    '''        
    meshes = defaultdict(list)
    for nn in os.listdir(path):
        if nn.startswith('solmesh_'):
            if nn.find('.') == -1:
                idx = int(nn.split('_')[1])
            else:
                idx = int(nn.split('.')[0].split('_')[1])
            meshes[idx].append(nn)

    solr = defaultdict(list)
    for nn in os.listdir(path):
        if nn.startswith('solr_'):
            if nn.find('.') == -1:
                idx = int(nn.split('_')[-1])
                name= '_'.join(nn.split('_')[1:-1])
            else:
                idx = int(nn.split('.')[0].split('_')[-1])
                name= '_'.join(nn.split('.')[0].split('_')[1:-1])
            solr[(name, idx)].append(nn)
    solr.default_factory = None

    soli = defaultdict(list)
    for nn in os.listdir(path):
        if nn.startswith('soli_'):
            if nn.find('.') == -1:
                idx = int(nn.split('_')[-1])
                name= '_'.join(nn.split('_')[1:-1])
            else:
                idx = int(nn.split('.')[0].split('_')[-1])
                name= '_'.join(nn.split('.')[0].split('_')[1:-1])
            soli[(name, idx)].append(nn)
    soli.default_factory = None

    extra = defaultdict(list)
    for nn in os.listdir(path):
        if nn.startswith('sol_extended'):
            aa = nn.split('.')
            name= aa[1]
            extra[name].append(nn)
    extra.default_factory = None
    '''
    soldirinfo = {'checkpoint': dict(cp),
                  'probes': dict(probes),
                  'cases': cases}
    return soldirinfo

def gather_soldirinfo_s(path):
    info = gather_soldirinfo(path)
    
    import cPickle, binascii
    data = binascii.b2a_hex(cPickle.dumps(info))
    
    return data
