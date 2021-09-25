'''
  list up the contents of sol directory
   solr
   soli
   solmesh
   probe_
   checkpoint_
'''
import os
from os.path import expanduser
from collections import defaultdict

def gather_soldirinfo(path):
    path = expanduser(path)
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
            if len(checkpoints[solvername]) > idx:            
                cp[solvername][(idx, checkpoints[solvername][idx])] = nn
    cp.default_factory=None

    probes = defaultdict(list)
    for nn in os.listdir(path):
        if nn.startswith('probe_'):
            if nn.find('.') == -1:
               signal = '_'.join(nn.split('_')[1:]) 
            else:
                #if int(nn.split('.')[1]) != 0: continue
                signal = '_'.join(nn.split('.')[0].split('_')[1:])
            probes[signal].append(nn)

    # sort probe files using the process number
    for key in probes:
        if len(probes[key]) > 1:
             xxx = [(int(x.split('.')[1]), x) for x in	probes[key]]
             xxx = [x[1] for x in sorted(xxx)]
             probes[key] = xxx            

    probes = dict(probes)
    cases = []
    cases = [(int(nn[5:]), nn) for nn in os.listdir(path) if nn.startswith('case')]
    cases = [xx[1] for xx in sorted(cases)]

    soldirinfo = {'checkpoint': dict(cp),
                  'probes': dict(probes),
                  'cases': cases}
    return soldirinfo

def gather_soldirinfo_s(path):
    try:
        info = gather_soldirinfo(path)
        result = (True, info)
    except:
        import traceback
        result = (False, traceback.format_exc())
        
    import petram.helper.pickle_wrapper as pickle    
    import binascii
    
    data = binascii.b2a_hex(pickle.dumps(result))
    
    return data
