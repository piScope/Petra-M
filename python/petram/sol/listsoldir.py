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

#
# CaseInfo (data structure to collect cases in solution directory)
#
#
class CaseInfo:
    def __init__(self, **kwargs):
        self._dict = {}
        self._dict .update(kwargs)
        self._info = ""
        self._requires_restoration = False
        
    def __repr__(self):
        keys = sorted(self._dict)
        items = ("{}={!r}".format(k, self._dict[k]) for k in keys)
        items = [self._info]+list(items)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self._dict == other._dict

    def __getattr__(self, name):
        if name == '_dict':
            if self._requires_restoration:
                self._requires_restoration = False
                self._restore(self._restoration_data)
                self._restoration_data = None
        
        if name in self._dict:
            return self._dict[name]
        else:
            raise AttributeError(name + " is not found")

    def __setstate__(self, state):
        self.__dict__["_dict"] = {}
        self.__dict__["_restoration_data"] = state[0]
        self.__dict__["_info"] = ""
        self._requires_restoration = True        
        self._restoration_data = state[0]
        
        for attr, value in state[1]:
           setattr(self, attr, value)
        
    def __getstate__(self):
        st = [(x, self.__dict__[x])
              for x in self.__dict__ if not x.startswith('_')]
        return (self._info, [(key, value) for key, value in self._dict.items()], ), st
    
    def _restore(self, restoration_data):
        self._info = restoration_data[0]
        for (key, value) in restoration_data[1]:
            self._dict[key] = value
            
    def restore(self):
        if self._requires_restoration:
            self._requires_restoration = False
            self._restore(self._restoration_data)
            self._restoration_data = None
        for x in self:
            x.restore()

    @property
    def info(self):
        return self._info

    @info.setter
    def info(self, value):
        self._info = value

    def __iter__(self):
        for key in self._dict:
            if isinstance(self._dict[key], CaseInfo):
                yield self._dict[key]

    @property
    def caselist(self):
        return list(self._dict)


def _collect_caseinfo(p):
    cases = [(int(x[5:]), x) for x in os.listdir(p) if x.startswith("case_")]
    cases = sorted(cases)
    cases = [x[1] for x in cases]
    if len(cases) == 0:
        return CaseInfo()

    info = None
    for x in os.listdir(p):
        if x.startswith("cases."):
            fname = os.path.join(p, x)
            fid = open(fname, "r")
            lines = fid.readlines()
            info = [(":".join(x.split(":")[1:])).strip() for x in lines]
            break

    kwargs = {}
    for x in cases:
        kwargs[x] = _collect_caseinfo(os.path.join(p, x))

    ret = CaseInfo(**kwargs)

    if info is not None:
        for k, x in enumerate(ret):
            x.info = info[k]
    return ret


def collect_caseinfo(p):
    from petram.mfem_config import use_parallel
    if use_parallel:
        from mpi4py import MPI
    else:
        from petram.helper.dummy_mpi import MPI

    try:
        ret = _collect_caseinfo(p)
        MPI.COMM_WORLD.Barrier()
    except:
        if use_parallel:
            MPI.COMM_WORLD.Abort()
        else:
            raise
    return ret

#
#
#


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

    cp = defaultdict(dict)  # cp["SolveStep1_TimeStep1"] = (1.0, dirname)
    for nn in os.listdir(path):
        if (nn.startswith('checkpoint_') and os.path.isdir(os.path.join(path, nn))):
            solvername = '_'.join(nn.split('_')[1:-1])
            idx = int(nn.split('_')[-1])
            if len(checkpoints[solvername]) > idx:
                cp[solvername][(idx, checkpoints[solvername][idx])] = nn
    cp.default_factory = None

    probes = gather_probes(path)

    # cases = []
    # cases = [(int(nn[5:]), nn)
    #         for nn in os.listdir(path) if nn.startswith('case')]
    # cases = [xx[1] for xx in sorted(cases)]

    cases = collect_caseinfo(path)

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


def gather_probes(path):
    probes = defaultdict(list)
    for nn in os.listdir(path):
        if nn.startswith('probe_'):
            if nn.find('.') == -1:
                signal = '_'.join(nn.split('_')[1:])
            else:
                # if int(nn.split('.')[1]) != 0: continue
                signal = '_'.join(nn.split('.')[0].split('_')[1:])
            probes[signal].append(nn)

    # sort probe files using the process number
    for key in probes:
        if len(probes[key]) > 1:
            xxx = [(int(x.split('.')[1]), x) for x in probes[key]]
            xxx = [x[1] for x in sorted(xxx)]
            probes[key] = xxx

    probes = dict(probes)
    return probes
