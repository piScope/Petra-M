from numbers import Number
import numpy as np
import os

from petram.mfem_config import use_parallel
if use_parallel:
    from mpi4py import MPI
    num_proc = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank
    smyid = '.'+'{:0>6d}'.format(myid)
else:
    myid = 0
    smyid = ''


def list_probes(dir):
    od = os.getcwd()
    os.chdir(dir)
    filenames = [name for name in os.listdir() if name.startswith('probe_')]

    if use_parallel:
        filenames = [f for f in filenames if '.'+f.split('.')[-1] == smyid]

    probenames = [n[6:] for n in filenames]
    os.chdir(od)

    return filenames, probenames


def load_probe(name):
    fid = open(name, 'r')
    format = int(fid.readline().split(':')[-1])

    if format == 0:
        value = load_format_0(fid)
        ydata = np.squeeze(value[:, 1:].transpose())
        xdata = value[:, 0]
        xdata = {'time': xdata}
    if format == 1:
        xdata, ydata, names = load_format_1(fid)
        xdata = {n: xdata[k] for k, n in enumerate(names)}

    fid.close()

    return xdata, ydata


def load_probes(dir, names):
    data = [load_probe(os.path.join(dir, n)) for n in names]

    xdata = data[0][0]
    ydata = [x[1] for x in data]

    # print("ydata shape", [x[1].shape for x in data])
    ydata = np.vstack(ydata)
    ydata = np.squeeze(ydata)

    return xdata, ydata


def load_format_0(fid):
    lines = fid.readlines()
    data = [[float(x) for x in l.split(',') if x.strip() != ''] for l in lines]
    data = np.array(data)
    return data


def load_format_1(fid):
    from petram.helper.variables import Variable, var_g

    xsize = int(fid.readline())
    xnames = [x.strip() for x in fid.readline().split(',')]

    lines = fid.readlines()

    data = [eval(l, var_g, {}) for l in lines]

    xdata = np.array([x[:xsize] for x in data]).transpose()
    ydata = np.array([x[xsize:] for x in data]).transpose()

    return xdata, ydata, xnames


class Probe(object):
    def __new__(cls, *args, **kargs):
        root_only = kargs.pop("root_only", False)
        if myid == 0 or not root_only:
            return object.__new__(cls)
        else:
            return None

    def __init__(self, name, idx=-1, xnames=None, **kwargs):
        '''
        idx is idx in blockvector. could be -1 in such case, Probe can not load data from
        sol. This is used in Parametric to gather all probe later
        '''
        self.name = name
        self.xnames = ['time'] if xnames is None else xnames
        self.sig = []
        self.t = []
        self.idx = idx
        self.finalized = False

    def write_file(self, filename=None, format=1, nosmyid=False):
        valid = self.finalize()
        if not valid:
            return

        if filename is None:
            if nosmyid:
                filename = 'probe_'+self.name
            else:
                filename = 'probe_'+self.name + smyid

        fid = open(filename, 'w')

        if format == 0:
            fid.write("format : 0\n")
        else:
            fid.write("format : 1\n")
            fid.write(str(len(self.time[0]))+"\n")
            fid.write(','.join(self.xnames)+"\n")

        for x, t in zip(self.sig_f, self.time):
            txt1 = ', '.join([str(xx) for xx in t])
            txt2 = ', '.join([str(xx) for xx in x])
            fid.write(txt1 + ', ' + txt2 + "\n")
        fid.close()

    def append_sol(self, sol, t=0.0):
        if self.idx != -1:
            self.sig.append(np.atleast_1d(sol[self.idx].toarray().flatten()))
            self.t.append(np.atleast_1d(t))

    def append_value(self, value, t=0.0):
        self.sig.append(np.atleast_1d(value))
        self.t.append(np.atleast_1d(t))

    def current_value(self, sol):
        return np.atleast_1d(sol[self.idx].toarray().flatten())

    def print_signal(self):
        if not self.finalized:
            self.finalize()

    def finalize(self):
        if len(self.sig) == 0:
            self.sig_f = -1
            self.valid = False
        else:
            shapes = [x.shape for x in self.sig]
            if np.all([len(x.shape) == 1 for x in self.sig]):
                self.sig_f = np.vstack(self.sig)
            else:
                shape2 = max([np.prod(x.shape) for x in self.sig])
                iscomplex = np.any([np.iscomplexobj(x) for x in self.sig])
                if iscomplex:
                    data = np.full((len(self.sig), shape2),
                                   np.nan, dtype=self.sig[0].dtype)
                else:
                    data = np.full((len(self.sig), shape2),
                                   np.nan, dtype=self.sig[0].dtype)
                for k, x in enumerate(self.sig):
                    l = np.prod(x.shape)
                    data[k, :l] = x.flatten()

                self.sig_f = data

            self.time = self.t
            self.valid = self.sig_f.size != 0
            # self.valid = True
        self.finalized = True
        return self.valid


class ProbeSignal():
    def __init__(self, path, load_names):
        """
        Probe Signal Used for Visualization
        Instatiated by ProbeSignals, which is namedtuple containing ProbeSignal
        """
        self._load_names = load_names
        self._path = path
        # not loaded yet
        self._array = None
        self._xarrays = None
        self._xnames = []

    def __repr__(self):
        return "ProbeSignal("+str(self._load_names)+")"

    def _load(self):
        if self._array is None:
            xdata, ydata = load_probes(self._path, self._load_names)
            self._array = np.asarray(ydata)
            self._xarrays = xdata

    def __getattr__(self, name):
        if name.startswith('__array'):
            # Allow default object behavior for array interface attributes
            return object.__getattr__(self, name)

        self._load()
        if name in self._xarrays:
            return self._xarrays[name]

    @property
    def x(self):
        self._load()
        if "x" in self._xarrays:
            return self._xarrays["x"]
        else:
            key = list(self._xarrays)[0]
            return self._xarrays[key]

    @property
    def t(self):
        self._load()
        if "t" in self._xarrays:
            return self._xarrays["t"]
        else:
            key = list(self._xarrays)[0]
            return self._xarrays[key]

    def __array__(self, dtype=None, copy=None):
        self._load()
        if dtype:
            return self._array.astype(dtype)
        return self._array

    # --- Forward common array behavior ---
    def __getitem__(self, key):
        self._load()
        return self._array[key]

    def __len__(self):
        self._load()
        return len(self._array)

    @property
    def shape(self):
        self._load()
        return self._array.shape

    @property
    def dtype(self):
        self._load()
        return self._array.dtype

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        #
        # called for example np.sin(z)
        #
        scalars = []
        for x in inputs:
            if isinstance(input, Number):
                scalars.append(input)
            if isinstance(x, ProbeSignal):
                scalars.append(np.asarray(x))
            else:
                scalars.append(x)
        return getattr(ufunc, method)(*scalars, **kwargs)

class ProbleSignalCollection:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            raise AttributeError(name + " is not found")

    def __iter__(self):
        for key in self.__dict__:
            if isinstance(self.__dict__[key], ProbleSignalCollection):
                yield self.__dict__[key]

def _collect_probesignals(p):
    cases = [(int(x.split("_")[-1]), x) for x in os.listdir(p) if x.startswith("case_")]
    cases = sorted(cases)
    cases = [x[1] for x in cases]

    cps = [(int(x.split("_")[-1]), x) for x in os.listdir(p) if x.startswith("cp_")]
    cps = sorted(cps)
    cps = [x[1] for x in cps]

    from petram.sol.listsoldir import gather_probes
    probes = gather_probes(p)

    kwargs = {}
    for x in cases:
        kwargs[x] = _collect_probesignals(os.path.join(p, x))
    for x in cps:
        kwargs[x] = _collect_probesignals(os.path.join(p, x))
    for x in probes:
        kwargs[x] = ProbeSignal(p, probes[x])
    return ProbleSignalCollection(**kwargs)

def collect_probesignals(p):
    if use_parallel:
        MPI.COMM_WORLD.Barrier()
    return _collect_probesignals(p)
