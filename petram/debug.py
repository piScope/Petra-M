from __future__ import print_function
import resource
import time
import textwrap
#####################################
#
# debug.py
#
#    provide a simple interface to make debug prints
#    allow for controling turn on and off of debug
#    print for each module
#
# Usage:
#    (in import section)
#    import ifigure.utils.debug as debug
#    dprint1, dprint2, dprint3 = debug.init_dprints('ArgsParser', level=0)
#
#    (then use it as follows)
#    debug.set_level('ArgsParser', 1)  # set level for ArsgParser 1
#    dprint1('hogehogehoge')           # print something
#
#    level 1 (dprint1) : usr feedback which will be turn on normally
#    level 2 (dprint2) : first level of debug print
#    level 3 (dprint3) : second level of debug print
#    setting debug_default_level to 0 will turn off all error print
#    (silent mode)


import traceback
import six

debug_mode = 1
debug_modes = {}
debug_default_level = 1
debug_essential_bc = False
debug_memory = False
debug_evaluator_mp = False
trim_debug_print = True

max_txt = 70*2


def set_debug_level(level):
    s = 1 if level == 0 else level/abs(level)
    globals()['debug_default_level'] = s*(abs(level) % 4)
    globals()['debug_essential_bc'] = abs(level) & (1 << 2) != 0
    globals()['debug_memory'] = abs(level) & (1 << 3) != 0


def dprint(*args, **kargs):
    s = ''
    for item in args:
        s = s + ' ' + str(item)

    if trim_debug_print and "notrim" not in kargs:
        s = textwrap.shorten(s, width=max_txt, placeholder='...')

    if debug_mode != 0:
        import sys
        print('DEBUG('+str(debug_mode)+')::'+s)


def find_by_id(_id_):
    '''
    find an object using id 
    '''
    import gc
    for obj in gc.get_objects():
        if id(obj) == _id_:
            return obj
    raise Exception("No found")


class DPrint(object):
    def __init__(self, name, level):
        self.name = name
        self.level = level

    def __call__(self, *args, **kargs):
        if 'stack' in kargs:
            traceback.print_stack()
        s = ''

        from petram.mfem_config import use_parallel
        if use_parallel:
            from mpi4py import MPI
            myid = MPI.COMM_WORLD.rank
        else:
            myid = 0

        for item in args:
            s = s + ' ' + str(item)

        if trim_debug_print and "notrim" not in kargs:
            s = textwrap.shorten(s, width=max_txt, placeholder='...')

        if self.name in debug_modes:
            if debug_modes[self.name] >= self.level:
                print('DEBUG('+str(self.name)+' ' + str(myid)+')::'+s)
        else:
            if debug_default_level < 0:
                if abs(debug_default_level) >= self.level:
                    print('DEBUG('+str(self.name)+' ' + str(myid)+')::'+s)
            else:
                if (abs(debug_default_level) >= self.level and
                        myid == 0):
                    print('DEBUG('+str(self.name)+' ' + str(myid)+')::'+s)


class RPrint(object):
    def __init__(self, name, head_only=False):
        self.name = name
        self.head_only = head_only

    def __call__(self, *args, **kargs):
        if 'stack' in kargs:
            traceback.print_stack()
        s = ''
        try:
            from mpi4py import MPI
            myid = MPI.COMM_WORLD.rank
        except ImportError:
            myid = 0

        if self.head_only and myid != 0:
            return
        for item in args:
            s = s + ' ' + str(item)

        if trim_debug_print and "notrim" not in kargs:
            s = textwrap.shorten(s, width=max_txt, placeholder='...')
        print(str(self.name)+'(' + str(myid)+')::'+s)


def regular_print(n, head_only=False):
    return RPrint(n, head_only)


def prints(n):
    return DPrint(n, 1), DPrint(n, 2), DPrint(n, 3)


def set_level(name, level):
    debug_modes[name] = level


def init_dprints(name, level=None):
    if level is not None:
        set_level(name, level)
    return prints(name)


def format_memory_usage(point="memory usage"):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return '''%s: usertime=%s systime=%s mem=%s mb
           ''' % (point, usage[0], usage[1], memory_usage_resource())
#                (usage[2]*resource.getpagesize())/1000000.0 )


def memory_usage_resource():
    import resource
    import sys
    rusage_denom = 1024.
    if sys.platform == 'darwin':
        # ... it seems that in OSX the output is different units ...
        rusage_denom = rusage_denom * rusage_denom
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
    return mem


try:
    import guppy
    hasGUPPY = True
except ImportError:
    hasGUPPY = False

if hasGUPPY:
    def format_heap__usage():
        from guppy import hpy
        h = hpy()
        return h.heap()
else:
    def format_heap__usage():
        pass


def use_profiler(method):
    def method2(self, *args, **kwargs):
        if self.use_profiler:
            import cProfile
            import pstats
            from six import StringIO
            pr = cProfile.Profile()
            pr.enable()

        val = method(self, *args, **kwargs)

        if self.use_profiler:
            pr.disable()
            s = StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            # print(s.getvalue())

            from petram.mfem_config import use_parallel
            if use_parallel:
                from mpi4py import MPI
                myid = MPI.COMM_WORLD.rank
                smyid = '.''{:0>6d}'.format(myid)
            else:
                smyid = ''
            ps.dump_stats("cProfile_"+self.name()+".out"+smyid)
        return val
    return method2


def flush_stdout(method):
    def method2(self, *args, **kwargs):
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        val = method(self, *args, **kwargs)
        return val

    return method2


class ConvergenceError(Exception):
    """Base class for exceptions in this module."""
    pass

    def __init__(self, message="Failed to converge"):
        self.message = message


'''
a decorator to time a method
'''


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        print("entering:", method.__name__.upper())
        log = kw.pop('log_time', None)
        result = method(*args, **kw)
        te = time.time()
        if log is not None:
            name = log.get('log_name', method.__name__.upper())
            log[name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        print("done:", method.__name__.upper())
        return result
    return timed


def handle_allow_python_function_coefficient(message):
    from petram.mfem_config import get_allow_python_function_coefficient

    dprint1 = DPrint("Python mode check", 1)

    if get_allow_python_function_coefficient() == "warn":
        import traceback
        traceback.print_stack()
        dprint1(message)
    elif get_allow_python_function_coefficient() == "ignore":
        pass
    elif get_allow_python_function_coefficient() == "always use Python coeff.":
        pass
    elif get_allow_python_function_coefficient() == "error":
        assert False, message
