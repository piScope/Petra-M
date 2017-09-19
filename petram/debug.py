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

debug_mode = 1
debug_modes = {}
debug_default_level = 1
debug_essentail_bc = False
debug_memory = False

def set_debug_level(level):
    s = 1 if level == 0 else level/abs(level)
    globals()['debug_default_level'] =   s*(abs(level) % 4)
    globals()['debug_essential_bc'] =  abs(level) & (1 << 2) != 0
    globals()['debug_memory'] =  abs(level) & (1 << 3) != 0

def dprint(*args):
    s = ''
    for item in args:
      s = s + ' ' +str(item)
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
        if 'stack' in kargs: traceback.print_stack()
        s = ''

        from petram.mfem_config import use_parallel
        if use_parallel:
            from mpi4py import MPI
            myid     = MPI.COMM_WORLD.rank
        else:
            myid = 0
        
        for item in args:
            s = s + ' ' + str(item)
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
    def __init__(self, name, head_only = False):
        self.name = name
        self.head_only = head_only
    def __call__(self, *args, **kargs):
        if 'stack' in kargs: traceback.print_stack()
        s = ''
        try:
           from mpi4py import MPI
           myid     = MPI.COMM_WORLD.rank
        except ImportError:
           myid = 0

        if self.head_only and myid != 0: return
        for item in args:
            s = s + ' ' + str(item)
        print(str(self.name)+'(' + str(myid)+')::'+s)

def regular_print(n, head_only = False):
    return RPrint(n, head_only)
    
def prints(n):
    return DPrint(n, 1), DPrint(n, 2),DPrint(n, 3)

def set_level(name, level):
    debug_modes[name] = level

def init_dprints(name, level=None):
    if level is not None: set_level(name, level)
    return prints(name)

import resource    
def format_memory_usage(point="memory usage"):
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return '''%s: usertime=%s systime=%s mem=%s mb
           '''%(point,usage[0],usage[1],
                (usage[2]*resource.getpagesize())/1000000.0 )

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

    
