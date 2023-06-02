use_parallel = False
numba_debug = False
allow_python_function_coefficient = "warn"

'''
config parameter can be manipulated during a run
use accesser to get a current value.
'''
def get_use_parallel():
    return globals()['use_parallel']
def get_numba_debug():
    return globals()['numba_debug']
def get_allow_python_function_coefficient():
    return globals()['allow_python_function_coefficient']


   

