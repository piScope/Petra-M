import sys
import time
import numpy as np
import parser
import weakref
import traceback
import subprocess as sp
import cPickle
import binascii
from weakref import WeakKeyDictionary as WKD
from weakref import WeakValueDictionary as WVD

from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

import multiprocessing as mp
from petram.sol.evaluators import Evaluator, EvaluatorCommon
from petram.sol.evaluator_mp import EvaluatorMPChild, EvaluatorMP

import thread
from threading import Timer, Thread
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty  # python 3.x
    
ON_POSIX = 'posix' in sys.builtin_module_names

def enqueue_output(p, queue, prompt):
    while True:
        line = p.stdout.readline()
        if line ==  (prompt + '\n'): break
        queue.put(line)
        if p.poll() is not None: return
    queue.put("??????")
    
def run_and_wait_for_prompt(p, prompt, verbose=True):    
    q = Queue()
    t = Thread(target=enqueue_output, args=(p, q, prompt))
    t.daemon = True # thread dies with the program
    t.start()

    lines = [" "]
    alive = True
    while lines[-1] != "??????":
        time.sleep(0.01)                
        try:  line = q.get_nowait() # or q.get(timeout=.1)
        except Empty:
            pass
            #print('no output yet' + str(p.poll()))
        else: # got line
            lines.append(line)
        if p.poll() is not None:
            alive = False
            print('proces terminated')
            break
    if verbose:
        print(lines)
    return lines[:-1], alive

def run_with_timeout(timeout, default, f, *args, **kwargs):
    if not timeout:
        return f(*args, **kwargs)
    try:
        timeout_timer = Timer(timeout, thread.interrupt_main)
        timeout_timer.start()
        result = f(*args, **kwargs)
        return result
    except KeyboardInterrupt:
        return default
    finally:
        timeout_timer.cancel()
        
def wait_for_prompt(p, prompt = '?', verbose = True):
    print("waiting for prompt")
    output = []
    alive = True
    while('True'):
        time.sleep(0.01)        
        line = run_with_timeout(1.0, '', p.stdout.readline)
        if p.poll() is not None:
            alive = False
            print('proces terminated')
            break
        if line.startswith(prompt): break
        output.append(line)
    if verbose:
        for x in output:
            print(x.strip())
        print("process active :" + str(alive))
    print("got prompot")
    return output, alive

def wait_for_prompt(p, prompt = '?', verbose = True):
    return run_and_wait_for_prompt(p, prompt, verbose=verbose)
        
def start_connection(host = 'localhost', num_proc = 2, user = ''):
    if user != '': user = user+'@'
    p= sp.Popen("ssh " + user + host + " 'printf $PetraM'", shell=True,
                stdout=sp.PIPE)
    ans = p.stdout.readlines()[0].strip()
    command = ans+'/bin/evalsvr'
    p = sp.Popen(['ssh', user + host, command], stdin = sp.PIPE,
                 stdout=sp.PIPE, stderr=sp.STDOUT,
                 close_fds = ON_POSIX,
                 universal_newlines = True)

    data, alive = wait_for_prompt(p, prompt = 'num_proc?')
    p.stdin.write(str(num_proc)+'\n')
    out, alive = wait_for_prompt(p)
    return p

def connection_test(host = 'localhost'):
    '''
    note that the data after process is terminated may be lost.
    '''
    p = start_connection(host = host, num_proc = 2)
    for i in range(5):
       p.stdin.write('test'+str(i)+'\n')
       out, alive = wait_for_prompt(p)
    p.stdin.write('e\n')
    out, alive = wait_for_prompt(p)

from petram.sol.evaluator_mp import EvaluatorMP
class EvaluatorServer(EvaluatorMP):
    def __init__(self, nproc = 2, logfile = False):
        return EvaluatorMP.__init__(self, nproc = nproc,
                                    logfile = logfile)
    
    def set_model(self, soldir):
        import os
        model_path = os.path.join(soldir, 'model.pmfm')
        if not os.path.exists(model_path):
           if 'case' in os.path.split(soldir)[-1]:
               model_path = os.path.join(os.path.dirname(soldir), 'model.pmfm')
        if not os.path.exists(model_path):
            assert False, "Model File not found: " + model_path
            
        self.tasks.put((3, model_path), join = True)

    
class EvaluatorClient(Evaluator):
    def __init__(self, nproc = 2, host = 'localhost',
                       soldir = '', user = ''):
        self.init_done = False        
        self.soldir = soldir
        self.solfiles = None
        self.nproc = nproc
        self.p = start_connection(host =  host,
                                  num_proc = nproc,
                                  user = user)

    def __del__(self):
        self.terminate_all()
        self.p = None

    def __call_server(self, name, *params, **kparams):
        if self.p is None: return
        
        command = [name, params, kparams]
        data = binascii.b2a_hex(cPickle.dumps(command))
        print("Sending request", command)
        self.p.stdin.write(data + '\n')
        
        output, alive = wait_for_prompt(self.p, verbose = False)
        if not alive:
           self.p = None
           return
        response = output[-1].strip()
        try:
            result = cPickle.loads(binascii.a2b_hex(response))
        except:
            traceback.print_exc()
            print "response", response
            print "output",  output
        #print 'output is', result
        if result[0] == 'ok':
            return result[1]
        elif result[0] == 'echo':
            print result[1]
        else:
            print output
            assert False, result[1]
        
    def set_model(self,  *params, **kparams):
        return self.__call_server('set_model', self.soldir)
        
    def set_solfiles(self,  *params, **kparams):
        return self.__call_server('set_solfiles', *params, **kparams)
        
    def make_agents(self,  *params, **kparams):
        return self.__call_server('make_agents', *params, **kparams)        
        
    def load_solfiles(self,  *params, **kparams):
        return self.__call_server('load_solfiles', *params, **kparams)        

    def set_phys_path(self,  *params, **kparams):
        return self.__call_server('set_phys_path', *params, **kparams)        
        
    def validate_evaluator(self,  *params, **kparams):
        if self.p is None: return False
        return self.__call_server('validate_evaluator', *params, **kparams)        

    def eval(self,  *params, **kparams):
        return self.__call_server('eval', *params, **kparams)

    def terminate_all(self):
        return self.__call_server('terminate_all')
        
    

