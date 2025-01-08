from __future__ import print_function

import os
import sys
import time
import numpy as np
import weakref
import traceback
import subprocess as sp
import petram.helper.pickle_wrapper as pickle
import binascii
try:
   import bz2  as bzlib
except ImportError:
   import zlib as bzlib   
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

from threading import Timer, Thread
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty  # python 3.x
    
ON_POSIX = 'posix' in sys.builtin_module_names

wait_time = 0.3

def enqueue_output(p, queue, prompt):
    while True:
        line = p.stdout.readline()
        #print("line", line)
        
        if len(line) == 0:
            time.sleep(wait_time)
            continue

        if line ==  (prompt + '\n'): break
        queue.put(line)
        if p.poll() is not None:
           break #return
    queue.put("??????")
    
def enqueue_output2(p, queue, prompt):
    # this assumes recievein two data (size and data)
    while True:
        line = p.stdout.readline()
        #line = line.decode('utf-8')
        if len(line) == 0:
            time.sleep(wait_time)
            continue
        else:
            try:
                if line.startswith('z'):
                    use_zlib = True
                    size = int(line[1:])
                else:
                    use_zlib = False                
                    size = int(line)
                break
            except:
                ### Error string from C++ layer may show up here!?
                print("Unexpected text received", line)   
    line2 = p.stdout.read(size+1)
    line2 = binascii.a2b_hex(line2[:-1].encode())
    if use_zlib:
        line2 = bzlib.decompress(line2)
    queue.put(line2)
    while True:
        line = p.stdout.readline()
        #line = line.decode('utf-8')        
        if len(line) == 0:
            time.sleep(wait_time)
            continue
        else:
            break
    if line !=  prompt + '\n':
         assert False, "I don't get prompt!??: " + line
    queue.put("??????")

def run_and_wait_for_prompt(p, prompt, verbose=True, withsize=False):    
    q = Queue()
    if withsize:
        t = Thread(target=enqueue_output2, args=(p, q, prompt))
    else:
        t = Thread(target=enqueue_output, args=(p, q, prompt))
    t.daemon = True # thread dies with the program
    t.start()

    lines = []; lastline = ""
    alive = True
    while lastline != "??????":
        try:  line = q.get_nowait() # or q.get(timeout=.1)
        except Empty:
            time.sleep(wait_time)                
            #print('no output yet' + str(p.poll()))
        else: # got line
            lines.append(line)
            lastline = lines[-1]            
        if p.poll() is not None:
            alive = False
            print('proces terminated')
            break

    if verbose:
        print("Data recieved: " + str(lines))
    else:
        pass
        #print("Data length recieved: " + str([len(x) for x in lines]))
    return lines[:-1], alive

def run_with_timeout(timeout, default, f, *args, **kwargs):
    import thread   
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
        
def wait_for_prompt(p, prompt = '?', verbose = True, withsize=False):
    return run_and_wait_for_prompt(p, prompt,
                                   verbose=verbose,
                                   withsize=withsize)
        
def start_connection(host='localhost',
                     num_proc=2,
                     user='',
                     soldir='',
                     ssh_opts=None,
                     mp_debug=False):
   
    if user != '':
       user = user+'@'

    opts = ssh_opts if ssh_opts is not None else []
    
    #p= sp.Popen("ssh " + user + host + " 'printf $PetraM'", shell=True,
    #            stdout=sp.PIPE,
    #            universal_newlines = True)    
    #ans = p.stdout.readlines()[0].strip()
    command = "$PetraM/bin/launch_evalsvr.sh"
    if soldir != '':
        command = 'cd ' + soldir + ';' + command
    print(command)
    p = sp.Popen(['ssh'] + opts + ['-t', user + host, command],
                 stdin = sp.PIPE,
                 stdout=sp.PIPE, stderr=sp.STDOUT,
                 close_fds = ON_POSIX,
                 universal_newlines = True)

    data, alive = wait_for_prompt(p, prompt = 'num_proc?')
    if data[-1].startswith("protocol"):
       p.evalsvr_protocol = int(data[-1].split(':')[-1])
    else:
       p.evalsvr_protocol = 1

    txt = str(num_proc)+',' + str(mp_debug) + '\n'
    print("protcoal/debug flat",  txt)
    p.stdin.write(txt)
    p.stdin.flush()
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
    def __init__(self, nproc = 2, logfile = 'queue'):
        return EvaluatorMP.__init__(self, nproc = nproc,
                                    logfile = logfile)
    
    def set_model(self, soldir):
        import os
        soldir = os.path.expanduser(soldir)        
        model_path = os.path.join(soldir, 'model_proc.pmfm')
        if not os.path.exists(model_path):
           if 'case' in os.path.split(soldir)[-1]:
               model_path = os.path.join(os.path.dirname(soldir), 'model_proc.pmfm')

        # if model_proc.pmfm is not available, try to make it from
        # model.pmfm

        if not os.path.exists(model_path):
            model0_path = os.path.join(soldir, 'model.pmfm')
            if not os.path.exists(model0_path):
                if 'case' in os.path.split(soldir)[-1]:
                    model0_path = os.path.join(os.path.dirname(soldir), 'model.pmfm')
            if not os.path.exists(model0_path):           
                assert False, "Model File not found: " + model0_path
            model_path = model0_path
            
        self.tasks.put((3, model_path), join = True)
        
        #import tempfile, shutil
        #tmpdir = tempfile.mkdtemp()
        #model_path = os.path.join(tmpdir, 'model.pmfm')
        #self.tasks.put((3, model_path), join = True)
        #self.tasks.put((3, "model_proc.pmfm"), join = True)
        #shutil.rmtree(tmpdir)

    
class EvaluatorClient(Evaluator):
    def __init__(self,
                 nproc=2,
                 host='localhost',
                 soldir='',
                 user='',
                 ssh_opts=None,
                 mp_debug=False):
       
        self.init_done = False        
        self.soldir = soldir
        self.solfiles = None
        self.nproc = nproc
        self.p = start_connection(host=host,
                                  num_proc=nproc,
                                  user=user,
                                  soldir=soldir,
                                  ssh_opts=ssh_opts,
                                  mp_debug=mp_debug)
        self.failed = False

    def __del__(self):
        self.terminate_all()
        if self.p is not None:
           if self.p.poll() is None:
               self.p.terminate()
        self.p = None

    def __call_server0(self, name, *params, **kparams):
        if self.p is None: return
        verbose = kparams.pop("verbose", False)
        force_protocol1 = kparams.pop("force_protocol1", False)
        prompt = kparams.pop("prompt", "?")
        nowait = kparams.pop("nowait", False)
        command = [name, params, kparams]
        data = binascii.b2a_hex(pickle.dumps(command))
        print("Sending request", command)
        self.p.stdin.write(data.decode('utf-8') + '\n')
        self.p.stdin.flush()

        if nowait:
           return
        protocol = 1 if force_protocol1 else self.p.evalsvr_protocol

        import threading
        print("calling wait for prompt", threading.current_thread())
        output, alive = wait_for_prompt(self.p,
                                        prompt=prompt, 
                                        verbose = verbose,
                                        withsize = protocol > 1)
        if not alive:
           self.p = None
           return
        if protocol > 1:
            response = output[-1]
        else:
            response = binascii.a2b_hex(output[-1].strip())
        try:
            result = pickle.loads(response)
            if verbose:
                print("result", result)
        except:
            traceback.print_exc()
            print("response", response)
            print("output", output)
            assert False, "Unpickle failed"
        #print('output is', result)
        if result[0] == 'ok':
            return result[1]
        elif result[0] == 'echo':
            print(result[1])
        else:
            print(result)
            #assert False, result[1]
            message = ''.join(result[1])
            assert False, message

    def __call_server(self, name, *params, **kparams):        
        try:
            return self.__call_server0(name, *params, **kparams)
        except IOError:
            self.failed = True
            raise
        except:
            raise


    def set_model(self,  *params, **kparams):
        return self.__call_server('set_model', self.soldir)
        
    def set_solfiles(self,  *params, **kparams):
        #kparams["verbose"] = True        
        return self.__call_server('set_solfiles', *params, **kparams)
        
    def make_agents(self,  *params, **kparams):
        return self.__call_server('make_agents', *params, **kparams)        

    def make_probe_agents(self,  *params, **kparams):
        return self.__call_server('make_probe_agents', *params, **kparams)        

    def load_solfiles(self,  *params, **kparams):
        return self.__call_server('load_solfiles', *params, **kparams)        

    def set_phys_path(self,  *params, **kparams):
        return self.__call_server('set_phys_path', *params, **kparams)        
        
    def validate_evaluator(self,  *params, **kparams):
        if self.p is None:
           return False
        #kparams["verbose"] = True
        return self.__call_server('validate_evaluator', *params, **kparams)

    def eval(self,  *params, **kparams):
        return self.__call_server('eval', *params, **kparams)
     
    def eval_pointcloud(self,  *params, **kparams):
        return self.__call_server('eval_pointcloud', *params, **kparams)
     
    def eval_integral(self,  *params, **kparams):
        return self.__call_server('eval_integral', *params, **kparams)
    
    def eval_probe(self,  *params, **kparams):
        return self.__call_server('eval_probe', *params, **kparams)

    def terminate_all(self):
        try:
            ret = self.__call_server('terminate_all',
                                      prompt='byebye',
                                      force_protocol1=True)
            
        except BrokenPipeError:
            ### when server-side client is dead, terminate connection
            print("Broken Pipe Error, teminating the connection")
            self.p.terminate()
            self.p = None
            return 
         
        if self.p is not None:
           if self.p.poll() is None:
               self.p.terminate()
        return ret

    def terminate_allnow(self):
        try:
            ret =  self.__call_server('terminate_all',
                                      nowait=True)
            
        except BrokenPipeError:
            ### when server-side client is dead, terminate connection
            print("Broken Pipe Error, teminating the connection")
            self.p.terminate()
            self.p = None
            return 
         
        if self.p is not None:
           if self.p.poll() is None:
               self.p.terminate()
        return ret     
        
    

