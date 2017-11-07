def run_parallel(path='', nproc = 1, debug=0, thread=True):
    '''
    debug keyword will overwrite debug level setting 
    in model file
    '''
    import subprocess as sp
    import sys
    import mfem
    import os
    from threading  import Thread
    import time

    try:
        from Queue import Queue, Empty
    except ImportError:
        from queue import Queue, Empty  # python 3.x

    ON_POSIX = 'posix' in sys.builtin_module_names

    def enqueue_output(out, queue):
       for line in iter(out.readline, b''):
            queue.put(line)
       out.close()

    del_path = False
    if path == '': 
        if model.param.eval('sol') is None:
            folder = model.scripts.helpers.make_new_sol()
        else:
            folder = model.param.eval('sol')
            folder.clean_owndir()
        path = os.path.join(folder.owndir(), 'model.pmfm')
        model.scripts.helpers.save_model(path)
        del_path = True
    print path    
    import petram
    from petram.helper.driver_path import parallel as driver

    mfem_path = petram.__path__[0]
    args = ['mpirun', '-n', str(nproc), driver, str(path), 
             os.path.dirname(mfem_path), str(debug)]
    p = sp.Popen(args, stdout=sp.PIPE, stderr=sp.STDOUT)

    if thread:
        q = Queue()
        t = Thread(target=enqueue_output, args=(p.stdout, q))
        t.daemon = True # thread dies with the program
        t.start()
    
        while(True):
           try:  
               line = q.get_nowait() # or q.get(timeout=.1)
           except Empty:
               if not t.is_alive(): break
               time.sleep(1.0)
               pass #print('no output yet')
           else: 
               print(line.rstrip('\r\n'))
    else:
        stdoutdata, stderrdata = p.communicate()
        print stdoutdata
    globals()['default_sol_path'] = os.path.dirname(path)
    globals()['default_glvis_args'] = ['-np', nproc]

    from petram.sol.solsets import read_sol, find_solfiles
    path = model.param.eval('sol').owndir()
    try:
        solfiles = find_solfiles(path = path)
        model.variables.setvar('solfiles', solfiles)          
    except:
        model.variables.delvar('solfiles')
    #if del_path: os.remove(path)
ans(run_parallel(*args, **kwargs))
