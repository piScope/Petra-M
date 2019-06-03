def call_glvis(solpath = None,
               np  = -1,
               thread = True):

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
    od = os.getcwd()
    if solpath is None: 
       solpath = model.param.eval('sol').owndir()
    os.chdir(solpath)
    args = ['glvis']
    if np < 0: 
        files = os.listdir(solpath)
        mfiles = [x for x in files if x.startswith('solmesh')]
        if mfiles[0] != 'solmesh': ## if not serial run result
#        args.extend([str(x) for x in globals()['default_glvis_args']])
            args.extend(['-np', str(len(mfiles))])
    else:
        args.extend(['-np', str(np)])
    args.extend(['-m', 'solmesh',  '-g',  'solr', '-gc', '2'])
    print(args)
    p = sp.Popen(args, stdout=sp.PIPE, stderr=sp.STDOUT)
    os.chdir(od)
    if thread:
        q = Queue()
        t = Thread(target=enqueue_output, args=(p.stdout, q))
        t.daemon = True # thread dies with the program
        t.start()
        while(t.is_alive()):
           try:
               line = q.get_nowait() # or q.get(timeout=.1)
           except Empty:
               time.sleep(1.0)
               pass #print('no output yet')
           else: 
               print(line.rstrip('\r\n'))
    else:
        stdoutdata, stderrdata = p.communicate()
        print(stdoutdata)
  
ans(call_glvis(*args, **kwargs))
