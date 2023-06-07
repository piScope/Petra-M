def run_parallel(path='', nproc=1, debug=0, thread=True):
    '''
    debug keyword will overwrite debug level setting 
    in model file
    '''
    import subprocess as sp
    import sys
    import mfem
    import os
    from threading import Thread
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
        m = model.param.getvar('mfem_model')
        try:
            m.generate_script(dir=folder.owndir())
        except:
            import traceback
            traceback.print_exc()
            return
        del_path = True

    import petram
    #from petram.helper.driver_path import parallel as driver
    opath = os.getcwd()

    #args = ['mpirun', '-n', str(nproc), driver, str(path),  str(debug)]
    args = ['mpirun', '-n', str(nproc), sys.executable, '-u',
            'model.py', '-p', '-d',  str(debug)]
    os.chdir(folder.owndir())
    p = sp.Popen(args, stdout=sp.PIPE, stderr=sp.STDOUT)

    if thread:
        q = Queue()
        t = Thread(target=enqueue_output, args=(p.stdout, q))
        t.daemon = True  # thread dies with the program
        t.start()

        while(True):
            try:
                line = q.get_nowait()  # or q.get(timeout=.1)
            except Empty:
                if not t.is_alive():
                    break
                time.sleep(1.0)
                pass  # print('no output yet')
            else:
                if isinstance(line, bytes):
                    line = line.decode('utf-8')
                    line = '\n'.join(
                        [x for x in line.split('\n') if len(x) > 0])
                print(line)
    else:
        stdoutdata, stderrdata = p.communicate()
        print(stdoutdata)

    p.kill()
    globals()['default_sol_path'] = os.path.dirname(path)
    globals()['default_glvis_args'] = ['-np', nproc]

    from petram.sol.solsets import read_sol, find_solfiles
    path = model.param.eval('sol').owndir()
    try:
        solfiles = find_solfiles(path=path)
        model.variables.setvar('solfiles', solfiles)
    except:
        model.variables.delvar('solfiles')
    #if del_path: os.remove(path)

    os.chdir(opath)


ans(run_parallel(*args, **kwargs))
