import sys
import subprocess as sp
import os
from threading import Thread
import time


def make_new_sol(model):
    if not model.has_child('solutions'):
        model.add_folder('solutions')

    folder = model.solutions.add_folder('sol')
    folder.mk_owndir()
    param = model.param
    param.setvar('sol', '='+folder.get_full_path())
    return folder


def save_model(model, path, meshfile_relativepath=False):
    od = model.param.eval('mfem_model')
    od.save_to_file(path, meshfile_relativepath=meshfile_relativepath)
    model.variables.setvar('modelfile_path', path)


def run_xxx(model, path='', debug=0, thread=True, array_id=1, array_len=1,
            run_command=None):
    '''
    debug keyword will overwrite debug level setting 
    in model file
    '''
    import mfem

    if run_command is None:
        run_command = []

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
            folder = make_new_sol(model)
        else:
            folder = model.param.eval('sol')
            folder.clean_owndir()
        path = folder.owndir()

    fpath = os.path.join(path, 'model.pmfm')

    save_model(model, fpath)

    m = model.param.getvar('mfem_model')
    try:
        m.generate_script(dir=path,
                          petram_array_id=array_id,
                          petram_array_len=array_len)
    except:
        import traceback
        traceback.print_exc()
        return

    #import petram
    #from petram.helper.driver_path import parallel as driver
    opath = os.getcwd()

    #os.chdir(path)
    p = sp.Popen(run_command, stdout=sp.PIPE, stderr=sp.STDOUT, cwd=path)

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
        p.wait()
    else:
        stdoutdata, stderrdata = p.communicate()
        print(stdoutdata)

    p.kill()
    globals()['default_sol_path'] = os.path.dirname(path)

    from petram.sol.solsets import read_sol, find_solfiles
    path = model.param.eval('sol').owndir()
    try:
        solfiles = find_solfiles(path=path)
        model.variables.setvar('solfiles', solfiles)
    except:
        model.variables.delvar('solfiles')
    #if del_path: os.remove(path)

    os.chdir(opath)


def run_parallel(model, path='', nproc=1, debug=0, thread=True,
                 array_id=1, array_len=1):
    run_command = ['mpirun', '-n', str(nproc),
                   sys.executable, '-u',
                   'model.py', '-p', '-d',  str(debug)]
    run_xxx(model, path=path, debug=debug, thread=thread,
            array_id=array_id, array_len=array_len,
            run_command=run_command)
    globals()['default_glvis_args'] = ['-np', nproc]


def run_serial(model, path='', debug=0, thread=True, array_id=1,
               array_len=1):
    run_command = [sys.executable, '-u',
                   'model.py', '-s', '-d',  str(debug)]
    run_xxx(model, path=path, debug=debug, thread=thread,
            array_id=array_id, array_len=array_len,
            run_command=run_command)
    globals()['default_glvis_args'] = []
