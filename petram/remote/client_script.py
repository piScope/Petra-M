from __future__ import print_function

import numpy as np
import datetime
import os 
import shlex 
import socket
import base64
import subprocess as sp

base_remote_path = '~/myscratch/mfem_batch'

def wdir_from_datetime():
    import datetime, socket
    txt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    hostname = socket.gethostname()
    txt = txt + '_' + hostname
    return txt

def make_remote_connection(model, host):
    '''
    host = 'eofe7.mit.edu'

    '''
    import ifigure

    proj = model.get_root_parent()
    p = proj.setting.parameters
    
    if p.hasvar('connection'):
        c = p.eval('connection')
    else:
        c = None
    if c is None:
        base = os.path.dirname(ifigure.__file__)
        f = os.path.join(base, 'add_on', 'setting', 'module', 'connection.py')
        c = proj.setting.add_absmodule(f)
        p.setvar('connection', '='+c.get_full_path())

    objname = host.split('.')[0]
    if not c.has_child(objname):
        c.call_method('add_connection', objname)
        proj.app.proj_tree_viewer.update_widget()
        obj = c.get_child(name = objname)        
        obj.setvar('server', host)
        obj.onSetting()
    else:
        obj = c.get_child(name = objname)
        obj.onSetting()
    return obj

def clean_remote_dir(model):
    param = model.param
    rwdir = param.eval('remtoe')['rwdir']
    if rwdir is None: return False

    host = param.eval('host')
    host.Execute('rm ' + rwdir + '/solmesh*')
    host.Execute('rm ' + rwdir + '/soli*')
    host.Execute('rm ' + rwdir + '/solr*')

    return True

def prepare_remote_dir(model, txt = '', dirbase = base_remote_path):
    model_dir = model.owndir()
    param = model.param
    if txt  == '':
        txt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        hostname = socket.gethostname()
        rwdir = os.path.join(dirbase,txt+'_'+hostname)
    else:
        rwdir = os.path.join(dirbase, txt)
 
    try:
        host = param.eval('host')
        host.Execute('mkdir -p ' + rwdir)
    except:
        assert  False, "Failed to make remote directory"
    param.eval('remote')['rwdir'] = rwdir

def send_file(model, skip_mesh = False):
    model_dir = model.owndir()
    param = model.param

    remote = param.eval('remote')
    host = param.eval('host')
    sol = param.eval('sol')
    sol_dir = sol.owndir()

    rwdir = remote['rwdir']
    mfem_model = param.eval('mfem_model')

    host.PutFile(os.path.join(sol_dir, 'model.pmfm'), rwdir + '/model.pmfm')

    if skip_mesh: return
    for od in mfem_model.walk():
        if not od.is_enabled(): continue
        if hasattr(od, 'use_relative_path'):
            path = od.get_real_path()
            dpath = rwdir+'/'+os.path.basename(od.path)
            host.PutFile(path, dpath)

def retrieve_files(model, rhs=False, matrix = False, sol_dir = None):
    model_dir = model.owndir()
    param = model.param
    if sol_dir is None:
        sol_dir = model.owndir()

    def get_files(host, key):
        xx = host.Execute('ls ' + os.path.join(rwdir, key+'*')).stdout.readlines()

        files = [x.strip() for x in xx if x.find(key) != -1]
        host.GetFiles(files, sol_dir)
        
        xx = host.Execute('ls -d ' + os.path.join(rwdir, 'case*')).stdout.readlines()
        print(xx)
        for x in xx:
            x0 = os.path.basename(x.strip())
            if not x0.startswith('case'): continue
            try: 
                long(x0[4:])
            except:
                continue
            print('testing', os.path.join(sol_dir, x0))
            if not os.path.exists(os.path.join(sol_dir, x0)): 
                os.mkdir(os.path.join(sol_dir, x0))
            yy = host.Execute('ls ' + os.path.join(rwdir, x0, key+'*')).stdout.readlines()
            print('!!!!!!!!!!', yy)
            files = [os.path.join(x.strip(), os.path.basename(y.strip())) for y in yy if y.find(key) != -1]
            host.GetFiles(files, os.path.join(sol_dir, x0))
            
    import os

    host = param.eval('host')
    remote = param.eval('remote')
    rwdir = remote['rwdir']

    get_files(host, 'solr')
    get_files(host, 'soli')
    get_files(host, 'solmesh')
    if matrix: get_files(host, 'matrix')
    if rhs: get_files(host, 'rhs')
 
def get_job_queue(model=None, host = None, user = None):
    if model is not None:
        param = model.param
        hosto = param.eval('host')
        host = hosto.getvar('server')
        user = hosto.getvar('user')
    #command = "ssh " + user+'@' + host + " 'printf $PetraM'"
    #p = sp.Popen(command, shell=True,  stdout=sp.PIPE)
    #lines = p.stdout.readlines()
    #PetraM = lines[-1].decode('utf-8').strip()

    command = "ssh " + user+'@' + host + " 'cat $PetraM/etc/queue_config'", 
    p= sp.Popen(command, shell=True, stdout=sp.PIPE)
    lines = [x.decode('utf-8') for x in p.stdout.readlines()]

    try:
        value = interpret_job_queue_file(lines)
    except BaseException:
        assert False, "Failed to load server queue config"
    return value

def interpret_job_queue_file(lines):
    lines = [x.strip() for x in lines if not x.startswith("#")
             if len(x.strip()) != 0]
    q = {'type': lines[0], 'queues':[]}
    for l in lines[1:]:
        if l.startswith('KEYWORD'):
            if not 'keywords' in q:
                q['keywords'] = []
            q['keywords'].append(l.split(':')[1])
        elif l.startswith('QUEUE'):
            q['queues'].append({'name':l.split(':')[1]})
        else:
            data = ':'.join(l.split(':')[1:])
            param = l.split(':')[0]
            q['queues'][-1][param] = data
    return q
    
def submit_job(model):
    param = model.param
    host = param.eval('host')
    remote = param.eval('remote') 
    rwdir = remote['rwdir']

    hostname = host.getvar('server')
    user = host.getvar('user')
    #p= sp.Popen("ssh " + user+'@' + hostname + " 'printf $PetraM'",
    #              shell=True, stdout=sp.PIPE)
    #PetraM = p.stdout.readlines()[0].decode('utf-8').strip()

    w = remote["walltime"]
    n = str(remote["num_cores"])
    N = str(remote["num_nodes"])
    o = str(remote["num_openmp"])
    q = str(remote["queue"])

    lk = []
    for k, v in remote["log_keywords"]:
       if v: lk.append(k.strip())
    lk = ','.join(lk)
    
    lt = str(remote["log_txt"])
    lt = "'".join(lt.split('"'))
    
    nt = str(remote["notification"])

    lk = base64.b64encode(lk.encode()).decode()    
    lt = base64.b64encode(lt.encode()).decode()

    q1 = q.strip().split("(")[0]
    q2 = "" if q.find("(") == -1 else "(".join(q.strip().split("(")[1:])[:-1]
    ## replace short/dev -> short_dev
    if q2 != "":
        q2 = "_".join(q2.split("/"))
    
    exe = ('$PetraM/bin/launch_petram.sh -N '+N + ' -P ' + n + ' -W ' + w +' -O ' + o + ' -Q ' + q1
           + ' -L ' + lt + ' -K ' + lk + ' -M ' + nt)
    if q2 != "":
       exe = exe +  ' -V ' + q2        

    # we use force_ssh so that submission script is not influcence
    # by the current enviroment. (it matters when client and server
    # is runningn on the same machine)
    p = host.Execute('cd '+rwdir+';'+exe, force_ssh=True)
    if p.stdout is not None:
         print(''.join(p.stdout.readlines()))



 


