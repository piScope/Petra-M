from __future__ import print_function

import numpy as np
import datetime
import os
import shlex
import socket
import base64
import traceback
import subprocess as sp


def communicate_with_timeout(p,
                             maxtimeout=np.inf,
                             timeout=2,
                             progdlg=None,
                             verbose=False):

    import wx
    if progdlg is not None:
        value = progdlg.GetValue()
    time = 0
    cancelled = False
    timeout_expired = False
    while True:
        #print("communicating", time)
        try:
            p.wait(timeout=timeout)
        except sp.TimeoutExpired:
            time = time + timeout
            wx.GetApp().Yield()
            if progdlg is not None:
                if progdlg:
                    flag, skipped = progdlg.Update(value)
                    cancelled = not flag
                else:
                    #print("widget deleted")
                    cancelled = True
                if cancelled:
                    p.kill()
                    return b"", b"", False, True
            if time > maxtimeout:
                timeout_expired = True
                p.kill()
                return b"", b"", True, False
            else:
                continue
        break
    outs, errs = p.communicate()
    if errs is not None:
        print(errs.decode())

    if verbose:
        if outs is not None:
            print(outs.decode())

    return outs, errs, timeout_expired, cancelled


def launch_ssh_command(model, command, verbose=True):
    param = model.param
    hosto = param.eval('host')
    host = hosto.getvar('server')
    user = hosto.getvar('user')
    opts = hosto.get_multiplex_opts()
    aopts = hosto.get_auth_opts(no_password=True)

    opts = ' '.join(opts)
    command = ("ssh -x " + opts + aopts + user +
               '@' + host + " '" + command + "'")

    if verbose:
        print("Executing on host (ssh): " + command)

    p = sp.Popen(command, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    return p


def wdir_from_datetime():
    import datetime
    import socket
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
        obj = c.get_child(name=objname)
        obj.setvar('server', host)
        obj.onSetting()
    else:
        obj = c.get_child(name=objname)
        obj.onSetting()
    return obj


def clean_remote_dir(model):
    param = model.param
    rwdir = param.eval('remtoe')['rwdir']
    if rwdir is None:
        return False

    host = param.eval('host')
    host.Execute('rm ' + rwdir + '/solmesh*')
    host.Execute('rm ' + rwdir + '/soli*')
    host.Execute('rm ' + rwdir + '/solr*')

    return True


def prepare_remote_dir(model, txt='', dirbase='', progdlg=None):
    model_dir = model.owndir()
    param = model.param
    if txt == '':
        txt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        hostname = socket.gethostname()
        rwdir = os.path.join(dirbase, txt+'_'+hostname)
    else:
        rwdir = os.path.join(dirbase, txt)

    host = param.eval('host')
    hostname = host.getvar('server')
    user = host.getvar('user')

    #p = host.Execute('mkdir -p ' + rwdir, nowait=True)
    command = 'mkdir -p ' + rwdir
    p = launch_ssh_command(model, command)
    #command = ("ssh -x -o PasswordAuthentication=no -o PreferredAuthentications=publickey " + user+'@' + host + " '" + command + "'")
    #p= sp.Popen(command, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)

    ret = communicate_with_timeout(p, timeout=2, progdlg=progdlg)
    _outs, _errs, timeout, cancelled = ret

    if timeout or cancelled:
        return True

    param.eval('remote')['rwdir'] = rwdir
    return False


def send_file(model, progdlg, skip_mesh=False, subdir=''):
    model_dir = model.owndir()
    param = model.param

    remote = param.eval('remote')
    host = param.eval('host')
    sol = param.eval('sol')
    sol_dir = sol.owndir()

    rwdir = remote['rwdir']
    if subdir != '':
        rwdir = rwdir+'/'+subdir
    mfem_model = param.eval('mfem_model')

    p = host.PutFile(os.path.join(sol_dir, 'model.pmfm'),
                     rwdir + '/model.pmfm',
                     nowait=True)
    ret = communicate_with_timeout(p, timeout=2, progdlg=progdlg)
    _outs, _errs, timeout, cancelled = ret
    if timeout or cancelled:
        return True

    if skip_mesh:
        return False
    for od in mfem_model.walk():
        if not od.is_enabled():
            continue
        if hasattr(od, 'use_relative_path'):
            path = od.get_real_path()
            dpath = rwdir+'/'+os.path.basename(od.path)
            p = host.PutFile(path, dpath, nowait=True)
            ret = communicate_with_timeout(p, timeout=2, progdlg=progdlg)
            _outs, _errs, timeout, cancelled = ret
            if timeout or cancelled:
                return True

    return False


def retrieve_files(model, rhs=False, matrix=False, sol_dir=None):
    model_dir = model.owndir()
    param = model.param
    if sol_dir is None:
        sol_dir = model.owndir()

    def get_files(host, key):
        xx = host.Execute('ls ' + os.path.join(rwdir, key+'*')
                          ).stdout.readlines()

        files = [x.strip() for x in xx if x.find(key) != -1]
        host.GetFiles(files, sol_dir)

        xx = host.Execute('ls -d ' + os.path.join(rwdir,
                                                  'case*')).stdout.readlines()
        print(xx)
        for x in xx:
            x0 = os.path.basename(x.strip())
            if not x0.startswith('case'):
                continue
            try:
                long(x0[4:])
            except:
                continue
            print('testing', os.path.join(sol_dir, x0))
            if not os.path.exists(os.path.join(sol_dir, x0)):
                os.mkdir(os.path.join(sol_dir, x0))
            yy = host.Execute('ls ' + os.path.join(rwdir, x0,
                                                   key+'*')).stdout.readlines()
            print('!!!!!!!!!!', yy)
            files = [os.path.join(x.strip(), os.path.basename(y.strip()))
                     for y in yy if y.find(key) != -1]
            host.GetFiles(files, os.path.join(sol_dir, x0))

    import os

    host = param.eval('host')
    remote = param.eval('remote')
    rwdir = remote['rwdir']

    get_files(host, 'solr')
    get_files(host, 'soli')
    get_files(host, 'solmesh')
    if matrix:
        get_files(host, 'matrix')
    if rhs:
        get_files(host, 'rhs')


def _get_job_queue(model, command,  host=None, user=None, progdlg=None, configext=''):
    '''
    param = model.param
    hosto = param.eval('host')
    host = hosto.getvar('server')
    user = hosto.getvar('user')
    '''
    # command = ("ssh -o PasswordAuthentication=no -o PreferredAuthentications=publickey " +
    #           user+'@' + host + " 'cat $PetraM/etc/queue_config'" )
    #p= sp.Popen(command, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)

    p = launch_ssh_command(model, command)
    ret = communicate_with_timeout(p, maxtimeout=30,
                                   timeout=2, progdlg=progdlg)
    outs, errs, timeout, cancelled = ret

    if timeout:
        return False, None
    if cancelled:
        return False, None

    lines = [x.strip() for x in outs.decode('utf-8').split('\n')]

    try:
        value = interpret_job_queue_file(lines)
    except BaseException:
        traceback.print_exc()
        return False, None
    return True, value


def get_job_queue(model, host=None, user=None, progdlg=None, configext=''):

    # try this first
    command = '$PetraM/bin/get_job_queues.sh'+configext
    ret = _get_job_queue(model, command,  host=host,
                         user=user, progdlg=progdlg, configext=configext)
    if ret[0]:
        return ret

    # try this (old method)
    command = 'cat $PetraM/etc/queue_config'+configext
    ret = _get_job_queue(model, command,  host=host,
                         user=user, progdlg=progdlg, configext=configext)
    return ret


def interpret_job_queue_file(lines):
    lines = [x.strip() for x in lines if not x.startswith("#")
             if len(x.strip()) != 0]

    if len(lines) == 0:
        return None

    q = {'type': lines[0], 'queues': [],
         'scratch': "~/myscratch", "notice": [], "versions": {}}
    for l in lines[1:]:
        if l.startswith('KEYWORD'):
            if not 'keywords' in q:
                q['keywords'] = []
            q['keywords'].append(l.split(':')[1])
        elif l.startswith('QUEUE'):
            q['queues'].append({'name': l.split(':')[1]})
        elif l.startswith('VERSION'):
            q['versions'][l.split(':')[1]] = []
        elif l.startswith('VEROPT'):
            version = l.split(':')[1]
            opt = l.split(':')[2]
            param = ':'.join(l.split(':')[3:])
            q['versions'][version].append((opt, param))
        elif l.startswith('SCRATCH'):
            q['scratch'] = l.split(':')[1].rstrip()
        elif l.startswith('NOTICE'):
            q['notice'].append((":".join(l.split(':')[1:])).rstrip())
        else:
            data = ':'.join(l.split(':')[1:])
            param = l.split(':')[0]
            q['queues'][-1][param] = data

    return q


def submit_job(model, progdlg=None, sh_command="$PetraM/bin/launch_petram.sh"):
    param = model.param
    host = param.eval('host')
    remote = param.eval('remote')
    rwdir = remote['rwdir']
    hostname = host.getvar('server')
    user = host.getvar('user')
    # p= sp.Popen("ssh " + user+'@' + hostname + " 'printf $PetraM'",
    #              shell=True, stdout=sp.PIPE)
    #PetraM = p.stdout.readlines()[0].decode('utf-8').strip()

    w = remote["walltime"]
    n = str(remote["num_cores"])
    N = str(remote["num_nodes"])
    o = str(remote["num_openmp"])
    q = str(remote["queue"])

    adv = str(remote["adv_opts"])
    adv = "\n".join([x.strip()
                    for x in adv.split("\n") if not x.strip().startswith("#")])
    env = str(remote["env_opts"])
    env = "\n".join([x.strip()
                    for x in env.split("\n") if not x.strip().startswith("#")])

    lk = []
    for k, v in remote["log_keywords"]:
        if v:
            lk.append(k.strip())
    lk = ','.join(lk)

    lt = str(remote["log_txt"])
    lt = "'".join(lt.split('"'))

    nt = str(remote["notification"])

    lk = base64.b64encode(lk.encode()).decode()
    lt = base64.b64encode(lt.encode()).decode()
    env = base64.b64encode(env.encode()).decode()
    adv = base64.b64encode(adv.encode()).decode()

    q1 = q.strip().split("(")[0]
    q2 = "" if q.find("(") == -1 else "(".join(q.strip().split("(")[1:])[:-1]
    # replace short/dev -> short_dev
    if q2 != "":
        q2 = "_".join(q2.split("/"))

    exe = (sh_command + ' -N ' + N + ' -P ' + n + ' -W ' + w + ' -O ' + o + ' -Q ' + q1
           + ' -L ' + lt + ' -K ' + lk + ' -M ' + nt)
    if q2 != "":
        exe = exe + ' -V ' + q2
    if adv != "":
        exe = exe + ' -A ' + adv
    if env != "":
        exe = exe + ' -E ' + env

    command = 'cd '+rwdir+';'+exe
    p = launch_ssh_command(model, command)
    ret = communicate_with_timeout(p, timeout=2, progdlg=progdlg, verbose=True)
    _outs, _errs, timeout, cancelled = ret

    if timeout or cancelled:
        return True
    return False
