'''

  * Utility for memory_report

  >>>  from petram.helper.memory_report import read_report
  >>>  mem_out, swap_out = read_report()
  or
  >>> mem_out, swap_out = read_report(file='')
  >>> figure();plot(mem_out['used'])

'''
import os 
from petram.mesh.nastran2mfem import NASReader, write_nas2mfem
from os.path import expanduser
import numpy as np
def str2number(s):
    s=s.replace('B', '')    
    s=s.replace('G', '*1e9')
    s=s.replace('M', '*1e6')
    s=s.replace('K', '*1e3')
    return eval(s)


def read_report(file = None, offset=None):
    if file is None:
        from ifigure.widgets.dialog import read
        path = read(message='Memory Report file to read', defaultdir=os.getcwd())
        if path == '': return
    else:
        path = expanduser(file)

    fid = open(path);lines=fid.readlines();fid.close()

    keys = lines[0].strip().split()
    
    mem_out = []
    swp_out = []

    for ll in lines[1:]:
        txts =ll.strip().split()
        if txts[0].startswith('Mem'):
            mem_out.append([str2number(s) for s in txts[1:]])
        if txts[0].startswith('Swap'):
            swp_out.append([str2number(s) for s in txts[1:]])
 
    mem_out = np.array(mem_out).transpose()
    swp_out = np.array(swp_out).transpose()
    
    mem_out = {keys[i]:mem_out[i] for i in range(len(mem_out))}
    swp_out = {keys[i]:swp_out[i] for i in range(len(swp_out))}

    return mem_out, swp_out 

