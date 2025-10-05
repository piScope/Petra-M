'''
    Run('xxxx.nas'): generate xxxx.mesh
'''
import os 
from petram.mesh.nastran2mfem import NASReader, write_nas2mfem
from os.path import expanduser


def nas2mfem(file = None, offset=None):
    if file is None:
        from ifigure.widgets.dialog import read
        path = read(message='Select NAS file to read', wildcard='*.nas')
        if path == '': return
    else:
        path = expanduser(file)
    if offset is None: offset = [0.0, 0.0, 0.0]
    mname = os.path.basename(path)
    dirname = os.path.dirname(path)
    if '.' in mname: 
        mname = '.'.join(mname.split('.')[:-1])
    mname = mname + '.mesh'
    print('reading file '+ path)
    reader = NASReader(path)
    reader.load()
    write_nas2mfem(os.path.join(dirname, mname),  reader, offset=offset)

ans(nas2mfem(*args, **kwargs))
