'''
    mesh_inspect

    tools to inspect mesh elements
'''
import numpy as np
import textwrap
from collections import defaultdict
from ifigure.interactive import figure

from petram.mfem_config import use_parallel

if use_parallel:
   import mfem.par as mfem
   from mfem.par import intp   
else:
   import mfem.ser as mfem
   from mfem.ser import intp

'''
Data collectors
'''   
def get_face_data(mesh, faces, refine=3):
    from petram.mesh.refined_mfem_geom import get_geom
    
    bases = defaultdict(list)
    
    for f in faces:
        b = mesh.GetFaceBaseGeometry(f)
        bases[b].append(f)
    bases = dict(bases)

    data ={}

    for key in bases:
        idx = bases[key]
        gt = mesh.GetFaceTransformation
        base = mesh.GetFaceBaseGeometry(idx[0])
        attrs = {x: x for x in idx}
        _attrs, ptx_x, ivert_x, iverte_x, attrs_x = get_geom(idx,
                                                            3,
                                                            base,
                                                            gt,
                                                            attrs,
                                                            3,
                                                            refine=refine)
        data[key] = (ptx_x, ivert_x, iverte_x, attrs_x)
    return data

def get_element_data(mesh, element, refine=3):
    faces, _void = mesh.GetElementFaces(element)
    data = get_face_data(mesh, faces, refine=refine)
    return data

def get_faces_containing_elements_data(mesh, faces, refine=5, win=None):
    all_faces1 = []
    all_faces2 = []    
    for face in faces:
        ts = mesh.GetFaceElementTransformations(face)
        el1 = ts.Elem1No
        el2 = ts.Elem2No
        f2, _void = mesh.GetElementFaces(el1)
        all_faces1.extend(f2)
        if el2 >= 0:
            f2, _void = mesh.GetElementFaces(el2)
            all_faces2.extend(f2)
            
    data1 = get_face_data(mesh, all_faces1, refine=refine)
    data2 = get_face_data(mesh, all_faces2, refine=refine)    
    
    return data1, data2
'''
Plot
'''   
def plot_faces(mesh, faces, refine=3, win=None, fc='b'):
    if win is None:
        win = figure()

    data = get_face_data(mesh, faces, refine=3)
    for k in data:
        ptx_x, ivert_x, iverte_x, attrs_x = data[k]
        win.solid(ptx_x, ivert_x, facecolor=fc, array_idx=attrs_x)
        win.solid(ptx_x, iverte_x, facecolor='k')
    return win

def plot_faces_containing_elements(mesh, faces, refine=5, fcs=None, win=None):

    if fcs is None:
        fcs = 'bg'
        
    if win is None:
        win = figure()
    
    data1, data2  = get_faces_containing_elements_data(mesh, faces, refine=refine)
    for k in data1:
        ptx_x, ivert_x, iverte_x, attrs_x = data1[k]
        win.solid(ptx_x, ivert_x, facecolor=fcs[0], array_idx=attrs_x)
        win.solid(ptx_x, iverte_x, facecolor='k')
    for k in data2:
        ptx_x, ivert_x, iverte_x, attrs_x = data2[k]
        win.solid(ptx_x, ivert_x, facecolor=fcs[1], array_idx=attrs_x)
        win.solid(ptx_x, iverte_x, facecolor='k')
        
    return win

def plot_element(mesh, element, refine=3, win=None, fc='b'):
    faces, _void = mesh.GetElementFaces(element)
    win = plot_face(mesh, faces, refine=refine, win=win, fc=fc)
    return win

'''
Invalid topology check
'''
def find_invalid_topology(mesh):
    '''
    We use the topology check in 
    https://github.com/mfem/mfem/blob/master/mesh/mesh.cpp#L2620

    We also returns the attributes of the elements containing invalid
    faces
    '''
    dim = mesh.Dimension()
    sdim = mesh.SpaceDimension()
    attrs = mesh.GetAttributeArray()

    if not (dim > 2 and sdim==dim):
        # this case does not need this check
        return [], []

    NFaces = mesh.GetNumFaces()

    invalid = []
    invalid_attrs = []
    elem1inf = intp()
    elem2inf = intp()
    
    for i in range(NFaces):
        info = mesh.GetFaceElementTransformations(i)
        mesh.GetFaceInfos(i, elem1inf, elem2inf)

        check = info.Elem2No < 0 or (elem2inf.value() % 2) != 0
        if not check:
            invalid.append(i)
            tmp = (attrs[info.Elem1No], attrs[info.Elem2No])
            invalid_attrs.append(tmp)
    return invalid, invalid_attrs


def check_invalid_topology(mesh, verbose=True, do_assert=False):
    invalid = find_invalid_topology(mesh)

    if verbose and len(invalid) > 0:
        print(str(len(invalid)) + " of invalid topology is found")
        
    if do_assert and len(invalid) > 0:
        assert False, str(len(invalid)) + " of invalid topology is found"

    return (len(invalid)==0, invalid)

def format_error(invalids, invalid_attrs, width=60):
    out1 = textwrap.wrap(', '.join([str(x) for x in invalids]),
                         width=width)
    unique_attrs = list(np.unique(np.array(invalid_attrs).flatten()))
    out2 = textwrap.wrap(', '.join([str(int(x)) for x in unique_attrs]),
                         width=width)
    out1 = ['    ' + x for x in out1]
    out2 = ['    ' + x for x in out2]    
    return '\n'.join([str(len(invalids))+" invalid Faces:"] + out1 + ["Found in domain:"] + out2)
    
