from petram.phys.weakform import get_integrators
import numpy as np

import petram
import petram.phys
from os import listdir
from os.path import dirname, basename, isfile, join
import glob
import warnings
import glob
import sys

# collect all physics module

_modulenames = []
for p in petram.phys.__path__:
    mm = glob.glob(join(p, "*", "__init__.py"))
    for item in [basename(dirname(m)) for m in mm]:
        if item == 'coeff2d':  # we don't use this anymore
            continue
        if not item in _modulenames:
            _modulenames.append(item)


def all_phys_models():
    modulenames = []
    for m in _modulenames:
        if m == "common":  # this is the name for common physics subroutines
            continue
        try:
            mname = 'petram.phys.'+m + '.'+m+'_model'
            __import__(mname, locals(), globals())

            mm = getattr(petram.phys, m)
            mmm = getattr(mm, m+'_model')
            invalid = hasattr(mmm, "dependency_invalid")

            if invalid:
                import petram.mfem_model as mm
                if mm.has_addon_access == "any" or mm.has_addon_access == m:
                    modulenames.append(m)
            else:
                modulenames.append(m)
        except ImportError:
            warnings.warn('Failed to import physcis module :' + mname)

    models = []
    classes = []
    for m in modulenames:
        mm = getattr(petram.phys, m)
        models.append(getattr(mm, m+'_model'))

        chk = [x.isdigit() for x in m]
        if True in chk:
            chk = chk.index(True)
            if hasattr(models[-1], 'model_basename'):
                bs = getattr(models[-1], 'model_basename')
            else:
                bs = m[:chk+2].upper()
            classname = bs + m[chk+2:].lower()
            classes.append(getattr(models[-1], classname))
        else:
            bs = getattr(models[-1], 'model_basename')
            classes.append(getattr(models[-1], bs))
    return models, classes


def get_phys_constraints(module):
    mname = 'petram.phys.'+module
    if mname not in sys.modules:
        try:
            __import__(mname, locals(), globals())
        except ImportError:
            warnings.warn('Failed to import physcis module :' + mname)
            raise
    try:

        paths = getattr(sys.modules[mname], '__path__')
    except BaseException:
        warnings.warn('Can not find path for :' + mname)
        raise

    constraints = {'domain': [], 'bdry': [],
                   'edge': [], 'pointt': [], 'pair': []}

    for p in paths:
        for f in listdir(p):
            name = basename(f)
            if not name.endswith('py'):
                continue
            if name.startswith('_'):
                continue
            name = name[:-3]
            mname2 = mname + '.' + name

            try:
                __import__(mname2, locals(), globals())
            except BaseException:
                warnings.warn('Failed to import physcis module :' + mname2)
                raise

            for key in constraints:
                if hasattr(sys.modules[mname2], key+'_constraints'):
                    m = getattr(sys.modules[mname2], key+'_constraints')
                    for x in m():
                        if not x in constraints[key]:
                            constraints[key].append(x)

    return constraints


bilinintegs = get_integrators('BilinearOps')
linintegs = get_integrators('LinearOps')

itg_choices = bilinintegs + linintegs


def restrict_coeff(coeff, isDomain, sel_index, engine, emesh_idx,
                   vec=False, matrix=False):

    from petram.engine import mfem
    mesh = engine.emeshes[emesh_idx]

    if isDomain:
        arrsize = max(np.max(mesh.attributes.ToList()), engine.max_attr)
    else:
        arrsize = max(np.max(mesh.bdr_attributes.ToList()), engine.max_bdrattr)

    if isinstance(sel_index, str):
        sel_index = [sel_index]
    if len(sel_index) == 1 and sel_index[0] == 'all':
        return coeff

    arr = [0]*arrsize
    for k in sel_index:
        arr[k-1] = 1
    arr = mfem.intArray(arr)

    if vec:
        return mfem.VectorRestrictedCoefficient(coeff, arr)
    elif matrix:
        return mfem.MatrixRestrictedCoefficient(coeff, arr)
    else:
        return mfem.RestrictedCoefficient(coeff, arr)


def restricted_integrator(engine, integrator, sel_index,
                          c_txt, cotype,  codim,  emesh_idx,
                          isDomain, ind_vars, local_ns, global_ns, real):
    '''
    make a integrator with restricted coefficient
    (sample input)
    integrator : 'MassIntegrator'
    sel_index  : ['all'], [1,2,3]
    c_txt      : return value of self.vt_coeff.make_value_or_expression(self)[0]
    cotype     : 'S', 'V', 'M', 'D' (coefficient type)
    codim      :  3 (coefficient dim)
    ind_vars   : ('x', 'y', 'z')

    others are obvious....;D
    '''
    from petram.engine import mfem
    from petram.phys.weakform import VCoeff, SCoeff, MCoeff, DCoeff

    if cotype == 'S':
        for b in itg_choices:
            if b[0] == integrator:
                break
        if not "S*2" in b[3]:
            coeff = SCoeff(c_txt, ind_vars, local_ns, global_ns,
                           real=real)
        else:  # so far this is only for an elastic integrator
            coeff = (SCoeff(c_txt, ind_vars, local_ns, global_ns,
                            real=real, component=0),
                     SCoeff(c_txt, ind_vars, local_ns, global_ns,
                            real=real, component=1))
        coeff = restrict_coeff(coeff, isDomain, sel_index, engine, emesh_idx)
    elif cotype == 'V':
        coeff = VCoeff(codim, c_txt, ind_vars, local_ns, global_ns,
                       real=real)
        coeff = restrict_coeff(coeff, isDomain, coeff, sel_index, engine, emesh_idx,
                               vec=True)
    elif cotype == 'M':
        coeff = MCoeff(codim, c_txt, ind_vars, local_ns, global_ns,
                       real=real)
        coeff = restrict_coeff(coeff, isDomain, coeff, sel_index, engine, emesh_idx,
                               matrix=True)
    elif cotype == 'D':
        coeff = DCoeff(codim, c_txt, ind_vars, local_ns, global_ns,
                       real=real)
        coeff = restrict_coeff(coeff, isDomain, sel_index, engine, emesh_idx,
                               matrix=True)

    if coeff is None:
        return
    #if not isinstance(coeff, tuple): coeff = (coeff, )

    integrator = getattr(mfem, integrator)
    itg = integrator(coeff)
    itg._linked_coeff = coeff  # make sure that coeff is not GCed.

    return itg


def default_lf_integrator(info1, isDomain):
    if (info1['element'].startswith('ND') or
            info1['element'].startswith('RT')):
        if isDomain:
            integrator = 'VectorFEDomainLFIntegrator'
        else:
            if info1['element'].startswith('ND'):
                integrator = 'VectorFEBoundaryTangentLFIntegrator'
            else:
                integrator = 'VectorFEBoundaryFluxLFIntegrator'
    elif info1['element'].startswith('DG'):
        assert False, "auto selection is not supported for GD"
    elif info1['vdim'] > 1:
        if isDomain:
            integrator = 'VectorDomainLFIntegrator'
        else:
            integrator = 'VectorBoundaryLFIntegrator'
    elif not isDomain:
        integrator = 'BoundaryLFIntegrator'
    else:
        integrator = 'DomainLFIntegrator'
    return integrator


def default_bf_integrator(info1, info2, isDomain):
    if info1 == info2:
        if (info1['element'].startswith('ND') or
                info1['element'].startswith('RT')):
            integrator = 'VectorFEMassIntegrator'
        elif info1['element'].startswith('DG'):
            assert False, "auto selection is not supported for GD"
        elif info1['vdim'] > 1:
            integrator = 'VectorMassIntegrator'
        elif not isDomain:
            integrator = 'BoundaryMassIntegrator'
        else:
            integrator = 'MassIntegrator'
    else:
        if (((info1['element'].startswith('ND') or
              info1['element'].startswith('RT')) and
             (info2['element'].startswith('ND') or
              info2['element'].startswith('RT')))):
            integrator = 'MixedVectorMassIntegrator'
        elif info1['element'].startswith('DG'):
            assert False, "auto selection is not supported for GD"
        elif (info1['vdim'] == 1 and info1['vdim'] == info2['vdim']):
            integrator = 'MixedScalarMassIntegrator'
        else:
            assert False, "No proper integrator is found"
    return integrator
