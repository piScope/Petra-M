from petram.mfem_model import MFEM_ModelRoot
model.scripts.helpers.create_ns('global')
root = MFEM_ModelRoot()
model.param.setvar('mfem_model', root)
root['General'].new_ns('global')
#model.scripts.helpers.rebuild_ns()
import mfem.ser as mfem
