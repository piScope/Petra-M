
import sys
model = proj.model1.mfem.param.eval("mfem_model")
for x in model.walk():
    mod = x.__class__.__module__
    cls = x.__class__.__name__
    newmod = '.'.join(['petram']+mod.split('.')[2:])
    mod = __import__(newmod)
    mod = sys.modules[newmod]
    newcls = getattr(mod, cls)
    x.__class__ = newcls
    
