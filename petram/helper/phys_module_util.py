import petram
import petram.phys
from os.path import dirname, basename, isfile, join
import glob
import warnings
import glob

# collect all physics module
modulenames = []
for p in petram.phys.__path__:
    mm = glob.glob(join(p, "*", "__init__.py"))
    modulenames.extend([basename(dirname(m)) for m in mm])
    
for m in modulenames:
    try:
        mname = 'petram.phys.'+m
        __import__(mname, locals(), globals())
    except ImportError:
        warnings.warn('Failed to import physcis module :' + mname)

def all_phys_modules():
    modules = [getattr(petram.phys, m) for m in modulenames]   
    return modulenames, modules

def all_phys_models():
    for m in modulenames:    
        try:
            mname = 'petram.phys.'+m + '.'+m+'_model'
            __import__(mname, locals(), globals())
        except ImportError:
            warnings.warn('Failed to import physcis module :' + mname)

    models = []
    classes = []
    for m in modulenames:
        mm = getattr(petram.phys, m)
        models.append(getattr(mm, m+'_model'))

        chk = ([x.isdigit() for x in m]).index(True)
        if hasattr(models[-1], 'model_basename'):
            bs = getattr(models[-1], 'model_basename')
        else:
            bs = m[:chk+2].upper()
        classname = bs + m[chk+2:].lower()
        classes.append(getattr(models[-1],classname))
    return models, classes



