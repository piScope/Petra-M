model = proj.model1.mfem.param.getvar("mfem_model")
for m in model.walk():
    mname = m.__class__.__module__
    tmp = mname.split('.'); tmp[0] = 'petram'
    mname = '.'.join(tmp)
    mod = __import__(mname, globals(), locals(), [m.__class__.__name__])

    c = getattr(mod, m.__class__.__name__)
    #print m.__class__, c
    m.__class__ = c

#model.set_guiscript('.scripts.helpers.open_gui')
