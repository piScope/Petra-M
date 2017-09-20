model.param.setvar('nproc', 2)
model.add_folder('namespaces')
model.add_folder('datasets')
model.add_folder('solutions')
model.scripts.helpers.reset_model()
model.set_guiscript('.scripts.helpers.open_gui')
model.scripts.helpers.create_ns('global')

param = model.param
param.setvar('mfem', None)
param.setvar('sol', None)
param.setvar('remote', None)
