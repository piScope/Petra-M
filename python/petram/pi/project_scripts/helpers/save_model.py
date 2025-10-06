def save_model(path, meshfile_relativepath=False, do_preprocess = False):
    od = model.param.eval('mfem_model')
    if do_preprocess:
        model.scripts.helpers.rebuild_ns()        
        engine.run_preprocess(model.namespaces, model.datasets)
    od.save_to_file(path, meshfile_relativepath=meshfile_relativepath)
    model.variables.setvar('modelfile_path', path)

ans(save_model(*args, **kwargs))
