def start_engine(mfemmodel=None):
    if mfemmodel is None: 
        mfemmodel = param.eval('mfem_model')
    if not 'engine' in globals():
        from petram.engine import SerialEngine
        engine = SerialEngine(model=mfemmodel)
        model.variables.setvar('engine', engine)
    else:
        engine = model.variables.getvar('engine')
    return engine
ans(start_engine())
