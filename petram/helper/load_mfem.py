'''
  small utility to load mfem with use_parallel
'''
def load(use_parallel):
    if use_parallel:
        import mfem.par as mfem
        from mpi4py import MPI        
    else:
        import mfem.ser as mfem
        MPI= None

    import petram
    import petram.mfem_config
    petram.mfem_config.use_parallel = use_parallel
    
    return mfem, MPI

