"""
Resolve MPI compiler wrappers selected by build settings, environment, or PATH.
"""
from shutil import which


def configure_mpi(bglb):
    """
    Locate the configured C/C++ wrappers without loading an MPI runtime.
    """
    for name in ('mpicc', 'mpicxx'):
        command = getattr(bglb, name)
        wrapper = which(command) if command else None
        if wrapper is None:
            raise RuntimeError(
                'Cannot find MPI compiler wrapper ' + name.upper() +
                ': ' + str(command) + '. Install MPI development tools or set ' +
                name.upper() + ' to an executable wrapper. To omit the native ' +
                'component, use -Cwith-superlu_dist=No.')
        setattr(bglb, name, wrapper)
        print('MPI compiler ' + name.upper() + ': ' + wrapper, flush=True)
