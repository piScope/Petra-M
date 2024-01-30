'''
 convert (small dense matrix) to chypremat/coo_matrix
 so that it can be used as an element of blockmatrix
'''
import numpy as np
from mfem.common.parcsr_extra import get_assumed_patitioning
from mfem.common.chypre import CHypreMat

# DO NOT IMPORT MPI in Global, sicne some routins will be used
# in serial mode too.
try:
    import mfem.par
    MFEM_PAR = True
except BaseException:
    MFEM_PAR = False


def Densemat2PyMat(mat):
    from scipy.sparse import coo_matrix, lil_matrix

    if MFEM_PAR:
        col_starts = get_assumed_patitioning(mat.shape[1])
        row_starts = get_assumed_patitioning(mat.shape[0])
        rows = row_starts[1] - row_starts[0]

        if np.iscomplexobj(mat):
            real = mat.real
            imag = mat.imag
        else:
            real = mat.astype(float, copy=False)
            imag = None
        # if real != 0.0:

        m1 = lil_matrix(real[row_starts[0]:row_starts[1], :])
        m1 = m1.tocsr()

        if imag is not None:
            m2 = lil_matrix(imag[row_starts[0]:row_starts[1], :])
            m2 = m2.tocsr()
        else:
            m2 = None
        return CHypreMat(m1, m2, col_starts=col_starts)
    else:
        m1 = coo_matrix(mat)
        return m1.tocsr()
