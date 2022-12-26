
'''

  NumbaCoefficient

   utility to use NumbaCoefficient more easily

'''
from petram.mfem_config import use_parallel

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem


class NumbaCoefficient():
    def __init__(self, coeff):
        self.is_complex = coeff.IsOutComplex()
        self.mfem_numba_coeff = coeff

        if self.is_complex:
            self.real = coeff.real
            self.imag = coeff.imag
        else:
            self.real = coeff
            self.imag = None

    @property
    def complex(self):
        return self.is_complex

    def get_real_coefficient(self):
        return self.real

    def get_imag_coefficient(self):
        return self.imag

    @property
    def sdim(self):
        return self.mfem_numba_coeff.SpaceDimension()

    @property
    def ndim(self):
        self.mfem_numba_coeff.GetNdim()

    @property
    def shape(self):
        if self.ndim == 0:
            return tuple()
        elif self.ndim == 1:
            return (self.mfem_numba_coeff.GetVDim(),)
        elif self.ndim == 2:
            return (self.mfem_numba_coeff.GetWidth(),
                    self.mfem_numba_coeff.GetHeight(),)
        else:
            assert False, "unsupported dim"

    def is_matrix(self):
        return self.mfem_numba_coeff.GetNdim() == 2

    def is_vector(self):
        return self.mfem_numba_coeff.GetNdim() == 1

    def __add__(self, other):
        '''
        ruturn sum coefficient
        '''
        assert isinstance(other, NumbaCoefficient), "must be NumbaCoefficient"
        assert self.shape == other.shape, "ndim must match to perform sum operation"

        func = '\n'.join(['def f(ptx, coeff1, coeff2)',
                          '    return coeff1 + coeff2'])

        if self.GetNDim() == 0:
            coeff = mfem.jit.scalar(sdim=self.sdim,
                                    complex=self.complex,
                                    dependency=(self.mfem_numba_coeff,
                                                other.mfem_numba_coeff),
                                    debug=False)
        elif self.mfemGetNDim() == 1:
            coeff = mfem.jit.vector(sdim=self.sdim,
                                    complex=self.complex,
                                    dependency=(self.mfem_numba_coeff,
                                                other.mfem_numba_coeff),
                                    debug=False,
                                    shape=self.shape)

        elif self.GetNDim() == 2:
            coeff = mfem.jit.matrix(sdim=self.sdim,
                                    complex=self.complex,
                                    dependency=(self.mfem_numba_coeff,
                                                other.mfem_numba_coeff),
                                    debug=False,
                                    shape=self.shape)

        else:
            assert False, "unsupported dim"

        return NumbaCoefficient(coeff)

    def __sub__(self, other):
        if isinstance(other, Variable):
            return self() - other()
        return self() - other


def GenerateSlaiceNumbaCoefficient(coeff):
    pass


def GenerateNumbaSumCoefficient(coeff1, coeff2):
    if coeff1.real is None:
        pass
    if coeff1.imag is None:
        pass
    pass


def GenerateNumbaProductCoefficient(coeff):
    pass


def GenerateNumbaPowCoefficient(coeff):
    pass


def GenerateNumbaAdjCoefficient(coeff):
    pass


def GenerateNumbaInvCoefficient(coeff):
    pass
