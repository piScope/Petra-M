
'''

  NumbaCoefficient

   utility to use NumbaCoefficient more easily

'''
from mfem.common.
class NumbaCoefficient():
    def __init__(self, coeff):
        try:
            self.real = coeff[0]
            self.imag = coeff[1]
        except:
            self.real = coeff
            self.imag = None

    def is_complex(self):
        return self.imag is not None

    def get_real_coefficient(self):
        return self.real

    def get_imag_coefficient(self):
        return self.imag





def GenerateSlaiceNumbaCoefficient(coeff):
    pass


def GenerateNumbaSumCoefficient(coeff):
    pass

def GenerateNumbaProductCoefficient(coeff):
    pass

def GenerateNumbaPowCoefficient(coeff):
    pass

def GenerateNumbaAdjCoefficient(coeff):
    pass

def GenerateNumbaInvCoefficient(coeff):
    pass
