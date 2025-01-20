#
#
#   PyBilinearform:
#     Additional bilinearform Integrator written in Python
#
#     Vector field operator:
#        PyVectorMassIntegrator :
#            (Mu_j, v_i)
#             M_ij is rank-2
#
#        PyVectorPartialIntegrator :
#            (Mdu_j/x_k, v_i)
#             M_ikj is rank-3
#        PyVectorWeakPartialIntegrator :
#            (Mu_j, -dv_i/dx_k)
#             M_ikj is rank-3. Note index order is the same as strong version
#
#        PyVectorDiffusionIntegrator :
#           (du_j/x_k M dv_i/dx_l)
#            M_likj is rank-4
#        PyVectorPartialPartialIntegrator :
#            (M du_j^2/x_kl^2, v_i)
#            M_iklj is rank-4
#        PyVectorWeakPartialPartialIntegrator : (
#            Mu_j, d^2v_i/dx^2)
#            M_iklj is rank-4. Note index order is the same as strong version
#
#   Copyright (c) 2024-, S. Shiraiwa (PPPL)
#
#

from petram.helper.integrators.pyvector_mass_integrator import PyVectorMassIntegrator
from petram.helper.integrators.pyvector_p_integrator import (PyVectorPartialIntegrator,
                                                             PyVectorWeakPartialIntegrator)
from petram.helper.integrators.pyvector_pp_integrator import (PyVectorDiffusionIntegrator,
                                                              PyVectorPartialPartialIntegrator,
                                                              PyVectorWeakPartialPartialIntegrator)

