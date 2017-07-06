import numpy as np
import scipy
import scipy.sparse.linalg
from  petram.solver.solver_utils import *

def mumps_solve(A, b, c = None, m = None, Acbc=False, use_null=False):
    '''
    solve A*x= b using mumps through PETSc

    if c(constraints) is given, it eliminates "dependent"
    DoF to reduce the size of A

    if c is given and m is None, m is assume to zero.
        [A    c^t] [x]    [b]
        [        ] [ ]  = [ ]
        [c     0 ] [l]    [m]

    Note: c can be narrower than A. if so zero is padded on the right
          side of c

    Approach (as of 2016/03/30 this does not work well since it makes
              Ac much more dense. Maybe sparse null space solver
              will make this work...)

       Then, 
            X = Xd + N Xn
       
       ,where N is the null space of c as
            c * N = 0

        Ud and Un is given by 
          c * Xd = m 
          Ac * Xn = bc
       , where Ac and bc are
          Ac = N^T K N
          bc = N^T (b - A*Ud)


    if m is not None:
        c*Xd = m
    needs to be answered, which is underdetermined. So I instead 
    minimize Xd*Xd = 0 under c*Xd = m. This leads to another similar
    linear system.
          [2     c^t][x]   [0]
          [         ][ ] = [ ]
          [c      0 ][l]   [m]
    This can be solved easily since [1,1] element is diagonal.

    '''
    from scipy.sparse import csr_matrix, bmat, hstack
    if c is not None and use_null:
        for cc, mm in zip(c, m):        
           print('handling constraints')
           asize = A.shape[1]
           csize = c.shape[1]
           delta = asize - csize
           if delta > 0:
               c = hstack((c, csr_matrix((c.shape[0], delta), dtype = A.dtype)), format = 'csr')
               s1, rr, N = null(c.toarray())  # can I use sparse version of this..!?
               N = csr_matrix(N)
           if m is not None:
                ll = mumps_solve(c.transpose(c), -2*m)
                Xd = -c.transpose().dot(ll)/2
                bc = N.transpose().dot(b - A.dot(Xd))
           else:
                bc = N.transpose().dot(b)
           Ac = N.transpose().dot(A.dot(N))            

           A = Ac; b = bc
    elif c is not None:
        if m is None: m = [None]*len(c)
        for cc, mm in zip(c, m):
            A, b = add_constraints(A, b, cc, m=mm)
    else:
        pass
    if Acbc: return Ac, bc
                                   
    from petsc4py import PETSc
    D = A.data
    I = A.indptr
    J = A.indices
    #LogSetup      = PETSc.Log.Stage("Setup")
    #LogSetup.push()
    A = PETSc.Mat().createAIJ(size=A.shape[0], csr=(I, J, D))

    opts = PETSc.Options()
    output = False
    if output:
       opts["mat_mumps_icntl_2"] = 6
       opts["mat_mumps_icntl_3"] = 6
       opts["mat_mumps_icntl_4"] = 6
    opts["mat_mumps_icntl_14"] = 50  # percentage increase in the estimated
                                     # working space
    #opts["mat_mumps_icntl_10"] = -3  # iterative refinement to improve the
    #                                 # computed solution
    opts["mat_mumps_cntl_3"] = 1e-12
    opts.setFromOptions()


    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
#    ksp.setUp()
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.setType('preonly')
    pc = ksp.getPC()
    pc.setFromOptions()
    pc.setType('lu')    
    pc.setFactorSolverPackage('mumps')
    #LogSetup.pop()
    x2, b2 = A.getVecs() # this b2 is not used
    b2 = PETSc.Vec().createWithArray(b)
    ksp.solve(b2,x2)
    sol = x2.getArray()
                                   
    if c is not None and use_null:
        if m is not None:
            sol = Xd + N.dot(sol)                        
        else:
            sol = N.dot(sol)                        
    return sol

