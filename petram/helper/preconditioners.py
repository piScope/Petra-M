
'''
   Copyright (c) 2018, S. Shiraiwa  
   All Rights reserved. See file COPYRIGHT for details.

   Precondirioners

   ### simple scenario,,,
   # In this mode, one choose preconditioner block from GUI
   # gui says.... A1 : ams(singular=True)

   # code in iterative_model
   g = DiagonalPrcGen(opr=opr, engin=engine, gui=gui)
   prc = g()
   ams.set_param(g, "A1")
   blk = ams(singular=True) # generate preconditioner
   prc.SetDiagonalBlock(kblockname, blk) # set to block


   ### scenario 2
   # in this mode, a user chooses operator type(diagonal, lowertriagular)
   # and block filling is done in decorated function
   # gui says.... D(*args, **kwargs)

   # code in iterative_model
   expr = self.gui.adv_prc  # expr: expression to create a generator. 
                            # (example) expr = "D('A1')"
   gen = eval(expr, self.gui._global_ns)
   gen.set_param(opr, engine, gui)
   M = gen()

   @prc.blk_diagonal (or @prc.blk_lowertriangular)
   def D(prc, g, *args, **kwargs):
       # first two argments are mandatory
       # prc: preconditioner such as mfem.BlockDiagonalPreconditioner
       # g  : preconditioner generator, which can be used to 
       #      access operator, gui, engine,,,,

       ams.set_param(g, "A1")
       blk = ams(singular=True) # generate preconditioner
       k = g.get_row_by_name("a")
       prc.SetDiagonalBlock(k, blk) # set to block
       return prc

   ### scenario 3
   # In this mode, a user has full control of preconditoner construction
   # Mult, SetOperator must defined
   # gui says.... S(*args, **kwargs)

   # code in iterative_model
   S.set_param(opr, engine, gui)
   prc = S()

   @prc.blk_generic
   def S(prc, g, *args, **kwargs):
       D.copy_param(g)
       prc1 = D()
       LT.copy_param(g)
       prc2 = LT()
       prc._prc1 = prc1
       prc._prc2 = prc2

   @S.Mult
   def S.Mult(prc, x, y):
       tmpy = mfem.Vector(); 
       prc._prc1.Mult(x, tmpy)
       prc._prc2.Mult(tmpy, y)

   @S.SetOperator
   def S.SetOperator(prc, opr):


'''
import weakref

from petram.mfem_config import use_parallel
if use_parallel:
   from petram.helper.mpi_recipes import *
   from mfem.common.parcsr_extra import *
   import mfem.par as mfem
   
   from mpi4py import MPI                               
   num_proc = MPI.COMM_WORLD.size
   myid     = MPI.COMM_WORLD.rank
   smyid = '{:0>6d}'.format(myid)
   from mfem.common.mpi_debug import nicePrint
else:
   import mfem.ser as mfem
   
class PreconditionerBlock(object):
    def __init__(self, func):
        self.func = func

    def set_param(self, prc, blockname):
        self.prc = prc
        self.blockname = blockname

    def __call__(self, *args, **kwargs):
        kwargs['prc'] = self.prc
        kwargs['blockname'] = self.blockname
        return self.func(*args, **kwargs)
     
class PrcCommon(object):
    def set_param(self, opr, engine, gui):
        self._opr = weakref.ref(opr)
        self.gui = gui
        self._engine  = weakref.ref(engine)
        
    def copy_param(self, g):
        self._opr = g._opr
        self.gui = g.gui
        self._engine  = g._engine
        
    @property
    def engine(self):
        return self._engine()
        
    @property
    def opr(self):
        return self._opr()
        
    def get_row_by_name(self, name):
        return self.engine.dep_var_offset(name)

    def get_col_by_name(self, name):
        return self.engine.r_dep_var_offset(name)

    def get_operator_block(self, r, c):
        # if linked_op exists (= op is set from python).
        # try to get it
        if hasattr(self.opr, "_linked_op"):
            try:
                return self.opr._linked_op[(r, c)]
            except KeyError:
                return None
        else:
            blk = self.opr.GetBlock(r, c)
            if use_parallel:
                return mfem.Opr2HypreParMat(blk)
            else:
                return mfem.Opr2SparseMat(blk)

    def get_diagoperator_by_name(self, name):
        r = self.get_row_by_name(name)
        c = self.get_row_by_name(name)
        return self.get_operator_block(r, c)
             
class PrcGenBase(PrcCommon):
    def __init__(self, func=None, opr=None, engine=None, gui=None):
        self.func = func
        self._params = (tuple(), dict())
        self.setoperator_func = None        
        if gui is not None: self.set_param(opr, engine, gui)
        
    def SetOperator(self, func):
        self.setoperator_func = func
         
         
class DiagonalPrcGen(PrcGenBase):
    def __call__(self, *args, **kwargs):
        offset = self.opr.RowOffsets()       
        D = mfem.BlockDiagonalPreconditioner(offset)
        if self.func is not None:
           self.func(D, self, *args, **kwargs)
        return D

class LowerTriangluarPrcGen(PrcGenBase):
    def __call__(self, *args, **kwargs):
        offset = self.opr.RowOffsets()              
        LT = mfem.BlockLowerTriangularPreconditioner(offset)
        if self.func is not None:        
           self.func(LT, self, *args, **kwargs)
        return LT
     
     
class GenericPreconditionerGen(PrcGenBase):
    def __init__(self, func=None, opr=None, engine=None, gui=None):
        PrcGenBase.__init__(self, func=func, opr=opr, engine=engine, gui=gui)
        self.mult_func = None
        self.setoperator_func = None

    def Mult(self, func):
        self.mult_func = func

    def __call__(self,  *args, **kwargs):
        assert self.mult_func is not None, "Mult is not defined"
        assert self.setoperator_func is not None, "SetOperator is not defined"        

        dargs, dkwargs = self._params
        assert len(dargs) == 0,  "Decorator allows only keyword argments"

        prc = GenericPreconditioner(self)
        
        for key in dkwargs:
           kwargs[key] = dkwargs[key]
        prc = self.func(prc,  *args, **kwargs)
        return prc

           
class _prc_decorator(object):
    def block(self, func):
        class deco(PreconditionerBlock):
            def __init__(self, func):
                self.func = func
        return deco(func)

        '''
        def dec(*args, **kwargs):
            obj = PreconditionerBlock(func)
            obj._params = (args, kwargs)
            return obj
        return dec
        '''
        
    def blk_diagonal(self, func):
        class deco(DiagonalPrcGen):
            def __init__(self, func):
                self.func = func
        return deco(func)

    def blk_lowertriangular(self, func):
        class deco(LowerTriangularPrcGen):
            def __init__(self, func):
                self.func = func
        return deco(func)                
     
    def blk_generic(self, *dargs, **dkargs):
        def wrapper(func):
            class deco(GenericPreconditionerGen):
                def __init__(self, func):
                    GenericPreconditionerGen.__init__(self, func)
                    self._params = (dargs, dkargs)
                    self.func = func
            return deco(func)
        return wrapper

prc = _prc_decorator()

#
#  prc.block
#
#    in prc block, following parameters are defined in kwargs
#       prc : block preconditioner generator
#       blockname : row name in prc

#       prc knows...
#       engine : petram.engin
#       gui    : iteretavie_solver_model object
#       opr    : operator to be smoothed

SparseSmootherCls = {"Jacobi": (mfem.DSmoother, 0),
                     "l1Jacobi": (mfem.DSmoother, 1),
                     "lumpedJacobi": (mfem.DSmoother, 2),
                     "GS": (mfem.GSSmoother, 0),
                     "forwardGS": (mfem.GSSmoother, 1),
                     "backwardGS": (mfem.GSSmoother, 2),}


def mfem_smoother(name, **kwargs):
    prc = kwargs.pop('prc')
    blockname = kwargs.pop('blockname')
    row = prc.get_row_by_name(blockname)
    mat = prc.get_operator_block(row, row)
    if use_parallel:
        smoother = mfem.HypreSmoother(mat)
        smoother.SetType(getattr(mfem.HypreSmoother, name))
    else:
        cls = SparseSmootherCls[name][0]
        arg = SparseSmootherCls[name][1]
        smoother = cls(mat, arg)
        smoother.iterative_mode = False
    return smoother
      
@prc.block
def GS(**kwargs):
    return mfem_smoother('GS', **kwargs)   
@prc.block
def l1GS(**kwargs):
    return mfem_smoother('l1GS', **kwargs)
@prc.block
def l1GStr(**kwargs):
    return mfem_smoother('l1GStr', **kwargs)
@prc.block
def forwardGS(**kwargs):
    return mfem_smoother('forwardGS', **kwargs)
@prc.block
def backwardGS(**kwargs):
    return mfem_smoother('backwardGS', **kwargs)
@prc.block
def Jacobi(**kwargs):
    return mfem_smoother('Jacobi', **kwargs)
@prc.block
def l1Jacobi(**kwargs):
    return mfem_smoother('l1Jacobi', **kwargs)
@prc.block
def lumpedJacobi(**kwargs):
    return mfem_smoother('lumpedJacobi', **kwargs)
@prc.block
def Chebyshev(**kwargs):
    return mfem_smoother('Chebyshev', **kwargs)
@prc.block
def Taubin(**kwargs):
    return mfem_smoother('Taubin', **kwargs)
@prc.block
def FIR(**kwargs):
    return mfem_smoother('FIR', **kwargs)
    
@prc.block
def ams(singular=True, **kwargs):
    prc = kwargs.pop('prc')
    blockname = kwargs.pop('blockname')
    
    print_level = kwargs.pop('print_level', -1)
    
    row = prc.get_row_by_name(blockname)
    mat = prc.get_operator_block(row, row)
    fes = prc.get_test_fespace(blockname)
    inv_ams = mfem.HypreAMS(mat, fes)
    if singular:
        inv_ams.SetSingularProblem()
    inv_ams.SetPrintLevel(print_level)
    return inv_ams

@prc.block
def schur(*names, **kwargs):
    # shucr("A1", "B1", scale=(1.0, 1e3))
    prc = kwargs.pop('prc')
    blockname = kwargs.pop('blockname')
    
    r0 = prc.get_row_by_name(blockname)
    c0 = prc.get_col_by_name(blockname)
    A0 = prc.get_operator_block(r0, c0)

    scales = kwargs.pop('scale', [1]*len(names))
    print_level = kwargs.pop('print_level', -1)
    for name, scale in zip(names, scales):
        r1 = prc.get_row_by_name(name)
        c1 = prc.get_col_by_name(name)
        B  = prc.get_operator_block(r0, c1)
        Bt = prc.get_operator_block(r1, c0)
        Bt  = Bt.Copy()
        B0 = get_block(A, r1, c1)
        if use_parallel:
             Md = mfem.HyprePaarVector(MPI.COMM_WORLD,
                                      B0.GetGlobalNumRows(),
                                      B0.GetColStarts())
        else:
            Md = mfem.Vector()
        A0.GetDiag(Md)
        Md *= scale
        if use_parallel:        
            Bt.InvScaleRows(Md)
            S = mfem.ParMult(B, Bt)
            invA0 = mfem.HypreBoomerAMG(S)
        else:
            S = mfem.Mult(B, Bt)
            invA0 = mfem.DSmoother(S)
        invA0.iterative_mode = False
        invA0.SetPrintLevel(print_level)
    return invA0
@prc.block
def mumps(guiname, **kwargs):
    # mumps("mumps1")
    from petram.solver.mumps_model import MUMPSPreconditioner
    prc = kwargs.pop('prc')
    blockname = kwargs.pop('blockname')
    
    r0 = prc.get_row_by_name(blockname)
    c0 = prc.get_col_by_name(blockname)
    A0 = prc.get_operator_block(r0, c0)

    invA0 =  MUMPSPreconditioner(A0, gui=prc.gui[guiname],
                                 engine=prc.engine)
    return invA0
    
@prc.block
def gmres(atol=1e-24, rtol=1e-12, max_num_iter=5,
          kdim=50, print_level=-1, preconditioner=None, **kwargs):
    prc = kwargs.pop('prc')
    blockname = kwargs.pop('blockname')

    if use_parallel:
        gmres = mfem.GMRESSolver(MPI.COMM_WORLD)
    else:
        gmres = mfem.GMRESSolver()
    gmres.iterative_mode = False
    gmres.SetRelTol(rtol)
    gmres.SetAbsTol(atol)
    gmres.SetMaxIter(max_num_iter)
    gmres.SetKDim(kdim)
    gmres.SetPrintLevel(print_level)    
    gmres.SetKDim(kdim)
    r0 = prc.get_row_by_name(blockname)
    c0 = prc.get_col_by_name(blockname)
    
    A0 = prc.get_operator_block(r0, c0)    

    gmres.SetOperator(A0)
    if preconditioner is not None:
        gmres.SetPreconditioner(preconditioner)
        # keep this object from being freed...
        gmres._prc = preconditioner
    return gmres


# these are here to use them in script w/o disnginguishing
# if mfem is mfem.par or mfem.ser
BlockDiagonalPreconditioner = mfem.BlockDiagonalPreconditioner
BlockLowerTriangularPreconditioner = mfem.BlockLowerTriangularPreconditioner

class GenericPreconditioner(mfem.Solver, PrcCommon):
    def __init__(self, gen):
        self.offset = gen.opr.RowOffsets()
        self.mult_func = gen.mult_func
        self.setoperator_func = gen.setoperator_func
        super(GenericPreconditioner, self).__init__()
        self.copy_param(gen)
        
    def Mult(self, *args):
        return self.mult_func(self, *args)
     
    def SetOperator(self, opr):
        opr = mfem.Opr2BlockOpr(opr)
        self._opr = weakref.ref(opr)
        self.offset = opr.RowOffsets()
        return self.setoperator_func(self, opr)

     

