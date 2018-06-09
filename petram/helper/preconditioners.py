
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


class _prc_decorator(object):
    def block(self, func):
        def dec(*args, **kwargs):
            obj = PreconditionerBlock(func)
            obj._params = (args, kwargs)
            return obj
        return dec
    def blk_diagonal(self, func):
        def dec(*args, **kwargs):
            obj = DiagonalPrcGen(func)
            return obj
        return dec
    def blk_lowertriangular(self, func):
        def dec(*args, **kwargs):
            obj = LowerTriangularPrcGen(func)
            return obj
        return dec
    def blk_generic(self, func):
        def dec(*args, **kwargs):
            obj = GenericPrcGen(func)
            return obj
        return dec

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

@prc.block
def mumps(guiname, **kwargs):
    # mumps("mumps1")
    from petram.solver.mumps_model import MUMPSPreconditioner
    prc = kwargs.pop('prc')
    blockname = kwargs.pop('blockname')
    
    r0 = prc.get_row_by_name(blockname)
    c0 = prc.get_col_by_name(blockname)
    A0 = prc.get_operator_block(r0, c0)

    invA0 = cls(A0, gui=prc.gui[guiprcname], engine=prc.engine)
    
@prc.block
def gmres(atol=1e-24, rtol=1e-12, max_num_iter=5,
          kdim=50, print_level=0, preconditioner=None, **kwargs):
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
    A0 = prc.get_operator_block(r0, c0)    

    gmres.SetOperator(A0)
    if preconditioner is not None:
        gmres.SetPreconditioner(preconditioner)
    return gmres

class PreconditionerBlock(object):
    def __init__(self, func):
        self.func = func

    def set_param(self, prc, blockname):
        self.prc = prc
        self.blockname = blockname

    def __call__(self):
        args, kwargs = self._params
        kwargs['prc'] = self.prc
        kwargs['blockname'] = self.blockname
        return self.func(*args, **kwargs)
     
class PrcBase(object):
    def set_param(self, opr, engine, gui):
        self._opr = weakref.ref(opr)
        self.gui = gui
        self._engine  = weakref.ref(engine)
        
    def copy_param(self, g):
        self.offset = g.offset
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
        try:
            return self.opr._linked_op[(r, c)]
        except KeyError:
            return None
        

class DiagonalPrcGen(PrcBase):
    def __init__(self, func=None, opr=None, engine=None, gui=None):
        self.func = func
        if gui is not None: self.set_param(opr, engine, gui)
        
    def __call__(self):
        offset = self.opr.RowOffsets()       
        D = mfem.BlockDiagonalPreconditioner(offset)
        if self.func is not None:
           self.func(D, self)
        return D

class LowerTriangluarPrcGen(PrcBase):
    def __init__(self, func=None, opr=None, engine=None, gui=None):   
        self.func = func
        if gui is not None: self.set_param(opr, engine, gui)
        
    def __call__(self):
        offset = self.opr.RowOffsets()              
        LT = mfem.BlockLowerTriangularPreconditioner(offset)
        if self.func is not None:        
           self.func(LT, self)
        return LT
        
class GenericPreconditioner(mfem.Solver):
    def __init__(self, offset, MultFunc, SetOperatorFunc):
        self.mult_func = MultFunc
        self.setoprator_func = SetOperatorFunc
     
    def Mult(self, *args):
        return self.mult_func(self, *args)
     
    def SetOperator(self, *args):
        return self.setoperator_func(self, *args)
   
class GenericPreconditionerGen(PrcBase):
    def __init__(self, func=None, opr=None, engine=None, gui=None): 
        self.func = func
        self.mult_func = None
        self.setoprator_func = None
        if gui is not None: self.set_param(opr, engine, gui)
        
    def Mult(func):
        self.mult_func = func

    def SetOperator(func):
        self.setoperator_func = func
        
    def __call__(self):
        offset = self.opr.RowOffsets()                     
        prc = GeneraticPreconditioner(offset,
                                      self.mult_func,
                                      self.seperator_func)
        self.func(prc, self)
        return prc
    
