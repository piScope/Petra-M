
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
    myid = MPI.COMM_WORLD.rank
    smyid = '{:0>6d}'.format(myid)
    from mfem.common.mpi_debug import nicePrint
else:
    import mfem.ser as mfem

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('Preconditioner')


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
    def set_param(self, opr, name, engine, gui):
        self._opr = weakref.ref(opr)
        self.gui = gui
        self.name = name  # variable name on opr
        self._engine = weakref.ref(engine)

    def copy_param(self, g):
        self._opr = g._opr
        self.gui = g.gui
        self._engine = g._engine

    @property
    def engine(self):
        return self._engine()

    @property
    def opr(self):
        return self._opr()

    def get_row_by_name(self, name):
        return self.name.index(name)

    def get_col_by_name(self, name):
        return self.name.index(name)

    def get_operator_block(self, r, c):
        # if linked_op exists (= op is set from python).
        # try to get it
        # print(self.opr._linked_op)
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

    def get_test_fespace(self, name):
        fes = self.engine.fespaces[name]

        return fes


class PrcGenBase(PrcCommon):
    def __init__(self, func=None, opr=None, engine=None, gui=None, name=None):
        self.func = func
        self._params = (tuple(), dict())
        self.setoperator_func = None
        if gui is not None:
            self.set_param(opr, name, engine, gui)

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
                     "backwardGS": (mfem.GSSmoother, 2), }


def _create_smoother(name, mat):
    if use_parallel:
        smoother = mfem.HypreSmoother(mat)
        smoother.SetType(getattr(mfem.HypreSmoother, name))
    else:
        cls = SparseSmootherCls[name][0]
        arg = SparseSmootherCls[name][1]
        smoother = cls(mat, arg)
        smoother.iterative_mode = False
    return smoother


def mfem_smoother(name, **kwargs):
    prc = kwargs.pop('prc')
    blockname = kwargs.pop('blockname')
    row = prc.get_row_by_name(blockname)
    col = prc.get_col_by_name(blockname)
    mat = prc.get_operator_block(row, col)

    if isinstance(mat, mfem.ComplexOperator):
        conv = mat.GetConvention()

        blockOffsets = mfem.intArray()
        blockOffsets.SetSize(3)
        blockOffsets[0] = 0
        blockOffsets[1] = mat.Height()//2
        blockOffsets[2] = mat.Height()//2
        blockOffsets.PartialSum()

        smoother = mfem.BlockDiagonalPreconditioner(blockOffsets)

        m_r = mat._real_operator
        #m_i = mat._imag_operator
        pc_r = _create_smoother(name, m_r)
        pc_i = mfem.ScaledOperator(pc_r,
                                   1 if conv == mfem.ComplexOperator.HERMITIAN else -1)

        smoother.SetDiagonalBlock(0, pc_r)
        smoother.SetDiagonalBlock(1, pc_i)
        smoother._smoothers = (pc_r, pc_i)

    else:
        smoother = _create_smoother(name, mat)
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
def schwarz(**kwargs):
    assert use_parallel, "Schwarz smoother supports only parallel mode"

    prc = kwargs.pop('prc')
    blockname = kwargs.pop('blockname')

    fes = prc.get_test_fespace(blockname)
    fes_info = prc.engine.fespaces.get_fes_info(fes)

    use_basemesh = kwargs.pop('basemesh', -1)
    if use_basemesh == -1:
        target = kwargs.pop('ref', fes_info['refine'])
        ref_level = fes_info['refine'] - target
        fes_info['refine'] = target
        fes0 = prc.engine.fespaces.get_fes_from_info(fes_info)
        pmesh = fes0.GetParMesh()
    else:
        pmesh = prc.engine.base_meshes[fes_info['emesh_idx']]
        ref_level = use_basemesh

    #dprint1(pmesh, pmesh.GetNE(), ref_level)

    row = prc.get_row_by_name(blockname)
    col = prc.get_col_by_name(blockname)
    mat = prc.get_operator_block(row, col)

    iter = kwargs.pop('iter', 1)
    theta = kwargs.pop('theta', 1)

    if isinstance(mat, mfem.ComplexOperator):
        conv = mat.GetConvention()

        m_r = mat._real_operator    # HypreParMatrix
        m_i = mat._imag_operator    # HypreParMatrix

        s = fes.GlobalTrueVSize()
        AZ = mfem.ComplexHypreParMatrix(m_r, m_i, False, False, conv)
        #AZ = mfem.ComplexHypreParMatrix(m_r, None, False, False, conv)
        M = mfem.ComplexSchwarzSmoother(pmesh, ref_level, fes, AZ)
        M.SetDumpingParam(theta)
        M.SetNumSmoothSteps(iter)
        M._linked_obj = (pmesh, fes, AZ)

    else:
        M = mfem.SchwarzSmoother(pmesh, ref_level, fes, mat)
        M._linked_obj = (pmesh, fes, mat)

    return M


@prc.block
def ams(singular=False, **kwargs):
    prc = kwargs.pop('prc')
    blockname = kwargs.pop('blockname')
    print_level = kwargs.pop('print_level', -1)

    row = prc.get_row_by_name(blockname)
    col = prc.get_col_by_name(blockname)
    mat = prc.get_operator_block(row, col)
    fes = prc.get_test_fespace(blockname)
    inv_ams = mfem.HypreAMS(mat, fes)
    if singular:
        inv_ams.SetSingularProblem()
    inv_ams.SetPrintLevel(print_level)
    inv_ams.iterative_mode = False
    return inv_ams


@prc.block
def boomerAMG(**kwargs):
    prc = kwargs.pop('prc')
    blockname = kwargs.pop('blockname')
    print_level = kwargs.pop('print_level', -1)

    row = prc.get_row_by_name(blockname)
    col = prc.get_col_by_name(blockname)
    mat = prc.get_operator_block(row, col)

    inv_boomeramg = mfem.HypreBoomerAMG(mat)
    inv_boomeramg.SetPrintLevel(print_level)
    inv_boomeramg.iterative_mode = False

    return inv_boomeramg


@prc.block
def schur(*names, **kwargs):
    # schur("A1", "B1", scale=(1.0, 1e3))
    prc = kwargs.pop('prc')
    blockname = kwargs.pop('blockname')

    r0 = prc.get_row_by_name(blockname)
    c0 = prc.get_col_by_name(blockname)

    scales = kwargs.pop('scale', [1]*len(names))
    print_level = kwargs.pop('print_level', -1)

    S = []
    for name, scale in zip(names, scales):
        r1 = prc.get_row_by_name(name)
        c1 = prc.get_col_by_name(name)
        B = prc.get_operator_block(r0, c1)
        Bt = prc.get_operator_block(r1, c0)

        B0 = prc.get_operator_block(r1, c1)
        if use_parallel:
            Bt = Bt.Transpose()
            Bt = Bt.Transpose()
            Md = mfem.HypreParVector(MPI.COMM_WORLD,
                                     B0.GetGlobalNumRows(),
                                     B0.GetColStarts())
        else:
            Bt = Bt.Copy()
            Md = mfem.Vector()
        B0.GetDiag(Md)
        Md *= scale
        if use_parallel:

            Bt.InvScaleRows(Md)
            S.append(mfem.ParMult(B, Bt))
        else:
            S.append(mfem.Mult(B, Bt))

    if use_parallel:
        from mfem.common.parcsr_extra import ToHypreParCSR, ToScipyCoo

        S2 = [ToScipyCoo(s) for s in S]
        for s in S2[1:]:
            S2[0] = S2[0]+s
        S = ToHypreParCSR(S2[0].tocsr())
        invA0 = mfem.HypreBoomerAMG(S)

    else:
        from mfem.common.sparse_utils import sparsemat_to_scipycsr

        S2 = [sparsemat_to_scipycsr(s).tocoo() for s in S]
        for s in S2[1:]:
            S2[0] = S2[0]+s
        S = mfem.SparseMatrix(S2.tocsr())
        invA0 = mfem.DSmoother(S)

    invA0.iterative_mode = False
    invA0.SetPrintLevel(print_level)
    invA0._S = S

    return invA0


@prc.block
def mumps(guiname, **kwargs):
    # mumps("mumps1")
    from petram.solver.mumps_model import MUMPSPreconditioner
    prc = kwargs.pop('prc')
    blockname = kwargs.pop('blockname')
    kwargs = {}
    if "silent" in kwargs:
        kwargs['silent'] = kwargs.pop('silent')
    r0 = prc.get_row_by_name(blockname)
    c0 = prc.get_col_by_name(blockname)
    A0 = prc.get_operator_block(r0, c0)

    invA0 = MUMPSPreconditioner(A0, gui=prc.gui[guiname],
                                engine=prc.engine,
                                **kwargs)
    return invA0


@prc.block
def gmres(atol=0.0, rtol=0.0, max_num_iter=5,
          kdim=50, print_level=-1,
          preconditioner=None, **kwargs):
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


@prc.block
def fgmres(atol=0.0, rtol=0.0, max_num_iter=5,
           kdim=50, print_level=-1,
           preconditioner=None, **kwargs):
    prc = kwargs.pop('prc')
    blockname = kwargs.pop('blockname')

    if use_parallel:
        fgmres = mfem.FGMRESSolver(MPI.COMM_WORLD)
    else:
        fgmres = mfem.FGMRESSolver()
    fgmres.iterative_mode = False
    fgmres.SetRelTol(rtol)
    fgmres.SetAbsTol(atol)
    fgmres.SetMaxIter(max_num_iter)
    fgmres.SetPrintLevel(print_level)
    fgmres.SetKDim(kdim)
    r0 = prc.get_row_by_name(blockname)
    c0 = prc.get_col_by_name(blockname)

    A0 = prc.get_operator_block(r0, c0)

    fgmres.SetOperator(A0)
    if preconditioner is not None:
        fgmres.SetPreconditioner(preconditioner)
        # keep this object from being freed...
        fgmres._prc = preconditioner
    return fgmres


@prc.block
def pcg(atol=0.0, rtol=0.0, max_num_iter=5,
        print_level=-1, preconditioner=None, **kwargs):
    prc = kwargs.pop('prc')
    blockname = kwargs.pop('blockname')

    if use_parallel:
        pcg = mfem.CGSolver(MPI.COMM_WORLD)
    else:
        pgc = mfem.CGSolver()
    pcg.iterative_mode = False
    pcg.SetRelTol(rtol)
    pcg.SetAbsTol(atol)
    pcg.SetMaxIter(max_num_iter)
    pcg.SetPrintLevel(print_level)
    r0 = prc.get_row_by_name(blockname)
    c0 = prc.get_col_by_name(blockname)

    A0 = prc.get_operator_block(r0, c0)

    pcg.SetOperator(A0)
    if preconditioner is not None:
        pcg.SetPreconditioner(preconditioner)
        # keep this object from being freed...
        pcg._prc = preconditioner
    return pcg


@prc.block
def bicgstab(atol=0.0, rtol=0.0, max_num_iter=5,
             print_level=-1, preconditioner=None, **kwargs):
    prc = kwargs.pop('prc')
    blockname = kwargs.pop('blockname')

    if use_parallel:
        bicgstab = mfem.BiCGSTABSolver(MPI.COMM_WORLD)
    else:
        bicgstab = mfem.BiCGSTABSolver()
    bicgstab.iterative_mode = False
    bicgstab.SetRelTol(rtol)
    bicgstab.SetAbsTol(atol)
    bicgstab.SetMaxIter(max_num_iter)
    bicgstab.SetPrintLevel(print_level)
    r0 = prc.get_row_by_name(blockname)
    c0 = prc.get_col_by_name(blockname)

    A0 = prc.get_operator_block(r0, c0)

    bicgstab.SetOperator(A0)
    if preconditioner is not None:
        bicgstab.SetPreconditioner(preconditioner)
        # keep this object from being freed...
        bicgstab._prc = preconditioner
    return bicgstab


# these are here to use them in script w/o disnginguishing
# if mfem is mfem.par or mfem.ser
BlockDiagonalPreconditioner = mfem.BlockDiagonalPreconditioner
BlockLowerTriangularPreconditioner = mfem.BlockLowerTriangularPreconditioner


class GenericPreconditioner(mfem.Solver, PrcCommon):
    def __init__(self, gen):
        self.offset = gen.opr.RowOffsets()
        self.mult_func = gen.mult_func
        self.setoperator_func = gen.setoperator_func
        self.name = gen.name
        super(GenericPreconditioner, self).__init__()
        self.copy_param(gen)

    def Mult(self, *args):
        return self.mult_func(self, *args)

    def SetOperator(self, opr):
        opr = mfem.Opr2BlockOpr(opr)
        self._opr = weakref.ref(opr)
        self.offset = opr.RowOffsets()
        return self.setoperator_func(self, opr)
