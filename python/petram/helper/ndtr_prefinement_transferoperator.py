import os
import sys
import numpy as np

from petram.mfem_config import use_parallel
#use_parallel = True

if use_parallel:
    from mpi4py import MPI
    from petram.helper.mpi_recipes import *
    import mfem.par as mfem
    from mfem.common.mpi_debug import nicePrint
else:
    import mfem.ser as mfem
    nicePrint = print

class NDTrPRefinementTransferOperator(mfem.Operator):
    def __init__(self, fesl, fesh):
        if use_parallel:
            mfem.Operator.__init__(self, fesh.GetTrueVSize(), fesl.GetTrueVSize())
        else:
            mfem.Operator.__init__(self, fesh.GetVSize(), fesl.GetVSize())

        self.isvar_order = fesl.IsVariableOrder() or fesh.IsVariableOrder()
        self.fesl = fesl
        self.fesh = fesh
        self.mesh = fesl.GetMesh()
        orderl = fesl.FEColl().GetOrder()
        orderh = fesh.FEColl().GetOrder()

        self.DoFTrans = [mfem.ND_TriDofTransformation(orderl),
                         mfem.ND_TriDofTransformation(orderh)]

        self.doftransl = mfem.DofTransformation()
        self.doftransh = mfem.DofTransformation()

        self.doftransl.SetDofTransformation(self.DoFTrans[0])
        self.doftransh.SetDofTransformation(self.DoFTrans[1])
        self.doftransl.SetFaceOrientations(mfem.intArray([0]))
        self.doftransh.SetFaceOrientations(mfem.intArray([0]))
        self.doftransl.SetVDim()
        self.doftransh.SetVDim()

        if use_parallel:
           self.P = fesl.GetProlongationMatrix()
           self.R = (fesh.GetHpRestrictionMatrix() if fesh.IsVariableOrder() else
                     fesh.GetRestrictionMatrix())
           if self.P is not None:
               self.tmpl = mfem.Vector(fesl.GetVSize())
               self.tmph = mfem.Vector(fesh.GetVSize())
           else:
               if self.R is not None:
                  self.tmph = mfem.Vector(fesh.GetVSize())
        else:
            self.P = None
            self.R = None

    def Mult(self, x, y):
        if self.P is not None:
            self.P.Mult(x, self.tmpl)
            self._Mult(self.tmpl, self.tmph)
            self.R.Mult(self.tmph, y)
        elif self.R is not None:
            self._Mult(x, self.tmph)
            self.R.Mult(self.tmph, y)
        else:
            self._Mult(x, y)

    def MultTranspose(self, x, y):
        if self.P is not None:
            self.R.MultTranspose(x, self.tmph)
            self._MultTranspose(self.tmph, self.tmpl)
            self.P.MultTranspose(self.tmpl, y)
        elif self.R is not None:
            self.R.MultTranspose(x, self.tmph)
            self._MultTranspose(self.tmph, y)
        else:
            self._MultTranspose(x, y)

    def _Mult(self, x, y):  #low to hgih
        y.Assign(0.0)

        h_dofs = mfem.intArray()
        l_dofs = mfem.intArray()
        h_vdofs = mfem.intArray()
        l_vdofs = mfem.intArray()

        T = mfem.IsoparametricTransformation()
        loc_prol = mfem.DenseMatrix()
        subX = mfem.Vector()
        subY = mfem.Vector()

        vdim = self.fesl.GetVDim()
        cached_geom = -1

        for i in range(self.mesh.GetNFaces()):
            self.fesl.GetFaceDofs(i, l_dofs)
            self.fesh.GetFaceDofs(i, h_dofs)

            geom = self.mesh.GetFaceGeometry(i)

            if geom == mfem.Geometry.TRIANGLE:
                doftl = self.doftransl
                dofth = self.doftransh
            else:
                doftl = None
                dofth = None

            if geom != cached_geom or self.isvar_order:
                h_fe = self.fesh.GetFaceElement(i)
                l_fe = self.fesl.GetFaceElement(i)
                T.SetIdentityTransformation(h_fe.GetGeomType())
                h_fe.GetTransferMatrix(l_fe, T, loc_prol)
                subY.SetSize(loc_prol.Height())
                cached_geom = geom

            for vd in range(vdim):
                l_dofs.Copy(l_vdofs);
                self.fesl.DofsToVDofs(vd, l_vdofs)
                h_dofs.Copy(h_vdofs);
                self.fesh.DofsToVDofs(vd, h_vdofs)

                x.GetSubVector(l_vdofs, subX);
                if doftl is not None:
                   doftl.InvTransformPrimal(subX)
                loc_prol.Mult(subX, subY);
                if dofth is not None:
                   dofth.TransformPrimal(subY)
                y.SetSubVector(h_vdofs, subY)


    def _MultTranspose(self, x, y): #high to low
        y.Assign(0.0)

        doftrans_h = mfem.DofTransformation()
        doftrans_l = mfem.DofTransformation()

        h_dofs = mfem.intArray()
        l_dofs = mfem.intArray()
        h_vdofs = mfem.intArray()
        l_vdofs = mfem.intArray()

        T = mfem.IsoparametricTransformation()
        loc_prol = mfem.DenseMatrix()
        subX = mfem.Vector()
        subY = mfem.Vector()

        vdim = self.fesl.GetVDim()
        cached_geom = -1

        processed = np.zeros(self.fesh.GetVSize())
        for i in range(self.mesh.GetNFaces()):
            self.fesl.GetFaceDofs(i, l_dofs)
            self.fesh.GetFaceDofs(i, h_dofs)

            geom = self.mesh.GetFaceGeometry(i)

            if geom == mfem.Geometry.TRIANGLE:
                doftl = self.doftransl
                dofth = self.doftransh
            else:
                doftl = None
                dofth = None

            if geom != cached_geom or self.isvar_order:
                h_fe = self.fesh.GetFaceElement(i)
                l_fe = self.fesl.GetFaceElement(i)
                T.SetIdentityTransformation(h_fe.GetGeomType())
                h_fe.GetTransferMatrix(l_fe, T, loc_prol)
                loc_prol.Transpose()
                subY.SetSize(loc_prol.Height())
                cached_geom = geom

            for vd in range(vdim):
                l_dofs.Copy(l_vdofs);
                self.fesl.DofsToVDofs(vd, l_vdofs)
                h_dofs.Copy(h_vdofs);
                self.fesh.DofsToVDofs(vd, h_vdofs)

                x.GetSubVector(h_vdofs, subX)

                if dofth is not None:
                    dofth.InvTransformDual(subX)

                for p in range(h_dofs.Size()):
                    if processed[self.fesl.DecodeDof(h_dofs[p])]:
                         subX[p] = 0.0

                loc_prol.Mult(subX, subY)
                if doftl is not None:
                    doftl.InvTransformDual(subY)

                y.AddElementVector(l_vdofs, subY)

            for p in range(h_dofs.Size()):
                processed[self.fesl.DecodeDof(h_dofs[p])] = 1


## standalone test

if __name__ == '__main__':
    mesh = mfem.Mesh.MakeCartesian3D(1, 1, 1, mfem.Element.TETRAHEDRON)
    if use_parallel:
        nicePrint("num face (serial)", mesh.GetNFaces())
        mesh = mfem.ParMesh(MPI.COMM_WORLD, mesh)
    ess_bdr = mfem.intArray([1]*mesh.bdr_attributes.Max())


    @mfem.jit.vector(vdim=3)
    def Esol(pt):
        return np.array([0, pt[1], 0])

    if use_parallel:
        FES = mfem.ParFiniteElementSpace
        GF = mfem.ParGridFunction
    else:
        FES = mfem.FiniteElementSpace
        GF = mfem.GridFunction

    # reference
    nicePrint("ND element")
    fec1 = mfem.ND_FECollection(1, 3)
    fec2 = mfem.ND_FECollection(2, 3)
    fes1 = FES(mesh, fec1)
    fes2 = FES(mesh, fec2)

    E1o = GF(fes1)
    E2o = GF(fes2)
    E1o.Assign(0.0)
    E2o.Assign(0.0)

    E1o.ProjectCoefficient(Esol)

    org_data = E1o.GetDataArray().copy()

    if use_parallel:
        opr = mfem.TrueTransferOperator(fes1, fes2)
    else:
        opr = mfem.TransferOperator(fes1, fes2)

    opr.Mult(E1o, E2o)
    opr.MultTranspose(E2o, E1o)

    nicePrint("after Mult->MultTranspose")
    nicePrint(E1o.GetDataArray())

    nicePrint("ND Trace element")
    fec1 = mfem.ND_Trace_FECollection(1, 3)
    fec2 = mfem.ND_Trace_FECollection(2, 3)
    fes1 = FES(mesh, fec1)
    fes2 = FES(mesh, fec2)

    E1 = GF(fes1)
    E2 = GF(fes2)
    E1.GetDataArray()[:] = org_data
    E2.Assign(0.0)

    opr = NDTrPRefinementTransferOperator(fes1, fes2)
    opr.Mult(E1, E2)
    opr.MultTranspose(E2, E1)
    nicePrint("after Mult->MultTranspose")
    nicePrint(E1.GetDataArray())

    nicePrint("difference")
    nicePrint(abs(E1.GetDataArray()-E1o.GetDataArray()))
