import os
import sys
import numpy as np

from petram.mfem_config import use_parallel
if use_parallel:
    from petram.helper.mpi_recipes import *
    import mfem.par as mfem
else:
    import mfem.ser as mfem

class NDTrPRefinementTransferOpr(mfem.Operator):
    def __init__(self, fesl, fesh):
        mfem.Operator.__init__(self, fesh.GetVSize(), fesl.GetVSize())
        
        self.isvar_order = fesl.IsVariableOrder() or fesh.IsVariableOrder()
        self.fesl = fesl
        self.fesh = fesh
        self.mesh = fesl.GetMesh()
        

    def Mult(self, x, y):  #low to hgih
        
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

        for i in range(self.mesh.GetNFaces()):
            fesl.GetFaceDofs(i, l_dofs)
            fesh.GetFaceDofs(i, h_dofs)

            geom = self.mesh.GetFaceGeometry(i)

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
                #doftrans_l.InvTransformPrimal(subX);
                loc_prol.Mult(subX, subY);
                #doftrans_h.TransformPrimal(subY);
                y.SetSubVector(h_vdofs, subY)


    def MultTranspose(self, x, y): #high to low

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
            fesl.GetFaceDofs(i, l_dofs)
            fesh.GetFaceDofs(i, h_dofs)

            geom = self.mesh.GetFaceGeometry(i)

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

                x.GetSubVector(h_vdofs, subX);
                
                for p in range(h_dofs.Size()):
                    if processed[self.fesl.DecodeDof(h_dofs[p])]:
                         subX[p] = 0.0;

                #doftrans_l.InvTransformPrimal(subX);
                loc_prol.Mult(subX, subY);
                #doftrans_h.TransformPrimal(subY);
                y.AddElementVector(l_vdofs, subY)

            for p in range(h_dofs.Size()):
                processed[self.fesl.DecodeDof(h_dofs[p])] = 1
        


