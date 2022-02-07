from petram.mfem_config import use_parallel
if use_parallel:
    from petram.helper.mpi_recipes import *
    import mfem.par as mfem
else:
    import mfem.ser as mfem

import numpy as np


class HierarchicalFiniteElementSpaces(object):
    '''
    obj.fespaces[name] : returns fes using level_idx

    fec, fes, P and refined_mesh are stored interally for reuse
    '''

    def __init__(self, owner=None):
        self._owner = owner
        self._hierarchies = {}
        self._dataset = {}
        self._fec_storage = {}
        self._fes_storage = {}
        self._p_storage = {}
        self._refined_mesh_storage = {}

    def __getitem__(self, name):
        if not name in self._dataset:
            raise KeyError()
        d = self._dataset[name]
        key1 = d[self._owner.level_idx]
        fes1 = self.get_or_allocate_fes(*key1)
        return fes1

    def __iter__(self):
        return self._dataset.__iter__()

    @property
    def hierarchies(self):
        self._hierarchies = {}

    def get_hierarchy(self, name):
        return self._hierarchies[name]

    def new_hierarchy(self, name, parameters=None):

        emesh_idx, element, order, fecdim, vdim = parameters

        if not (emesh_idx, 0) in self._refined_mesh_storage:
            self._refined_mesh_storage[(
                emesh_idx, 0)] = self._owner.emeshes[emesh_idx]

        mesh = self._refined_mesh_storage[(emesh_idx, 0)]
        mesh.GetEdgeVertexTable()

        fec = self.get_or_allocate_fec(element, order, fecdim)
        fes = self.get_or_allocate_fes(
            emesh_idx, 0, element, order, fecdim, vdim)

        h = self._owner.new_fespace_hierarchy(mesh, fes, False, False)

        self._hierarchies[name] = h
        self._dataset[name] = []
        self._dataset[name].append(
            (emesh_idx, 0, element, order, fecdim, vdim))

    def get_or_allocate_fec(self, element, order, fecdim):
        entry = (element, order, fecdim)
        if entry in self._fec_storage:
            return self._fec_storage[entry]
        else:
            fec_class = getattr(mfem, element)
            fec = fec_class(order, fecdim)
            self._fec_storage[entry] = fec
            return fec

    def get_or_allocate_fes(self, emesh_idx, refine, element, order, fecdim, vdim):
        entry = (emesh_idx, refine, element, order, fecdim, vdim)
        mesh = self._refined_mesh_storage[(emesh_idx, refine)]

        if entry in self._fes_storage:
            return self._fes_storage[entry]
        else:
            fec = self.get_or_allocate_fec(element, order, fecdim)
            fes = self._owner.new_fespace(mesh, fec, vdim)
            self._fes_storage[entry] = fes
            return fes

    def get_or_allocate_transfer(self, name, key1, key2, engine):
        if (key1, key2) in self._p_storage:
            return self._p_storage[(key1, key2)]
        else:
            P = self._new_transfer_operator(name, key1, key2, engine)
            self._p_storage[(key1, key2)] = P
        return P

    def add_same_level(self, name, engine):
        emesh_idx, old_refine, element, order, fecdim, vdim = self._dataset[name][-1]

        m2 = self._refined_mesh_storage[(emesh_idx, old_refine)]
        fec = self.get_or_allocate_fec(element, order, fecdim)

        key1 = (emesh_idx, old_refine, element, order, fecdim, vdim)

        fes1 = self.get_or_allocate_fes(*key1)
        P = self.get_or_allocate_transfer(name, key1, key1, engine)

        h = self._hierarchies[name]
        h.AddLevel(m2, fes1, P, False, False, False)

        self._dataset[name].append(key1)
        return len(self._dataset[name])

    def add_mesh_refined_level(self, name, engine, inc=1, refine_dom=None):
        emesh_idx, old_refine, element, order, fecdim, vdim = self._dataset[name][-1]

        new_refine = old_refine+1
        if not (emesh_idx, new_refine) in self._refined_mesh_storage:
            m = self._refined_mesh_storage[(emesh_idx, old_refine)]
            m2 = self._owner.new_mesh_from_mesh(m)
            for i in range(inc):
                if refine_dom is None:
                    m2.UniformRefinement()
                else:
                    attr = m2.GetAttributeArray()
                    idx = list(np.where(np.in1d(attr, refine_dom))[0])
                    idx0 = mfem.intArray(idx)
                    m2.GeneralRefinement(idx0)  # this is parallel refinement
            m2.GetEdgeVertexTable()
            self._refined_mesh_storage[(emesh_idx, old_refine+inc)] = m2
        else:
            m2 = self._refined_mesh_storage[(emesh_idx, new_refine)]

        fec = self.get_or_allocate_fec(element, order, fecdim)

        key1 = (emesh_idx, old_refine, element, order, fecdim, vdim)
        key2 = (emesh_idx, new_refine, element, order, fecdim, vdim)

        fes1 = self.get_or_allocate_fes(*key1)
        fes2 = self.get_or_allocate_fes(*key2)
        P = self.get_or_allocate_transfer(name, key1, key2, engine)

        h = self._hierarchies[name]
        h.AddLevel(m2, fes2, P, False, False, False)

        self._dataset[name].append(key2)
        return len(self._dataset[name])

    def add_order_refined_level(self, name, engine, inc=1):
        emesh_idx, old_refine, element, order, fecdim, vdim = self._dataset[name][-1]

        m = self._refined_mesh_storage[(emesh_idx, old_refine)]

        key1 = (emesh_idx, old_refine, element, order, fecdim, vdim)
        key2 = (emesh_idx, old_refine, element, order+inc, fecdim, vdim)

        fes1 = self.get_or_allocate_fes(*key1)
        fes2 = self.get_or_allocate_fes(*key2)
        P = self.get_or_allocate_transfer(name, key1, key2, engine)

        h = self._hierarchies[name]
        h.AddLevel(m, fes2, P, False, False, False)

        self._dataset[name].append(key2)
        return len(self._dataset[name])

    def get_fes_info(self, fes):
        for key in self._fes_storage:
            if self._fes_storage[key] == fes:
                emesh_idx, refine, element, order, fecdim, vdim = key
                if hasattr(fes, 'GroupComm'):
                    m = fes.GetParMesh()
                else:
                    m = fes.GetMesh()
                return {'emesh_idx': emesh_idx,
                        'refine': refine,
                        'element': element,
                        'order': order,
                        'dim': m.Dimension(),
                        'sdim': m.SpaceDimension(),
                        'vdim': vdim,
                        'fecdim': fecdim}
        return None

    def get_fes_from_info(self, info):
        emesh_idx = info['emesh_idx']
        refine = info['refine']
        element = info['element']
        order = info['order']
        vdim = info['vdim']
        fecdim = info['fecdim']
        key = emesh_idx, refine, element, order, fecdim, vdim
        return self._fes_storage[key]

    def get_fes_emesh_idx(self, fes):
        info = self.get_fes_info(fes)
        if info is not None:
            return info['emesh_idx']
        else:
            return None

    def get_fes_levels(self, name):
        return len(self._dataset[name])

    def get_mesh(self, name, ref_level=0):
        emesh_idx, refine, element, order, fecdim, vdim = self._dataset[name][-1]
        m = self._refined_mesh_storage[(emesh_idx, refine-ref_level)]
        #m = self._refined_mesh_storage[(emesh_idx, 0)]
        return m

    def _new_transfer_operator(self, name, key1, key2, engine, use_matrix_free=True):
        '''
        fes1 : coarse grid
        fes2 : fine grid
        '''
        fes1 = self._fes_storage[key1]
        fes2 = self._fes_storage[key2]

        parallel = hasattr(fes1, 'GroupComm')
        if use_matrix_free:
            if key1 == key2:
                return mfem.IdentityOperator(fes1.GetTrueVSize())
            elif parallel:
                return mfem.TrueTransferOperator(fes1, fes2)
            else:

                P1 = None if fes1.Conforming() else fes1.GetConformingProlongation()
                if P1 is None:
                    P1 = mfem.IdentityOperator(fes1.GetTrueVSize())

                R2 = None if fes2.Conforming() else fes2.GetConformingRestriction()
                if R2 is None:
                    R2 = mfem.IdentityOperator(fes2.GetTrueVSize())

                Opr1 = mfem.TransferOperator(fes1, fes2)
                opr = mfem.TripleProductOperator(
                    R2, Opr1, P1, False, False, False)
                opr._linked_opr = (R2, Opr1, P1)
                return opr

        else:
            if parallel:
                PP = mfem.OperatorPtr(mfem.Operator.Hypre_ParCSR)

                fes2.GetTransferOperator(fes1, PP)
                PP.SetOperatorOwner(False)
                P = PP.Ptr()
                return P
            else:
                assert False, "Should not come here"
