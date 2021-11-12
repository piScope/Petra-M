from petram.mfem_config import use_parallel
if use_parallel:
    from petram.helper.mpi_recipes import *
    import mfem.par as mfem
else:
    import mfem.ser as mfem


class HierarchicalFiniteElementSpace(object):
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

    def get_or_allocate_transfer(self, key1, key2):
        if (key1, key2) in self._p_storage:
            return self._p_storage[(key1, key2)]
        else:
            fes1 = self._fes_storage[key1]
            fes2 = self._fes_storage[key2]
            P = self._owner.new_transfer_operator(fes1, fes2)
            self._p_storage[(key1, key2)] = P
        return P

    def add_uniformly_refined_level(self, name):

        emesh_idx, old_refine, element, order, fecdim, vdim = self._dataset[name][-1]

        new_refine = old_refine+1
        if not (emesh_idx, new_refine) in self._refined_mesh_storage:
            m = self._refined_mesh_storage[finest_data[0]]
            m2 = self._owner.new_mesh_from_mesh(m)
            m2.UniformRefinement()
            m2.GetEdgeVertexTable()
        else:
            m2 = self._refined_mesh_storage[(emesh_idx, new_refine)]

        fec = self.get_or_allocate_fec(element, order, fecdim)

        key1 = (emesh_idx, old_refine, element, order, fecdim, vdim)
        key2 = (emesh_idx, new_refine, element, order, fecdim, vdim)

        fes1 = self.get_or_allocate_fes(*key1)
        fes2 = self.get_or_allocate_fes(*key2)
        P = self.get_or_allocate_transfer(key1, key2)

        h = self._hierarchies[name]
        h.AddLevel(m2, fes2, P, False, False, False)

        self._dataset[name].append(key2)
        return len(self._dataset[name])

    def add_order_refined_level(self, name, inc=1):
        emesh_idx, old_refine, element, order, fecdim, vdim = self._dataset[name][-1]

        m = self._refined_mesh_storage[(emesh_idx, old_refine)]

        key1 = (emesh_idx, old_refine, element, order, fecdim, vdim)
        key2 = (emesh_idx, old_refine, element, order+inc, fecdim, vdim)

        fes1 = self.get_or_allocate_fes(*key1)
        fes2 = self.get_or_allocate_fes(*key2)
        P = self.get_or_allocate_transfer(key1, key2)

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
                        'vdim': vdim, }
        return None

    def get_fes_emesh_idx(self, fes):
        info = self.get_fes_info(fes)
        if info is not None:
            return info['emesh_idx']
        else:
            return None

    def get_fes_levels(self, name):
        return len(self._dataset[name])
