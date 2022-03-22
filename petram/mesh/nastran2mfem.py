
'''
    read .nas file and make MFEM mesh file

    note:
       it attempts to eliminate low order surfaces.
       it supports 2nd order element only for a pure tet mesh.
    supported card
        GRID
        CTETRA
        CHEXA
        CTRIA6
        CTRIA3
        CQUAD8
        PSOLID
        PSHELL

    index in nas starts from 1.
    index in mfem starts from 0.

'''
import numba
from numba import jit, types, prange
import numpy as np
import re
from scipy.sparse import coo_matrix


fwidth = 16


class NASReader(object):
    def __init__(self, filename):
        self.filename = filename
        self.dataset = None

    def load(self):
        # check format length
        num_lines = sum(1 for line in open(self.filename))
        num_lines_base = num_lines/20
        print("number of lines in file: ", num_lines)
        fid = open(self.filename, 'r')

        print("checking format...")
        counter = 0
        counter2 = 0
        while True:
            l = fid.readline()
            if l == '':
                print("cannot determine file format whether it is small or large format")
                print("processing the file assuming it is a small format file")
                globals()['fwidth'] = 8
                break
            if l.startswith('+CONT'):
                globals()['fwidth'] = 8
                print('short format ' + str(num_lines) + ' lines')
                break
            elif l.startswith('*CONT'):
                globals()['fwidth'] = 16
                print('long format ' + str(num_lines) + ' lines')
                break
            counter = counter + 1
            if int(counter % (num_lines/5)) == 0:
                print("read ", (counter2*20), "% done")
                counter2 = counter2 + 1
        fid.close()

        print("...done")

        fid = open(self.filename, 'r')
        while True:
            l = fid.readline()
            if l.startswith('$ Grid data section'):
                break

        num_grid = 0
        while True:
            l = fid.readline()
            if l.startswith('GRID'):
                num_grid = num_grid + 1
            if l.startswith('$ Element data section'):
                break
        fid.close()
        fid = open(self.filename, 'r')
        cl = ''

        while True:
            l = self._read_line(fid)
            if l.startswith('$ Grid data section'):
                break

        grids = [None]*num_grid  # reading grid
        for i in range(num_grid):
            l = self._read_line(fid)
            if l.startswith('$ Element data section'):
                break
            g = self.parse_grid_fixed(l)
            grids[i] = np.array((float(g[3]), float(g[4]), float(g[5])))
        grids = np.vstack(grids)

        elems = {'TRIA6': [],
                 'TRIA3': [],
                 'TETRA': [],
                 'HEXA': [],
                 'QUAD4': [],
                 'QUAD8': []}  # reading elements
        print("reading elements")
        ll = 0
        while True:
            l = self._read_line(fid)
            if l.startswith('$ Property data section'):
                break
            if l.startswith('$ Element data section'):
                continue
            if l.startswith('CTRIA6'):
                elems['TRIA6'].append(self.parse_tria6_fixed(l))
            elif l.startswith('CTRIA3'):
                elems['TRIA3'].append(self.parse_tria3_fixed(l))
            elif l.startswith('CTETRA'):
                elems['TETRA'].append(self.parse_tetra_fixed(l))
            elif l.startswith('CHEXA'):
                elems['HEXA'].append(self.parse_hexa_fixed(l))
            elif l.startswith('CQUAD8'):
                elems['QUAD8'].append(self.parse_quad8_fixed(l))
            elif l.startswith('CQUAD4'):
                elems['QUAD4'].append(self.parse_quad4_fixed(l))
            else:
                print("Element not supported: " + l)
                continue
            ll += 1
            if int(ll % num_lines_base) == 0:
                print(str(int(ll/num_lines_base)*5) +
                      "% done.(" + str(ll) + ")")
        print("reading elements   (done)")
        for key in elems:
            if len(elems[key]) > 0:
                print(str(len(elems[key])) + " elements of " + key)
        new_elems = {}
        new_elems_f = {}

        edge_matrix_3d = coo_matrix((len(grids), len(grids)), dtype=bool)
        has_ho = False

        is_3d = len(elems['TETRA']) > 0 or len(elems['HEXA']) > 0

        if len(elems['TETRA']) > 0:
            print("Processing TETRA...")
            TETRA = np.vstack([np.array([int(x) for x in g[3:7]])
                               for g in elems['TETRA']])
            # [np.array((int(g[3]), int(g[4]), int(g[5]), int(g[6]),))
            #       for g in elems['TETRA']])
            TETRAF = np.vstack([np.array([int(x) for x in g[3:]])
                                for g in elems['TETRA']])
            TETRA_ATTR = np.array([int(g[2])
                                   for g in elems['TETRA']])  # PSOLID ID
            idx = [len(np.unique(x)) == 4 for x in TETRA]
            if not all(idx):
                print("some TETRA has no volume")
                TETRA = TETRA[idx, :]
                TETRA_ATTR = TETRA_ATTR[idx]

            new_elems['TETRA'] = TETRA-1
            new_elems['TETRA_ATTR'] = TETRA_ATTR
            new_elems_f['TETRA'] = TETRAF-1

            idx = [(0, 1), (1, 2), (2, 0), (3, 0),
                   (3, 1), (3, 2)]

            d1 = np.hstack([new_elems['TETRA'][:, x] for x, y in idx])
            d2 = np.hstack([new_elems['TETRA'][:, y] for x, y in idx])
            d3 = np.ones(len(d2), dtype=bool)

            edge_rows_tet = np.max(np.vstack((d1, d2)), axis=0)
            edge_cols_tet = np.min(np.vstack((d1, d2)), axis=0)
            # edge_rows_tet = np.hstack([d1, d2])#max(np.vstack((d1, d2)), axis=0)
            # edge_cols_tet = np.hstack([d2, d1])#np.min(np.vstack((d1, d2)), axis=0)

            edge_matrix_3d.row = np.hstack((edge_matrix_3d.row, edge_rows_tet))
            edge_matrix_3d.col = np.hstack((edge_matrix_3d.col, edge_cols_tet))
            edge_matrix_3d.data = np.hstack((edge_matrix_3d.data, d3))
            print("done")

        if len(elems['HEXA']) > 0:
            print("Processing HEXA...")
            HEXA = np.vstack([np.array((int(g[3]), int(g[4]), int(g[5]), int(g[6]),
                                        int(g[7]), int(g[8]), int(g[9]), int(g[10]),))
                              for g in elems['HEXA']])
            TEXAF = np.vstack([np.array([int(x) for x in g[3:]])
                               for g in elems['HEXA']])
            HEXA_ATTR = np.array([int(g[2])
                                  for g in elems['HEXA']])  # PSOLID ID
            new_elems['HEXA'] = HEXA-1
            new_elems['HEXA_ATTR'] = HEXA_ATTR
            new_elems_f['HEXA'] = HEXAF-1

            idx = [(0, 1), (1, 2), (2, 3), (3, 0),
                   (0, 4), (1, 5), (2, 6), (3, 7),
                   (4, 5), (5, 6), (6, 7), (7, 0), ]

            d1 = np.hstack([new_elems['HEXA'][:, x] for x, y in idx])
            d2 = np.hstack([new_elems['HEXA'][:, y] for x, y in idx])
            d3 = np.ones(len(d2), dtype=bool)

            edge_rows_hex = np.max(np.vstack((d1, d2)), axis=0)
            edge_cols_hex = np.min(np.vstack((d1, d2)), axis=0)
            # edge_rows_tet = np.hstack([d1, d2])#max(np.vstack((d1, d2)), axis=0)
            # edge_cols_tet = np.hstack([d2, d1])#np.min(np.vstack((d1, d2)), axis=0)

            edge_matrix_3d.row = np.hstack((edge_matrix_3d.row, edge_rows_hex))
            edge_matrix_3d.col = np.hstack((edge_matrix_3d.col, edge_cols_hex))
            edge_matrix_3d.data = np.hstack((edge_matrix_3d.data, d3))
            print("done")

        edge_matrix_3d = edge_matrix_3d.tocsr()

        if len(elems['TRIA6']) > 0:
            print("Processing TRIA6...")
            TRIA6 = np.vstack([np.array((int(g[3]), int(g[4]), int(g[5]), ))
                               for g in elems['TRIA6']])
            TRIA6F = np.vstack([np.array([int(x) for x in g[3:]])
                                for g in elems['TRIA6']])
            TRIA6_ATTR = np.array([int(g[2]) for g in elems['TRIA6']])

            idx = [len(np.unique(x)) == 3 for x in TRIA6]
            if not all(idx):
                print("some TRIA6 has no area")
                TRIA6 = TRIA6[idx, :]
                TRIA6F = TRIA6F[idx, :]
                TRIA6_ATTR = TRIA6_ATTR[idx]

            idx = [(0, 1), (1, 2), (2, 0)]

            TRIA6 = TRIA6 - 1

            d1 = np.hstack([TRIA6[:, x] for x, y in idx])
            d2 = np.hstack([TRIA6[:, y] for x, y in idx])
            rows = np.max(np.vstack((d1, d2)), axis=0)
            cols = np.min(np.vstack((d1, d2)), axis=0)

            flags = np.array(edge_matrix_3d[rows, cols].reshape(3, -1))
            flags = (np.sum(flags, axis=0) == 3).flatten()

            if np.sum(flags) != flags.shape[0]:
                print("some TRIA6 is not surface of TETRA!",
                      flags.shape[0]-np.sum(flags))
                # save the location of strange tria6
                self.garbage = TRIA6[np.logical_not(flags), :]
                TRIA6 = TRIA6[flags, :]
                TRIA6_ATTR = TRIA6_ATTR[flags]

            self.edge_matrix_3d = edge_matrix_3d

            new_elems['TRIA6'] = TRIA6
            new_elems['TRIA6_ATTR'] = TRIA6_ATTR
            new_elems_f['TRIA6'] = TRIA6F-1

            if is_3d:
                if len(elems['QUAD8']) > 0:
                    print(
                        "!!!! this 3D mesh file contains ho quad. this is not supported.")
                    print("!!! linear mesh will be generated")
                else:
                    has_ho = True
            print("done")

        if len(elems['TRIA3']) > 0:
            TRIA3 = np.vstack([np.array((int(g[3]), int(g[4]), int(g[5]), ))
                               for g in elems['TRIA3']])
            TRIA3_ATTR = np.array([int(g[2])
                                   for g in elems['TRIA3']])  # PSHELL ID
            new_elems['TRIA3'] = TRIA3-1
            new_elems['TRIA3_ATTR'] = TRIA3_ATTR
            new_elems_f['TRIA3'] = TRIA3-1

        if len(elems['QUAD8']) > 0:
            print("Processing QUAD8...")
            QUAD8 = np.vstack([np.array((int(g[3]), int(g[4]), int(g[5]), int(g[6])))
                               for g in elems['QUAD8']])
            QUAD8F = np.vstack([np.array([int(x) for x in g[3:]])
                                for g in elems['QUAD8']])
            QUAD8_ATTR = np.array([int(g[2])
                                   for g in elems['QUAD8']])  # PSHELL ID
            new_elems['QUAD8'] = QUAD8-1
            new_elems['QUAD8_ATTR'] = QUAD8_ATTR
            new_elems_f['QUAD8'] = QUAD8F-1

            print("done")

        if len(elems['QUAD4']) > 0:
            QUAD4 = np.vstack([np.array((int(g[3]), int(g[4]), int(g[5]), int(g[6])))
                               for g in elems['QUAD4']])
            QUAD4_ATTR = np.array([int(g[2])
                                   for g in elems['QUAD4']])  # PSHELL ID
            new_elems['QUAD4'] = QUAD4-1
            new_elems['QUAD4_ATTR'] = QUAD4_ATTR
            new_elems_f['QUAD4'] = QUAD4-1

        elems = new_elems

        props = {'PSOLID': [],
                 'PSHELL': []}
        print("reading shell/solid")
        while True:
            l = self._read_line(fid)
            if l.startswith('ENDDATA'):
                break
            if l.startswith('PSOLID'):
                props['PSOLID'].append(self.parse_psolid_fixed(l))
            elif l.startswith('PSHELL'):
                props['PSHELL'].append(self.parse_pshell_fixed(l))
            else:
                pass
            ll += 1
            if ll % num_lines_base == 0:
                print(str(ll/num_lines_base) + "% done.")
        print("reading shell/solid ...(done)")
        PSHELL = np.array([int(g[1]) for g in props['PSHELL']])  # PSHELL
        PSOLID = np.array([int(g[1]) for g in props['PSOLID']])  # PSOLID

        props = {'PSOLID': PSOLID,
                 'PSHELL': PSHELL}

        dataset = {'PROPS': props,
                   'ELEMS': elems,
                   'GRIDS': grids}
        if has_ho:
            dataset["ELEMS_HO"] = new_elems_f
            self.has_ho = True
        else:
            self.has_ho = False

        fid.close()
        self.dataset = dataset

    def plot_tet(self, idx, **kwargs):
        from ifigure.interactive import solid

        grids = self.dataset['GRIDS']
        tet = self.dataset['ELEMS']['TETRA']
        i = tet[idx]
        pts = [grids[[i[0], i[1], i[2]]],
               grids[[i[1], i[2], i[3]]],
               grids[[i[2], i[3], i[0]]],
               grids[[i[3], i[0], i[1]]]]
        pts = np.rollaxis(np.dstack(pts), 2, 0)
        solid(pts, **kwargs)

    def _read_line(self, fid):
        cl = ''
        while True:
            line = fid.readline()
            l = line.rstrip("\r\n")
            if (l.startswith('+CONT') or
                    l.startswith('*CONT')):
                l = cl + l[8:]
            if (l.strip().endswith('+CONT') or
                    l.strip().endswith('*CONT')):
                cl = l.strip()[:-5]
                continue
            break
        return l

    def parse_grid_fixed(self, l):
        d = fwidth
        if fwidth == 16:
            l = ' '*8+l
#        cards= [l[d*i:d*(i+1)].strip() for i in range(8)]
        cards = re.findall('.'*d, l)
        return cards

    def parse_tetra_fixed(self, l):
        d = 8
#        cards= [l[d*i:d*(i+1)].strip() for i in range(13)]
        cards = re.findall('.'*d, l)
        return cards

    def parse_tria6_fixed(self, l):
        d = 8
#        cards= [l[d*i:d*(i+1)].strip() for i in range(9)]
        cards = re.findall('.'*d, l)
        return cards

    def parse_tria3_fixed(self, l):
        d = 8
#        cards= [l[d*i:d*(i+1)].strip() for i in range(9)]
        cards = re.findall('.'*d, l)
        return cards

    def parse_hexa_fixed(self, l):
        d = 8
#        cards= [l[d*i:d*(i+1)].strip() for i in range(23)]
        cards = re.findall('.'*d, l)
        return cards

    def parse_quad8_fixed(self, l):
        d = 8
#        cards= [l[d*i:d*(i+1)].strip() for i in range(11)]
        cards = re.findall('.'*d, l)
        return cards

    def parse_quad4_fixed(self, l):
        d = 8
#        cards= [l[d*i:d*(i+1)].strip() for i in range(11)]
        cards = re.findall('.'*d, l)
        return cards

    def parse_pshell_fixed(self, l):
        d = 8
#        cards= [l[d*i:d*(i+1)].strip() for i in range(2)]
        cards = re.findall('.'*d, l)
        return cards

    def parse_psolid_fixed(self, l):
        d = 8
        cards = re.findall('.'*d, l)
#        cards= [l[d*i:d*(i+1)].strip() for i in range(2)]
        return cards


def dist(p1, p2):
    d = np.sqrt(np.sum(p1 - p2)**2)
    return d


def linearized_grids(grids):
    new_grids = np.vstack([grids[0], grids[1],  grids[2], grids[3],
                           (grids[0] + grids[1])/2.0,
                           (grids[1] + grids[2])/2.0,
                           (grids[0] + grids[2])/2.0,
                           (grids[0] + grids[3])/2.0,
                           (grids[1] + grids[3])/2.0,
                           (grids[2] + grids[3])/2.0])
    return new_grids


def prepare_ho_node(mesh):
    import mfem.ser as mfem
    nfec = mfem.H1_FECollection(2,
                                mesh.Dimension())
    # mfem.BasisType.ClosedUniform)
    nfes = mfem.FiniteElementSpace(mesh,
                                   nfec,
                                   mesh.SpaceDimension(),
                                   mfem.Ordering.byVDIM)
    mesh.SetNodalFESpace(nfes)
    mesh._linked_obj = (nfes, nfec)


def write_ho_tet_mesh(filename, reader, ho_thre=1e-20):
    import mfem.ser as mfem

    @jit(numba.int64(numba.float64[:], numba.float64[:, :]), cache=False)
    def find_closest0(ptx, grids):
        x1 = ptx[0]
        y1 = ptx[1]
        z1 = ptx[2]
        for jj in range(10):
            d2 = ((x1 - grids[jj, 0])**2 +
                  (y1 - grids[jj, 1])**2 +
                  (z1 - grids[jj, 2])**2)
            if d2 < ho_thre:
                return jj
        assert(False)
        return -1

    @jit(parallel=True, cache=False)
    def find_closest_tet_ptx(ptx, grids):
        idx = np.empty(10, dtype=numba.int64)
        for jj in prange(10):
            idx[jj] = find_closest0(ptx[jj], grids)
        return idx

    mesh = mfem.Mesh(filename)

    prepare_ho_node(mesh)
    n = mesh.GetNE()

    node_gf = mesh.GetNodes()
    nfes = node_gf.FESpace()
    nodes = node_gf.GetDataArray().copy()

    node_gf2 = mfem.GridFunction(mesh._linked_obj[0])
    nodes2 = node_gf2.GetDataArray()

    grid_all = reader.dataset['GRIDS']
    i_tetra = reader.dataset["ELEMS_HO"]["TETRA"]

    for i in range(n):
        # for i in [0]:
        vdofs = nfes.GetElementVDofs(i)
        ptx = nodes[vdofs].reshape(3, -1).transpose()
        grids = grid_all[i_tetra[i]]

        new_grids = linearized_grids(grids)

        idx = find_closest_tet_ptx(ptx, new_grids)

        tmp = grids[idx]
        data = np.hstack((tmp[:, 0], tmp[:, 1], tmp[:, 2]))
        nodes2[vdofs] = data

        if (i % 50000) == 0:
            print("processing elements "+str(i) + "/" + str(n))

    node_gf.Assign(nodes2)

    filename2 = filename[:-5]+"_order2.mesh"
    mesh.Save(filename2)


def write_nas2mfem(filename,  reader, exclude_bdr=None, offset=None,
                   skip_unused_bdry=True, ho_thre=1e-20, skip_ho=False):

    geom_type = {'TETRA': 4,
                 'TRIA6': 2,
                 'TRIA3': 2,
                 'HEXA':  5,
                 'QUAD8': 3,
                 'QUAD4': 3, }
    '''                     
        SEGMENT = 1
        TRIANGLE = 2
        SQUARE = 3
        TETRAHEDRON = 4
        CUBE = 5
        '''

    if exclude_bdr is None:
        exclude_bdr = []
    if offset is None:
        offset = [0.0, 0.0, 0.0]
    if reader.dataset is None:
        reader.load()

    data = reader.dataset
    fid = open(filename, 'w')

    grid = data['GRIDS']
    elems = data['ELEMS']

    el_3d = ['TETRA', 'HEXA']
    if 'TETRA' in elems:
        el_2d = ['TRIA6', 'TRIA3']
    else:
        el_2d = ['QUAD8', 'QUAD4']

    n3d = np.sum([len(elems[x]) for x in el_3d if x in elems])

    if n3d > 0:
        unique_grids = list(np.unique(np.hstack([elems[name].flatten()
                                                 for name in el_3d if name in elems])))
    else:
        unique_grids = list(np.unique(np.hstack([elems[name].flatten()
                                                 for name in el_2d if name in elems])))
    nvtc = len(unique_grids)
    print('unique_grid (done)....' + str(nvtc))

    ndim = grid.shape[-1]
    nelem = 0
    nbdry = 0

    for k in elems:
        if k in el_3d:
            nelem = nelem + len(elems[k+'_ATTR'])
        if k in el_2d:
            tmp = [x for x in elems[k+'_ATTR'] if not x in exclude_bdr]
            nbdry = nbdry + len(tmp)

    fid.write('MFEM mesh v1.0\n')
    fid.write('\n')
    fid.write('dimension\n')
    fid.write(str(ndim) + '\n')
    fid.write('\n')
    fid.write('elements\n')
    fid.write(str(nelem) + '\n')

    rev_map = {unique_grids[x]: x for x in range(len(unique_grids))}
    reader.grid_mapping = (unique_grids, rev_map)

    for name in el_3d:
        if not name in elems:
            continue
        vidx = elems[name]
        attr = elems[name+'_ATTR']
        gtyp = geom_type[name]
        txts = [None]*len(attr)
        for i in range(len(attr)):
            txt = [str(attr[i]), str(gtyp)]
#                txt.extend([str(unique_grids.index(x)) for x in vidx[i]])
            txt.extend([str(rev_map[x]) for x in vidx[i]])
            txts[i] = ' '.join(txt)
        fid.write('\n'.join(txts))
    fid.write('\n')

    # count valid 2d elements
    # sometimes .nas contains a boundary element which are not used
    # in 3D mesh. By default we skip this

    bdry_checks = {}
    bdry_flags = {}
    for name in el_2d:
        if not name in elems:
            continue
        vidx = elems[name]
        ss = vidx.shape
        bdry_check = (np.in1d(vidx, unique_grids)).reshape(ss)
        bdry_flag = [len(x) == np.sum(x) for x in bdry_check]
        bdry_checks[name] = bdry_check
        bdry_flags[name] = bdry_flag

    n_validbdry = np.sum([np.sum(bdry_flags[x]) for x in bdry_flags])

    fid.write('boundary\n')
    fid.write(str(n_validbdry) + '\n')

    print("number of bdry in file:", nbdry)
    print("number of used bdry in file:", n_validbdry)
    for name in el_2d:
        if not name in elems:
            continue

        vidx = elems[name]

        attr = elems[name+'_ATTR']
        gtyp = geom_type[name]
        txts = [None]*n_validbdry

        bdry_flag = bdry_flags[name]
        k = 0
        for i in range(len(attr)):
            if attr[i] in exclude_bdr:
                continue
            if not bdry_flag[i]:
                continue
            txt = [str(attr[i]), str(gtyp)]
#                txt.extend([str(unique_grids.index(x)) for x in vidx[i]])
            txt.extend([str(rev_map[x]) for x in vidx[i]])
            txts[k] = ' '.join(txt)
            k = k + 1

        fid.write('\n'.join(txts))
    fid.write('\n')
    print("Writing vertices", nvtc)
    fid.write('vertices\n')
    fid.write(str(nvtc) + '\n')
    fid.write(str(ndim) + '\n')
    txts = [None]*nvtc
    for i in range(nvtc):
        txt = [str(x+offset[kk]) for kk, x in enumerate(grid[unique_grids[i]])]
        txts[i] = ' '.join(txt)
    fid.write('\n'.join(txts))
    print("Done")

    if reader.has_ho and not skip_ho:
        print("generating 2nd order mesh. this may take a while")
        write_ho_tet_mesh(filename, reader, ho_thre=ho_thre)
    fid.close()
