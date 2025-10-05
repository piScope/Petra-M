
import numpy as np


def do_connect_pairs(ll):
    d = {}

    flags = [False, ]*len(ll)
    count = 1
    d[ll[0][0]] = ll[0][1]
    flags[0] = True
    while count < len(ll):
        for k, f in enumerate(flags):
            if f:
                continue
            l = ll[k]
            if l[0] in d.keys():
                d[l[1]] = l[0]
                flags[k] = True
                count = count + 1
                break
            elif l[1] in d.keys():
                d[l[0]] = l[1]
                flags[k] = True
                count = count + 1
                break
            else:
                pass
        else:
            break
    key = list(d)[0]
    pt = [key]
    lmax = len(d)
    while d[key] != pt[0]:
        pt.append(d[key])
        key = d[key]
        if len(pt) > lmax:
            break
    if d[key] == pt[0]:
        pt.append(pt[0])

    ll = [l for l, f in zip(ll, flags) if not f]
    return pt, ll


def connect_pairs(ll):
    ret = []
    rest = ll
    while len(rest) != 0:
        pt, rest = do_connect_pairs(rest)
        ret.append(pt)
    if len(ret) == 1:
        return pt
    return ret


def do_find_circle_center(p1, p2, p3,  norm):
    p = p2 - p1
    q = np.cross(norm, p)
    d = np.sqrt(np.sum(p**2))

    p = p/np.sqrt(np.sum(p**2))
    q = q/np.sqrt(np.sum(q**2))

    a = np.sum((p3-p1)*p)
    b = np.sum((p3-p1)*q)

    m = np.linalg.inv(np.array([[0, b/2.], [1., -a/2.]]))
    v = np.array([[a/2.-d/2.], [b/2.]])

    s, t = np.dot(m, v)
    c = p1 + (p2-p1)/2.+s[0]*q
    return c


def find_circle_center_radius(vv, norm):
    '''
    assuming that point are somewhat equally space, I will
    scatter three points around the circle given by input points

    this approch reduces the risk of having three points colinear,
    Note that we need to have sufficently large number of points...

    '''
    k = len(vv)-2
    ii = np.linspace(0, len(vv)-1, 4).astype(int)

    pts = [do_find_circle_center(vv[i+ii[0]], vv[i+ii[1]], vv[i+ii[2]], norm)
           for i in range(ii[1]-ii[0])]

    ctr = np.mean(pts, 0)
    r = np.mean(np.sqrt(np.sum((vv - ctr)**2, 1)))
    return ctr, r


def connect_pairs2(ll):
    '''
    connect paris of indices to make a loop

    (example)
    >>> idx = array([[1, 4],  [3, 4], [1,2], [2, 3],])
    >>> connect_pairs(idx)
    [[1, 2, 3, 4, 1]]
    '''
    if not isinstance(ll, np.ndarray):
        ll = np.array(ll)

    idx = np.where(ll[:, 0] > ll[:, 1])[0]
    t1 = ll[idx, 0]
    t2 = ll[idx, 1]
    ll[idx, 0] = t2
    ll[idx, 1] = t1

    ii = np.vstack([np.arange(ll.shape[0]), ]*2).transpose()
    d = np.ones(ll.shape[0]*ll.shape[1]).reshape(ll.shape)
    from scipy.sparse import csr_matrix, coo_matrix
    m = coo_matrix((d.flatten(), (ii.flatten(), ll.flatten())),
                   shape=(len(ll), np.max(ll+1)))
    mm = m.tocsc()
    ic = mm.indices
    icp = mm.indptr
    mm = m.tocsr()
    ir = mm.indices
    irp = mm.indptr

    def get_start(taken):
        idx = np.where(np.logical_and(np.diff(icp) == 1, taken == 0))[0]
        nz = np.where(np.logical_and(np.diff(icp) != 0, taken == 0))[0]
        if len(nz) == 0:
            return
        if len(idx) > 0:
            #print('Open end found')
            pt = (ic[icp[idx[0]]], idx[0])
        else:
            pt = (ic[icp[nz[0]]], nz[0])
        pts = [pt]
        return pts

    def hop_v(pt):
        ii = pt[1]
        ii = [icp[ii], icp[ii+1]-1]
        next = ic[ii[1]] if ic[ii[0]] == pt[0] else ic[ii[0]]
        return (next, pt[1])

    def hop_h(pt):
        ii = pt[0]
        ii = [irp[ii], irp[ii+1]-1]
        next = ir[ii[1]] if ir[ii[0]] == pt[1] else ir[ii[0]]
        return (pt[0], next)

    def trace(pts):
        loop = [pts[-1][1]]
        while True:
            pts.append(hop_v(pts[-1]))
            # rows.append(pts[-1][0])
            pts.append(hop_h(pts[-1]))
            if pts[-1][1] in loop:
                break  # open ended
            loop.append(pts[-1][1])
            if pts[-1] == pts[0]:
                break
        return loop

    taken = (icp*0)[:-1]
    loops = []
    while True:
        pts = get_start(taken)
        if pts is None:
            break
        loop = trace(pts)
        loops.append(loop)
        taken[loop] = 1

    return loops


def find_crosssetional_shape(mesh, a, b, c, d):
    '''

    inspect mesh and find geometry represents crosssection of mesh
    slice by a plane 
       ax + by + cz + d = 0
    returns
       return shapes, shapes_attr, edge2point, coords

       shape : [(array of edges),,,]
       shape_attr : domain attribute
       edge2point : edges of shape
       coords : node point of edges.

       note: shape is mixture of triangle, quad, -- hex.

    '''
    from petram.mfem_config import use_parallel
    if use_parallel:
        import mfem.par as mfem
    else:
        import mfem.ser as mfem
    v = mfem.Vector()
    mesh.GetVertices(v)
    vv = v.GetDataArray().copy()
    vv = vv.reshape(3, -1)
    val = vv[0, :]*a + vv[1, :]*b + vv[2, :]*c + d

    attrs = mesh.GetAttributeArray()

    class number_generator(object):
        def __init__(self):
            self._d = {}
            self.num = -1

        def get_number(self, p):
            try:
                if p[0] < p[1]:
                    pp = (p[0], p[1])
                else:
                    pp = (p[1], p[0])
            except:
                pp = p

            if pp in self._d:
                return self._d[pp], False
            else:
                self.num = self.num+1
                self._d[pp] = self.num
                return self.num, True

        def get_reverse_map(self):
            return {self._d[p]: p for p in self._d}

    shapes = []
    shapes_attr = []
    shape_ptx = []

    Nv = number_generator()  # vertex numbers
    Ne = number_generator()  # edge numbers

    face_done = np.zeros(mesh.GetNFaces(), dtype=bool)
    face_cache = {}

    for iele in range(mesh.GetNE()):
        inodes = mesh.GetElementVertices(iele)
        ifaces, void = mesh.GetElementFaces(iele)

        f = val[inodes]

        if np.all(f >= 0) or np.all(f <= 0):
            if np.sum(f == 0) < 3:
                continue  # element is one side of plane.
                # only edge/node may be on the plane.
            # face is on the plane
            ifaces = [x for x in ifaces if not face_done[x]]

            for iface in ifaces:
                face_done[iface] = True
                f = val[mesh.GetFaceVertices(iface)]

                if np.all(f == 0):
                    iedges, orts = mesh.GetFaceEdges(iface)
                    shape = []
                    for iedge, ort in zip(iedges, orts):
                        iv = mesh.GetEdgeVertices(iedge)
                        n1, isNew = Nv.get_number(iv[0])
                        if isNew:
                            shape_ptx.append(vv[:, iv[0]])
                        n2, isNew = Nv.get_number(iv[1])
                        if isNew:
                            shape_ptx.append(vv[:, iv[1]])

                        e1, isNew = Ne.get_number((n1, n2))

                        shape.append(e1)
                    shapes.append(shape)
                    shapes_attr.append(attrs[iele])
                    break
                else:
                    pass

        else:
            # face intersect with the plane
            shape = []
            for iface in ifaces:
                if iface in face_cache:
                    e1 = face_cache[iface]
                    shape.append(e1)
                else:
                    iedges = mesh.GetFaceEdges(iface)[0]
                    edge0 = []
                    for iedge in iedges:
                        iv = mesh.GetEdgeVertices(iedge)
                        iv1, iv2 = iv
                        f = val[iv]
                        if np.max(f) * np.min(f) < 0:
                            ptx = (vv[:, iv1]*val[iv2] - vv[:, iv2]
                                   * val[iv1])/(- val[iv1] + val[iv2])
                            idx = (iv1, iv2)
                        elif f[0] == 0:
                            ptx = vv[:, iv1]
                            idx = iv1
                        elif f[1] == 0:
                            ptx = vv[:, iv2]
                            idx = iv2
                        else:
                            continue
                        n, isNew = Nv.get_number(idx)
                        if n in edge0:
                            continue
                        if isNew:
                            shape_ptx.append(ptx)
                        edge0.append(n)
                        c = c+1
                        if c == 1:
                            break
                    if len(edge0) < 2:
                        continue
                    e1, isNew = Ne.get_number(edge0)
                    shape.append(e1)
                    face_cache[iface] = e1
            shapes.append(shape)
            shapes_attr.append(attrs[iele])

    edge2point = Ne.get_reverse_map()
    coords = np.vstack(shape_ptx)
    return shapes, shapes_attr, edge2point, coords


def find_cp_pc_parameter(mesh, abcd, e1, gsize=None, gcount=100, origin=None, attrs=None):
    '''
    make cut-plane point_clund

    cp is defined by norm (a, b, c) and d as ax+by+cz+d = 0
    e1 is one axes on cp
    e2 is computed automatically

    gsize : grid size

    '''
    from petram.mfem_config import use_parallel
    if use_parallel:
        import mfem.par as mfem
    else:
        import mfem.ser as mfem

    v = mfem.Vector()
    mesh.GetVertices(v)
    sdim = mesh.SpaceDimension()
    dim = mesh.Dimension()
    vv = v.GetDataArray().copy()

    vv = vv.reshape(sdim, -1)
    if sdim == 3:
        pass
    elif sdim == 2:
        vv = np.vstack([vv, vv[:1, :]*0])
    else:
        assert False, "1D mesh is not supported"

    norm = np.array(abcd[:3])
    norm = norm/np.sqrt(np.sum(norm**2))

    if sdim == 3 and dim == 3:
        # flag elements which are on the plane
        flag = np.zeros(mesh.GetNV(), dtype=bool)
        for iele, f in enumerate(mesh.IsElementOnPlaneArray(*abcd)):
            if not f:
                continue
            flag[mesh.GetElementVertices(iele)] = True
    else:
        flag = np.ones(mesh.GetNV(), dtype=bool)
    '''
    val =  vv[0, :]*abcd[0] + vv[1,:]*abcd[1] + vv[2,:]*abcd[2] + abcd[3]
    for iele in range(mesh.GetNE()):
        inodes = mesh.GetElementVertices(iele)
        f = val[inodes]
        
        if np.all(f > 0) or np.all(f < 0):
            continue

        if attrs is not None and not (mesh.GetAttribute(iele) in attrs):
            continue
        
        flag[mesh.GetElementVertices(iele)] = True
    '''

    points = np.transpose(vv[:, flag])

    p1 = np.sum(points*abcd[:3], 1)+abcd[3]
    dd = np.sqrt(np.sum(np.array(abcd[:3])**2))
    p1 = p1/dd
    points = points - norm.reshape(3)*p1.reshape(-1, 1)

    e2 = np.cross(norm, e1)
    e1 = np.cross(e2, norm)
    e1 = e1 / np.sqrt(np.sum(e1**2))
    e2 = e2 / np.sqrt(np.sum(e2**2))

    if len(points) == 0:
        return None

    origin = points[0] if origin is None else np.array(origin)

    x = np.sum((points-origin)*e1, -1)
    y = np.sum((points-origin)*e2, -1)

    xmax = np.max(x)
    xmin = np.min(x)
    ymax = np.max(y)
    ymin = np.min(y)

    dx = xmax-xmin
    dy = ymax-ymin

    if gsize is None:
        try:
            gsize1 = dx/(gcount[0]-1)
            gsize2 = dx/(gcount[1]-1)
        except:
            gsize1 = max(dx/(gcount-1), dy/(gcount-1))
            gsize2 = gsize1
    else:
        try:
            gsize1 = gsize[0]
            gsize2 = gsize[1]
        except:
            gsize1 = gsize
            gsize2 = gsize

    return {"origin": origin, "e1": e1, "e2": e2,
            "x": (xmin, xmax, gsize1), "y": (ymin, ymax, gsize2)}


def generate_pc_from_cpparam(origin=None, e1=None, e2=None, x=None, y=None):
    xmin, xmax, xsize = x
    ymin, ymax, ysize = y
    xx = np.linspace(xmin, xmax, int((xmax-xmin)/xsize))
    yy = np.linspace(ymin, ymax, int((ymax-ymin)/ysize))

    XX, YY = np.meshgrid(xx, yy)
    e1 = np.atleast_2d(e1)
    e2 = np.atleast_2d(e2)
    pc = np.array(origin) + e1*XX.reshape(-1, 1) + e2*YY.reshape(-1, 1)
    pc = pc.reshape(len(yy), len(xx), 3)
    return pc


def make_cp_pc(mesh, abcd, e1, gsize=None, gcount=100):
    param = find_cp_pc_parameter(mesh, abcd, e1, gsize=gsize, gcount=gcount)
    pc = generate_pc_from_cpparam(param)
    return pc
