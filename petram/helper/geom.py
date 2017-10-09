import numpy as np
def do_connect_pairs(ll):
    d = {}

    flags = [False,]*len(ll)
    count = 1
    d[ll[0][0]] = ll[0][1]
    flags[0] = True
    while count < len(ll):
       for k, f in enumerate(flags):
           if f: continue
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
    key = d.keys()[0]
    pt = [key]
    lmax = len(d.keys())
    while d[key] != pt[0]:
        pt.append(d[key])
        key = d[key]
        if len(pt) > lmax: break
    if d[key] == pt[0]: pt.append(pt[0])
   
    ll = [l for l, f in zip(ll, flags) if not f]
    return pt, ll

def connect_pairs(ll):
    ret = []
    rest = ll    
    while len(rest) != 0:
       pt, rest = do_connect_pairs(rest)
       ret.append(pt)
    if len(ret) == 1: return pt
    return ret

def do_find_circle_center(p1, p2, p3,  norm):
    dp1 = p2 - p1
    dp2 = p3 - p2
    mp1 = (p2 + p1)/2.0
    mp2 = (p3 + p2)/2.0
    v1 = np.cross(norm, dp1)
    v2 = np.cross(norm, dp2)
    m = np.transpose(np.vstack((v1, v2)))
    dmp = mp1 - mp2
    if np.linalg.det(m[[0,1],:]) != 0:
       a, b = np.dot(np.linalg.inv(m[[0,1],:]), dmp[[0,1]])
    elif np.linalg.det(m[[0,2],:]) != 0:
       a, b = np.dot(np.linalg.inv(m[[0,2],:]), dmp[[0,2]])
    elif np.linalg.det(m[[1,2],:]) != 0:
       a, b = np.dot(np.linalg.inv(m[[1,2],:]), dmp[[1,2]])
    else:
        print(p1, p2, p3)
        raise ValueError("three points does not span a surface")

    return mp1 - v1*a

def find_circle_center_radius(vv, norm):
    '''
    assuming that point are somewhat equally space, I will
    scatter three points around the circle given by input points

    this approch reduces the risk of having three points colinear,
    Note that we need to have sufficently large number of points...
  
    '''
    k = len(vv)-2
    ii = np.linspace(0, len(vv)-1, 4).astype(int)
    print(vv.shape, ii)
   
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

    idx = np.where(ll[:,0] > ll[:,1])[0]
    t1 = ll[idx,0]
    t2 = ll[idx,1]
    ll[idx,0] = t2
    ll[idx,1] = t1

    ii = np.vstack([np.arange(ll.shape[0]),]*2).transpose()
    d  = np.ones(ll.shape[0]*ll.shape[1]).reshape(ll.shape)
    from scipy.sparse import csr_matrix, coo_matrix
    m = coo_matrix((d.flatten(), (ii.flatten(), ll.flatten())), 
                   shape=(len(ll),np.max(ll+1)))
    mm = m.tocsc()
    ic = mm.indices; icp = mm.indptr
    mm = m.tocsr()
    ir = mm.indices; irp = mm.indptr

    def get_start(taken):
       idx = np.where(np.logical_and(np.diff(icp) == 1, taken == 0))[0]
       nz  = np.where(np.logical_and(np.diff(icp) != 0, taken == 0))[0]
       if len(nz) == 0: return
       if len(idx) > 0:
          #print('Open end found')
          pt = (ic[icp[idx[0]]], idx[0])
       else:
          pt = (ic[icp[nz[0]]], nz[0])
       pts = [pt]
       return pts

    def hop_v(pt):
       ii = pt[1]
       ii = [icp[ii],icp[ii+1]-1]
       next = ic[ii[1]] if ic[ii[0]] == pt[0] else ic[ii[0]]
       return (next, pt[1])
    def hop_h(pt):
       ii = pt[0]
       ii = [irp[ii],irp[ii+1]-1]
       next = ir[ii[1]] if ir[ii[0]] == pt[1] else ir[ii[0]]
       return (pt[0], next)

    def trace(pts):
        loop = [pts[-1][1]]
        while True:
            pts.append(hop_v(pts[-1]))
            #rows.append(pts[-1][0])
            pts.append(hop_h(pts[-1]))
            if pts[-1][1] in loop : break # open ended
            loop.append(pts[-1][1])
            if pts[-1] == pts[0]: break
        return loop

    taken = (icp*0)[:-1]
    loops = []
    while True:
        pts = get_start(taken)
        if pts is None: break
        loop = trace(pts)
        loops.append(loop)
        taken[loop] = 1

    return loops

