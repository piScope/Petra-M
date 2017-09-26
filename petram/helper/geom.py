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
