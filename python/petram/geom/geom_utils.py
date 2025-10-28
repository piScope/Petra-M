from collections import defaultdict

import numpy as np
from scipy.linalg import null_space, lstsq


def rotation_mat(ax, an):
    '''
    matrix to rotate arounc ax by an [rad]
    '''
    c = np.cos(an)
    s = np.sin(an)
    ax = ax / np.sqrt(np.sum(ax**2))
    R = np.array([[c + (1 - c) * ax[0]**2,
                   ax[0] * ax[1] * (1 - c) - ax[2] * s,
                   ax[0] * ax[2] * (1 - c) + ax[1] * s],
                  [ax[0] * ax[1] * (1 - c) + ax[2] * s,
                   c + (1 - c) * ax[1]**2,
                   ax[1] * ax[2] * (1 - c) - ax[0] * s],
                  [ax[0] * ax[2] * (1 - c) - ax[1] * s,
                   ax[1] * ax[2] * (1 - c) + ax[0] * s,
                   c + (1 - c) * ax[2]**2]])

    return R


def normal2points(p1, eps=1e-13):
    '''
    normal vector defined by a surface made from the group of
    points

    p1 : [#, 3] matrix, where # is the number of points
    '''
    p1 = np.hstack([p1, np.atleast_2d(np.array([1] * len(p1))).transpose()])
    u, s, vh = np.linalg.svd(p1)

    null_mask = (s <= eps)
    if sum(null_mask) == 0:
        print("no null space??", p1, s)
    null_space = np.compress(null_mask, vh, axis=0)
    norm = null_space[0, :3]
    norm = norm / np.sqrt(np.sum(norm**2))
    return norm


def map_points_in_geom_info(info1, info2, th=1e-10):
    '''
    info = ptx, p, l, s, v
        pts = array(:, 3)
        p   = point -> point index
        l   = line -> point
        s   = surface -> line
        v   = volume -> surface

    We restricts points to those involved in lines
    '''
    ptx1 = info1[0]
    ptx2 = info2[0]

    # reduce the size of ptx1 to search for the points wihch is relavent
    tmp = list(info1[1])
    ptx1 = np.vstack([ptx1[info1[1][k], :] for k in tmp])
    info11 = {k: i for i, k in enumerate(tmp)}

    # print(info11)
    # print(info2)

    # iverts -> p
    #
    iv2p1 = {info11[k]: k for k in info11}
    iv2p2 = {info2[1][k]: k for k in info2[1]}

    dist = np.array([np.min(np.sum((ptx1 - ptx2[k, :])**2, 1))for k in iv2p2])
    if np.any(dist > th):
        print(dist)
        print("ptx1", ptx1)
        print("ptx2", ptx2)
        assert False, "could not able to find vertex mapping"

    # {point in info2 : point in info1}
    pmap_r = {iv2p2[k]: iv2p1[np.argmin(np.sum((ptx1 - ptx2[k, :])**2, 1))]
              for k in iv2p2}
    # {point in info1 : point in info2}
    pmap = {pmap_r[k]: k for k in pmap_r}

    #print(pmap, pmap_r)
    #print("src", np.vstack([ptx1[info11[k],:] for k in pmap]))
    #print("det", np.vstack([ptx2[info2[1][k],:] for k in pmap_r]))

    return pmap, pmap_r


def dist(x, y, trans=None):
    if trans is not None:
        x = trans(x)
    d = np.sqrt(np.sum((x - y)**2))
    return d


def map_lines_in_geom_info(info1, info2, pmap_r, th=1e-10, trans=None):
    lmap = {}
    lmap_r = {}
    #print("search for lines for ", info1[2], info2[2], pmap_r)
    for l in info2[2]:
        p1, p2 = pmap_r[info2[2][l][0]], pmap_r[info2[2][l][1]]
        for x in info1[2]:
            #print("checking ", x, " for ", l)
            if (info1[2][x][0] == p1 and
                    info1[2][x][1] == p2):
                if dist(info1[5][x], info2[5][l], trans) > th:
                    print("rejectedy by dist(1)", dist(
                        info1[5][x], info2[5][l], trans))
                    continue
                lmap[x] = l
                lmap_r[l] = x
                break
            elif (info1[2][x][0] == p2 and
                  info1[2][x][1] == p1):
                if dist(info1[5][x], info2[5][l], trans) > th:
                    print("rejectedy by dist(2)", dist(
                        info1[5][x], info2[5][l], trans))
                    continue
                lmap[x] = -l
                lmap_r[l] = -x
                break
            else:
                pass
        else:
            print("current data", lmap_r)
            assert False, "could not find line mapping for " + str(l)
    return lmap, lmap_r


def map_surfaces_in_geom_info(info1, info2, lmap_r):
    smap = {}
    smap_r = {}

    #print("search for surfaces for ", info1[3], info2[3], lmap_r)
    for s in info2[3]:
        tmp = sorted([abs(lmap_r[x]) for x in info2[3][s]])
        for x in info1[3]:
            if sorted(info1[3][x]) == tmp:
                smap[x] = s
                smap_r[s] = x
                break
            else:
                pass
        else:
            assert False, "could not find surface mapping for " + str(s)

    return smap, smap_r


def map_volumes_in_geom_info(info1, info2, smap_r):
    vmap = {}
    vmap_r = {}

    for v in info2[4]:
        tmp = sorted([smap_r[x] for x in info2[4][v]])
        for x in info1[4]:
            if sorted(info1[4][x]) == tmp:
                vmap[x] = v
                vmap_r[v] = x
                break
            else:
                pass
        else:
            assert False, "could not find volume mapping for " + str(s)
    return vmap, vmap_r


def find_s_pairs(src, dst, s, l_pairs):
    s_pairs = {}
    for s1 in src:
        l2 = set([abs(l_pairs[l1]) for l1 in s[s1]])
        for s2 in dst:
            if set(s[s2]) == l2:
                s_pairs[s1] = s2
                break

    return s_pairs


def find_normal_from_edgedata(edge_tss, lines):
    ii = np.isin(edge_tss[2], lines)
    ptx = edge_tss[0][np.unique(edge_tss[1][ii])]
    n1 = normal2points(ptx)
    #print("normal based on edge tss", n1)
    return n1


def find_translate_between_surface(src, dst, edge_tss, geom=None,
                                   geom_data=None,
                                   min_angle=0.1,
                                   mind_eps=1e-10,
                                   axan=None):

    ptx, p, l, s, v, mid_points = geom_data
    s2l = s

    l1 = np.unique(np.hstack([s[k] for k in src]).flatten())
    l2 = np.unique(np.hstack([s[k] for k in dst]).flatten())
    p1p = np.unique(np.hstack([l[k] for k in l1]).flatten())
    p2p = np.unique(np.hstack([l[k] for k in l2]).flatten())

    i1 = p1p - 1
    i2 = p2p - 1

    p1 = ptx[i1, :]
    p2 = ptx[i2, :]

    if axan is None:
        n1 = find_normal_from_edgedata(edge_tss, l1)
        n2 = find_normal_from_edgedata(edge_tss, l2)

        ax = n1
        an = 0.0

        '''
        ax = ax/np.linalg.norm(ax)
        n3 = np.cross(ax, n1)
        xx = np.sum(n2*n1)
        yy = np.sum(n2*n3)
        #an = np.arcsin(np.linalg.norm(ax))
        an = np.arctan2(yy, xx)
        #print("p2, axis angle", xx, yy, p2, ax, an)
        '''
    else:
        ax, an = axan
        ax = np.array(ax, dtype=float)
        ax = ax / np.linalg.norm(ax)
        an = 0.0

    def find_mapping(ax, an, p1, p2):
        if an != 0.0:
            R = rotation_mat(ax, -an)
        else:
            R = np.diag([1, 1, 1.])

    # check two possible orientation
        p3 = np.dot(R, p2.transpose()).transpose()

        # try all transpose
        for i in range(len(p1)):
            d = p3[0] - p1[i]
            p3t = p3 - d
            mind = np.array(
                [np.min(np.sqrt(np.sum((p3t - pp)**2, 1))) for pp in p1])
            if np.all(mind < mind_eps):
                mapping = [
                    np.argmin(
                        np.sqrt(
                            np.sum(
                                (p3t - pp)**2,
                                1))) for pp in p1]
                trans = d
                return d, mapping, R
        return None, None, None

    if abs(an * 180. / np.pi) < min_angle:
        an = 0.0
    if abs(abs(an * 180. / np.pi) - 180.) < min_angle:
        an = 0.0

    d, mapping, R = find_mapping(ax, an, p1, p2)

    if d is None:
        if an > 0.:
            an = -np.pi + an
        else:
            an = np.pi + an
        d, mapping, R = find_mapping(ax, an, p1, p2)
        if d is None:
            assert False, "auto trans failed (no mapping between vertices)"

    p_pairs = dict(zip(p1p, p2p[mapping]))  # point mapping

    affine = np.zeros((4, 4), dtype=float)
    affine[:3, :3] = np.linalg.inv(R)
    affine[:3, -1] = np.dot(np.linalg.inv(R), d)
    affine[-1, -1] = 1.0

    R = affine[:3, :3]
    D = affine[:3, -1]

    def translate(pos):
        pos = (np.dot(R, pos).transpose() + D)
        return pos

    #print("l1", [(ll, l[ll], translate(mid_points[ll])) for ll in l1])
    #print("l2", [(ll, l[ll], mid_points[ll]) for ll in l2])

    #
    #  look for pairs of lines using edge vertex and midpoints
    #
    l_pairs = {}
    for ll in l1:
        target = tuple(sorted((p_pairs[l[ll][0]], p_pairs[l[ll][1]])))
        m1 = translate(mid_points[ll])
        for ll2 in l2:
            m2 = mid_points[ll2]
            dist = np.sqrt(np.sum((m1 - m2)**2))
            if tuple(sorted(l[ll2])) == target and dist < mind_eps:
                l_pairs[ll] = ll2
                break

#    l2dict = {tuple(sorted(l[ll])): ll for ll in l2}
#    l_pairs = {
#        ll: l2dict[tuple(sorted((p_pairs[l[ll][0]], p_pairs[l[ll][1]])))] for ll in l1}

    px = np.dot(np.linalg.pinv(-R + np.diag((1, 1, 1))), -d)

    s_pairs = find_s_pairs(src, dst, s2l, l_pairs)
    #print("px, d", px, d)
    return ax, an, px, d, affine, p_pairs, l_pairs, s_pairs


def find_rotation_between_surface(src, dst, edge_tss, geom=None,
                                  geom_data=None,
                                  min_angle=0.1,
                                  mind_eps=1e-10,
                                  axan=None):

    if geom is not None:
        ptx, cells, cell_data, l, s, v, geom = geom._gmsh4_data
    else:
        cell_data = None
        ptx, p, l, s, v, mid_points = geom_data
    s2l = s

    l1 = np.unique(np.hstack([s[k] for k in src]).flatten())
    l2 = np.unique(np.hstack([s[k] for k in dst]).flatten())
    p1p = np.unique(np.hstack([l[k] for k in l1]).flatten())
    p2p = np.unique(np.hstack([l[k] for k in l2]).flatten())

    if cell_data is None:
        i1 = p1p - 1
        i2 = p2p - 1
    else:
        i1 = np.array([np.where(cell_data['vertex']['geometrical'] == ii)[
                      0] for ii in p1p]).flatten()
        i2 = np.array([np.where(cell_data['vertex']['geometrical'] == ii)[
                      0] for ii in p2p]).flatten()
    p1 = ptx[i1, :]
    p2 = ptx[i2, :]
    #n1 = normal2points(p1)
    #n2 = normal2points(p2)
    n1 = find_normal_from_edgedata(edge_tss, l1)
    n2 = find_normal_from_edgedata(edge_tss, l2)

    #print(p1, p2, n1, n2, axan)
    if axan is None:
        c = np.sum(n1 * n2)
        s = np.sqrt(np.sum(np.cross(n1, n2)**2))
        an = np.arctan2(s, c)

        # we assume angle is less than 90 deg.
        if an > np.pi / 2.0:
            an = an - np.pi
        if an < -np.pi / 2.0:
            an = an - np.pi

        M = np.vstack((n1, n2))
        b = np.array([np.sum(n1 * p1[0]), np.sum(n2 * p2[0])])

        ax = null_space(M).flatten()
        px, res, rank, ss = lstsq(M, b, rcond=None)

        print("p2, axis angle", px, ax, an)

    else:
        ax, an = axan
        ax = np.array(ax, dtype=float)
        ax = ax / np.linalg.norm(ax)
        an = np.pi / 180. * an
        px = np.array([0, 0, 0])

    def find_mapping(px, ax, an, p1, p2):
        if an != 0.0:
            R = rotation_mat(ax, -an)
        else:
            R = np.diag([1, 1, 1.])

        # check two possible orientation
        p3 = np.dot(R, (p2 - px).transpose()).transpose() + px

        # try all transpose
        #print("p1, p2, p3 (1)", p1, p2, p3)
        for i in range(len(p1)):
            d = p3[0] - p1[i]
            p3t = p3 - d
            mind = np.array(
                [np.min(np.sqrt(np.sum((p3t - pp)**2, 1))) for pp in p1])
            if np.all(mind < mind_eps):
                mapping = [
                    np.argmin(
                        np.sqrt(
                            np.sum(
                                (p3t - pp)**2,
                                1))) for pp in p1]
                trans = d
                return d, mapping, R
        return None, None, None

    if abs(an * 180. / np.pi) < min_angle:
        an = 0.0
    if abs(abs(an * 180. / np.pi) - 180.) < min_angle:
        an = 0.0
    print('trying', an)
    d, mapping, R = find_mapping(px, ax, an, p1, p2)

    if d is None:
        if an > 0.:
            an2 = -np.pi + an
        else:
            an2 = np.pi + an
        print('trying', an2)
        d, mapping, R = find_mapping(px, ax, an2, p1, p2)
        if d is None:
            an2 = -an

            print('trying', an2)
            d, mapping, R = find_mapping(px, ax, an2, p1, p2)
            if d is None:
                if an > 0.:
                    an2 = -np.pi - an
                else:
                    an2 = np.pi - an
                print('trying', an2)
                d, mapping, R = find_mapping(px, ax, an2, p1, p2)
        an = an2
    if d is None:
        assert False, "auto trans failed (no mapping between vertices)"

    p_pairs = dict(zip(p1p, p2p[mapping]))  # point mapping

    #print("l1", [(ll, l[ll]) for ll in l1])
    #print("l2", [(ll, l[ll]) for ll in l2])
    l2dict = {tuple(sorted(l[ll])): ll for ll in l2}

    l_pairs = {
        ll: l2dict[tuple(sorted((p_pairs[l[ll][0]], p_pairs[l[ll][1]])))] for ll in l1}

    affine = np.zeros((4, 4), dtype=float)
    affine[:3, :3] = np.linalg.inv(R)
    affine[:3, -1] = np.dot(np.linalg.inv(R), d)
    affine[-1, -1] = 1.0

    # if axan is None:
    #    px = np.dot(np.linalg.pinv(-R+np.diag((1,1,1))),-d)
    print("ax, an, px", ax, an, px)

    s_pairs = find_s_pairs(src, dst, s2l, l_pairs)

    return ax, an, px, d, affine, p_pairs, l_pairs, s_pairs


def find_rotation_between_surface2(src, dst, vol, edge_tss,
                                   geom_data=None,
                                   min_angle=0.1,
                                   mind_eps=1e-10,
                                   axan=None):
    '''
    find a rotational transform from src to dest

    based on finding a volume chain between src and dest
    volume is a hint to specify the volumes in the chains
    '''
    ptx, p, l, s, v, mid_points = geom_data
    s2l = s

    l1 = np.unique(np.hstack([s[k] for k in src]).flatten())
    l2 = np.unique(np.hstack([s[k] for k in dst]).flatten())
    p1p = np.unique(np.hstack([l[k] for k in l1]).flatten())
    p2p = np.unique(np.hstack([l[k] for k in l2]).flatten())

    # first find angle using surface normal
    i1 = p1p - 1
    i2 = p2p - 1

    p1 = ptx[i1, :]
    p2 = ptx[i2, :]

    n1 = find_normal_from_edgedata(edge_tss, l1)
    n2 = find_normal_from_edgedata(edge_tss, l2)
    #n1 = normal2points(p1)
    #n2 = normal2points(p2)

    cos = np.sum(n1 * n2)
    sin = np.sqrt(np.sum(np.cross(n1, n2)**2))
    an = np.arctan2(sin, cos)
    ax = np.cross(n1, n2)
    ax = ax / np.sqrt(np.sum(ax**2))

    all_s = sum([v[k] for k in vol], [])
    all_l = sum([s[k] for k in all_s], [])
    lateral_l = np.setdiff1d(all_l, np.hstack((l1, l2)))

    # next find volume chain
    def search_volume_chain(start_volume, start_face):
        sf = start_face
        sv = start_volume

        ret = ([sv], [sf], [])
        while(True):
            if sv not in vol:
                return False, None
            front_e = s[sf]   # front edge
            faces = set(v[sv]).difference([sf])
            # opposit face
            f2 = [f for f in faces if len(set(s[f]).intersection(l1)) == 0]

            lateral_edges = set(sum([s[f] for f in v[sv]], [])).difference(
                s[f2[0]] + s[sf])

            if len(f2) != 1:
                assert False, "opposite face must be one"

            if f2[0] in dst:
                ret[2].append(list(lateral_edges))
                ret[1].append(f2[0])
                break

            next_volumes = [k for k in v if len(
                set(v[k]).intersection(f2)) != 0]
            if len(next_volumes) != 2:
                return False, None
                # no more search

            sv = next_volumes[0] if next_volumes[1] == sv else next_volumes[1]
            sf = f2[0]
            ret[0].append(sv)
            ret[1].append(sf)
            ret[2].append(list(lateral_edges))
        return True, ret

    chains = {}

    for sf in src:
        start_volumes = [k for k in v if len(
            set(v[k]).intersection([sf])) != 0]

        found = False
        for sv in start_volumes:
            if sv not in vol:
                continue
            # starting from sf, build volume and face chain.
            success, ret = search_volume_chain(sv, sf)
            found = found or success
            if success:
                chains[sf] = ret[2]
        else:
            if not found:
                assert False, "can not find the body chain to destination from " + \
                    str(sf)

    p_map = {x: None for x in p1p}
    p_done = []

    for sf in src:
        for lg in chains[sf]:
            for e in lg:
                p1, p2 = l[e]
                if p1 in p_done or p2 in p_done:
                    continue
                if p1 in p_map:
                    p_map[p1] = p2
                    p_map[p2] = None
                    p_done.append(p1)
                elif p2 in p_map:
                    p_map[p2] = p1
                    p_map[p1] = None
                    p_done.append(p2)
                else:
                    pass
    p_pairs = {}
    for p1 in p1p:
        next = p1
        while True:
            if p_map[next] is None:
                p_pairs[p1] = next
                break
            next = p_map[next]

    def check_distance(p_pairs, R, ptx):
        ddd = []
        for i, k in enumerate(p_pairs):
            #print("checking points", k, p_pairs[k])
            p1 = ptx[k - 1, :]
            p2 = ptx[p_pairs[k] - 1, :]
            ddd.append(np.sqrt(np.sum((p2 - np.dot(R, p1))**2)))
        # print("average distance (must be less than)", np.max(np.abs(ddd-np.mean(ddd))),
        #      mind_eps)
        return np.max(np.abs(ddd - np.mean(ddd))) < mind_eps

    def rot_center(p1, p2, an, ax):
        mid = (p1 + p2) / 2.0

        dd = np.sqrt(np.sum((p1 - mid)**2))
        xx = np.cross(p1 - p2, ax)
        xx = xx / np.sqrt(np.sum(xx**2))
        return mid + xx * dd / \
            np.tan(an / 2.0), mid - xx * dd / np.tan(an / 2.)

    def check_good(p1, p2, an):
        # print(np.sqrt(np.sum((p1-p2)**2)))
        if np.sqrt(np.sum((p1 - p2)**2)) < mind_eps:
            return True
        n1 = (p1 - p2) / np.sqrt(np.sum((p1 - p2)**2))
        return np.abs(np.abs(np.sum(ax * n1)) - 1) < mind_eps

    # using angle and normal we can find the center...

    def try_ax_an(ax, an):
        R = rotation_mat(ax, an)
        R_half = rotation_mat(ax, an / 2.0)
        if not check_distance(p_pairs, R, ptx):
            return False, False, False, False

        good = [True, True]
        d = np.array([0, 0, 0])
        for i, k in enumerate(p_pairs):
            #print("checking points", k, p_pairs[k])
            p1 = ptx[k - 1, :]
            p2 = ptx[p_pairs[k] - 1, :]
            d = d + (p2 - np.dot(R, p1))

            c1, c2 = rot_center(p1, p2, an, ax)
            if i == 0:
                px1 = c1
                px2 = c2
            else:
                if not check_good(px1, c1, an) and not check_good(px1, c2, an):
                    good[0] = False
                if not check_good(px2, c1, an) and not check_good(px2, c2, an):
                    good[1] = False
        px = c1 if good[0] else c2
        d = d / len(p_pairs)

        # test mid-point of lateral edge is on the half angle location.
        pt1, pt2 = l[lateral_l[0]]

        if pt1 in p_pairs:
            check = dist(mid_points[lateral_l[0]],
                         R_half.dot(ptx[p[pt1]] - px) + px)
        else:
            check = dist(mid_points[lateral_l[0]],
                         R_half.dot(ptx[p[pt2]] - px) + px)

        #print("mid point check", check)
        if check > mind_eps:
            return False, False, False, False

        return R, px, d, check

    R, px, d, check = try_ax_an(ax, an)
    if not check:
        if an > 0.0:
            an = an - np.pi
        else:
            an = an + np.pi
        R, px, d, check = try_ax_an(ax, an)
        if not check:
            assert False, "can not find center"

    print("ax, an, px, d", ax, an, px, d)

    l2dict = {tuple(sorted(l[ll])): ll for ll in l2}

    l_pairs = {
        ll: l2dict[tuple(sorted((p_pairs[l[ll][0]], p_pairs[l[ll][1]])))] for ll in l1}

    pmap_r = {p_pairs[x]: x for x in p_pairs}

    def translation(x):
        return d + np.dot(R, x)
    info1 = list(geom_data)
    info2 = list(geom_data)
    info1[2] = {x: geom_data[2][x] for x in l1}
    info2[2] = {x: geom_data[2][x] for x in l2}
    l_pairs, l_pairs_r = map_lines_in_geom_info(info1, info2, pmap_r, th=mind_eps,
                                                trans=translation)

    affine = np.zeros((4, 4), dtype=float)
    affine[:3, :3] = R
    affine[:3, -1] = d
    affine[-1, -1] = 1.0

    s_pairs = find_s_pairs(src, dst, s2l, l_pairs)

    return ax, an, px, d, affine, p_pairs, l_pairs, s_pairs
