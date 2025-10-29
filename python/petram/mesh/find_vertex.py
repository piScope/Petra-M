from collections import defaultdict
import numpy as np

def find_vertex(mesh, bb_edges):
    

    corners = {}
    for key in bb_edges:
        iv = np.hstack([mesh.GetEdgeVertices(ie) for ie in bb_edges[key]])
        seen = defaultdict(int)
        for iiv in iv:
            seen[iiv] += 1
        corners[key] = [kk for kk in seen if seen[kk]==1]

    ivert = np.unique(np.hstack([corners[key]
                                 for key in corners])).astype(int, copy=False)
    return corners, ivert
        
