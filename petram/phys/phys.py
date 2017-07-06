class Phys(object):
    def __init__(self, mesh = None, domain = None):
        self.mesh = mesh
        self.domain = domain if domain is not None else []
        self.vol = []
        self.face = []
        self.edge = []
        self.point = []
    def __del__(self):
        self.vol = None
        self.face = None
        self.edge = None
        self.point = None
        
    def add_vol(self, vol):
        self.vol.append(vol)
        
    def add_face(self, face):
        self.face.append(face)

    def add_edge(self, edge):
        self.edge.append(edge)

    def add_point(self, point):
        self.point.append(point)

                          
                                       
                         
        
