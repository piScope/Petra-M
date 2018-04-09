class CommWorld(object):
    def __init__(self):
        self.rank = 1
        self.myid = 0
        
    def Barrier(self):
        pass
    def bcast(self, *args, **kwargs):
        pass
    
class MPIclass(object):
    def __init__(self, *args, **kwargs):
        self.COMM_WORLD = CommWorld()
MPI = MPIclass()        

        
