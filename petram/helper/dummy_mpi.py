from __future__ import print_function

class CommWorld(object):
    def __init__(self):
        self.size = 1
        self.rank = 0
        
    def Barrier(self):
        pass
    def bcast(self, *args, **kwargs):
        return args[0]

    def gather(self, *args, **kwargs):
        return [args[0]]
    
    def allgather(self, *args, **kwargs):
        return [args[0]]
    
class MPIclass(object):
    def __init__(self, *args, **kwargs):
        self.COMM_WORLD = CommWorld()
MPI = MPIclass()

def nicePrint(x):
    print(x)

        
