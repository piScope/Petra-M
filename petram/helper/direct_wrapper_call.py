'''

 Direct Wrappwer Call provides the access to low level (PyMFEM)
 functionality during a Petra-M simulation

'''
class DWC(object):
    def __init__(self):
        pass

    def postprocess(self, mfem, *args):
        ''' 
        postprocess is called from solvestep after store_sol
        '''
        raise NotImplementedError("postprocess must be implemented by a user")
