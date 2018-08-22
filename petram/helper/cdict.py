'''
   CDict is a dictionary allowing to access items using class-like 
   syntax
'''
class tmp(object):
    pass

err_msg = 'cdict member name should be string, not starts from "_"'

class CDict(dict):
    def __getattr__(self, attr):
      # only attributes not starting with "_" are organinzed
      # in the tree
       if hasattr(self, attr):
           return dict.__getattr__(self, attr)
       if not attr.startswith("_"):
           try:
               return self.__getitem__(attr)
           except:
               raise
       raise AttributeError(err_msg)

    def __setattr__(self, attr, val):
       if hasattr(self, attr):
           return dict.__setattr__(self, attr, val)           
       if isinstance(attr, str):
           if not attr.startswith("_"):
               return self.__setitem__(attr, val)
       raise AttributeError(err_msg)           

    def __setitem__(self, key, value):
        ## check if key can be accessed as a class attribute
        try:
            if key.startswith('_'):
                 raise AttributeError(err_msg)
        except:
            raise AttributeError(err_msg)            
        o = tmp()
        try:
            exec("o."+key+"=3")
        except:
            raise AttributeError(err_msg)                                   
        dict.__setitem__(self, key, value)
    

