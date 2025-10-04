'''
  DotDict 
    Dictionary container, which allows for dictionary look up by dot

'''
class DotDict(object):
    def __init__(self, d):
        self._d = d

    def __getattr__(self, attr):
      # only attributes not starting with "_" are organinzed
      # in the tree
        if not attr.startswith("_"):
            if attr in self._d:
                return self._d[attr]
            raise AttributeError('Attribute member not found :'+attr+':'+str(self))
            # return self._d.setdefault(attr, TreeDict())
        raise AttributeError('Attribute name should not start from "_"')
        
