#
# this file is not used anymore...
#
from petram.helper.variables import var_g

g = var_g.copy()

def eval_value(value):
    return eval(value, g, {})    

def validator(value, param, ctrl):
    try:
       x = eval_value(value)
    except:
       return False
    return True
