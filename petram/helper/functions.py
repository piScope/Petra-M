# this module import functions from numpy

import numpy as np
names = ['pi', 'sin', 'cos', 'exp', 'sqrt', 'log', 'arctan2', 'max', 'abs']

f = {n:getattr(np, n) for n in names}
