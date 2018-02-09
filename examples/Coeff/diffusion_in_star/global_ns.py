import numpy as np
r = [1, 0.5, 1., 0.5, 1, 0.5, 1, 0.5, 1., 0.5]
star_x = np.array([r[i]*cos(i/10.*pi*2) for i in range(10)])
star_y = np.array([r[i]*sin(i/10.*pi*2) for i in range(10)])
star_z = np.array([0]*10)

from petram.helper.variables import variable
@variable.array(complex = False, shape = (2,))
def c_diff(x, y):
    
    c = x/np.sqrt(x**2 + y**2)
    s = -y/np.sqrt(x**2 + y**2)

    a = 1.0
    b = 100.0
    return [[a*c*c + b*s*s, (b-a)*c*s], 
            [(b-a)*c*s, a*s*s + b*c*c]]