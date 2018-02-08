from petram.helper.variables import variable

ph = 0.0

@variable.float
def test(x, y, z):
    return y*10+3
