def multi(arr, target):
    if target.ndim - arr.ndim > 0:
       arr = arr.reshape(arr.shape + (1,) * (target.ndim - arr.ndim))
    elif target.ndim - arr.ndim < 0:
       target = target.reshape(target.shape + (1,) * (-target.ndim + arr.ndim))
    return arr*target

def div(arr, target):
    if target.ndim - arr.ndim > 0:
       arr = arr.reshape(arr.shape + (1,) * (target.ndim - arr.ndim))
    elif target.ndim - arr.ndim < 0:
       target = target.reshape(target.shape + (1,) * (-target.ndim + arr.ndim))
    return arr/target

def add(arr, target):
    if target.ndim - arr.ndim > 0:
       arr = arr.reshape(arr.shape + (1,) * (target.ndim - arr.ndim))
    elif target.ndim - arr.ndim < 0:
       target = target.reshape(target.shape + (1,) * (-target.ndim + arr.ndim))
    return arr+target


