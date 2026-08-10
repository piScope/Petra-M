#
#  utlities for handling cutplane
#
class _XY(tuple):
    def __call__(self, value):
        return (0, 0, 1., -value)


class _YZ(tuple):
    def __call__(self, value):
        return (1, 0, 0., -value)


class _ZX(tuple):
    def __call__(self, value):
        return (0, 1., 0., -value)

def process_abcd(abcd_txt, ns, parent_widget=None):
    ll = {"YZ": _YZ((1, 0, 0., 0)),
          "XY": _XY((0., 0, 1., 0)),
          "ZX": _ZX((0., 1, 0., 0)),
          "yz": _YZ((1, 0, 0., 0)),
          "xy": _XY((0., 0, 1., 0)),
          "zx": _ZX((0., 1, 0., 0)), }
    # add all combinations
    ll["ZY"] = ll["YZ"]
    ll["zy"] = ll["YZ"]
    ll["YX"] = ll["XY"]
    ll["yx"] = ll["XY"]
    ll["XZ"] = ll["ZX"]
    ll["xz"] = ll["ZX"]

    abcd_value = eval(abcd_txt, ll, ns)

    lens = [1, 1, 1, 1]
    for i in range(4):
        try:
            lens[i] = len(abcd_value[i])
        except TypeError:
            pass
    num_planes = max(lens)
    planes = [None]*4

    for i in range(4):
        if lens[i] == 1:
            planes[i] = [abcd_value[i]]*num_planes
        elif lens[i] == num_planes:
            planes[i] = abcd_value[i]
        else:
            if parent_widget is not None:
                import wx
                import ifigure.widgets.dialog as dialog
                msg = 'a, b, c, d has to have the same lenght when multiple planes are defined'
                wx.CallAfter(
                    dialog.showtraceback,
                    parent=parent_widget,
                    txt='Can not determin planes',
                    title='Error',
                    traceback=msg)
            return None
    return planes
