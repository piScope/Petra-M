from __future__ import print_function
from petram.phys.weakform import get_integrators
#
#  a script to produce icon images
#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import base64
import os

import wx

b64encode = base64.urlsafe_b64encode


def encode(txt):
    return (b64encode(txt.encode())).decode()


# names, domains, ranges, coeffs, dims, wf_forms, strong_forms))
bilinintegs = get_integrators('BilinearOps', return_all=True)
linintegs = get_integrators('LinearOps', return_all=True)


def correct_latex(txt):
    txt = txt.strip()
    txt = txt.replace('\\grad', '\\nabla')
    txt = txt.replace('\\{', '{')
    txt = txt.replace('\\cross', '\\times')
    txt = txt.replace('\\curl', '\\nabla\\times')
    txt = txt.replace('\\div', '\\nabla\\cdot')
    txt = txt.replace('\\ddx', '\\frac{d}{dx}')
    return txt

##################################################


def generate_pix(data, header):

    save_path = os.path.join(os.path.dirname(
        __file__), '..', '..', 'data', 'icon')

    dpi = 72
    dpi2 = dpi*4

    F = plt.figure(num=None, figsize=(3.5, 0.3), dpi=dpi, facecolor='w')
    x = [0, 1, 1, 0, 0]
    y = [1, 1, 0, 0, 1]

    for name, _domain, _range, _coeff, _dim, wf_form, strong_form in data:
        ax = plt.subplot(111)
        ax.cla()
        ax.tick_params(length=0)
        ax.set_axis_off()
        txt = correct_latex(wf_form)

        print("text", wf_form, txt)
        plt.text(0.01, 0.3, txt)

        if len(strong_form.strip()) > 0:
            txt = "[ $\\approxeq$ " + correct_latex(strong_form) + " ]"

            plt.text(0.51, 0.3, txt)

        filename = os.path.join(save_path, header + name + '.png')
        print('filename', filename)
        ed = ax.transAxes.transform([(0, 0), (1, 1)])
        bbox = Bbox.from_extents(
            ed[0, 0]/dpi, ed[0, 1]/dpi, ed[1, 0]/dpi, ed[1, 1]/dpi)
        plt.savefig(filename, dpi=dpi2, format='png', bbox_inches=bbox)
    ##################################################


if __name__ == '__main__':
    generate_pix(bilinintegs, 'form_')
    generate_pix(linintegs, 'form_')
