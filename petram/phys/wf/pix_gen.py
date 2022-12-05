from __future__ import print_function
from petram.phys.weakform import get_integrators
#
#  a script to produce icon images
#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.transforms import Bbox
import base64
import os

import wx

rc("text", usetex=True)

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
    dpi2 = dpi*8

    F = plt.figure(num=None, figsize=(3.5, 0.3), dpi=dpi, facecolor='w')
    x = [0, 1, 1, 0, 0]
    y = [1, 1, 0, 0, 1]

    for name, _domain, _range, _coeff, _dim, wf_form, strong_form in data:
        ax = plt.subplot(111)
        ax.cla()
        ax.tick_params(length=0)
        ax.set_axis_off()

        txt1 = correct_latex(wf_form)
        if len(strong_form.strip()) > 0:
            txt2 = correct_latex(strong_form)
            #txt2 = "$\\left[\\equiv "+ txt2[1:-1] + "\\right]$"
            txt2 = "$[\\approx " + txt2[1:-1] + "]$"
            #txt2 = "$\\approxeq "+ txt2[1:-1] + "$"
        else:
            txt2 = ""
        print("text", txt1, txt2)

        plt.text(0.01, 0.3, txt1)

        if txt2 != "":
            plt.text(0.51, 0.3, txt2)

        filename = os.path.join(save_path, header + name + '.png')
        print('filename', filename)
        ed = ax.transAxes.transform([(0, 0), (1, 1)])
        bbox = Bbox.from_extents(
            ed[0, 0]/dpi, ed[0, 1]/dpi, ed[1, 0]/dpi, ed[1, 1]/dpi)
        plt.savefig(filename, dpi=dpi2, format='png', bbox_inches=bbox)

    ax = plt.subplot(111)
    ax.cla()
    ax.tick_params(length=0)
    ax.set_axis_off()

    filename = os.path.join(save_path, header + 'none.png')
    print('filename', filename)
    ed = ax.transAxes.transform([(0, 0), (1, 1)])
    bbox = Bbox.from_extents(
        ed[0, 0]/dpi, ed[0, 1]/dpi, ed[1, 0]/dpi, ed[1, 1]/dpi)
    plt.savefig(filename, dpi=dpi2, format='png', bbox_inches=bbox)

    ##################################################


if __name__ == '__main__':
    generate_pix(bilinintegs, 'form_')
    generate_pix(linintegs, 'form_')
