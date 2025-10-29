'''

   gmsh_primitive

   This file define GUI interface. See gmsh_geom_wrapper for actual
   geometry generation routine

'''
import numpy as np
from collections import defaultdict

from petram.geom.gmsh_config import has_gmsh
from petram.geom.vtable_geom import *
from petram.geom.gmsh_geom_model import use_gmsh_api
from petram.geom.gmsh_geom_model import get_geom_key
from petram.geom.gmsh_geom_model import GmshPrimitiveBase as GeomPB
from petram.phys.vtable import VtableElement, Vtable


import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('GmshPrimitives')


invalid_pdata = (('NotUsedValue', VtableElement('NotUsedValue', type='array',
                                                guilabel='not_implemented',
                                                default='0.0',
                                                tip="This panel is not implemented")),)
pdata = (('xarr', VtableElement('xarr', type='array',
                                guilabel='X',
                                default='0.0',
                                tip="X")),
         ('yarr', VtableElement('yarr', type='array',
                                guilabel='Y',
                                default='0.0',
                                tip="Y")),
         ('zarr', VtableElement('zarr', type='array',
                                guilabel='Z',
                                default='0.0',
                                tip="Z")),)


def get_numbers(objs, targets):
    return [objs[t] if t in objs else int(t) for t in targets]


class Point(GeomPB):
    vt = Vtable(pdata)

    @classmethod
    def fancy_menu_name(self):
        return 'Point'

    @classmethod
    def fancy_tree_name(self):
        return 'Point'


class PointOCC(Point):
    vt = Vtable(pdata)

    @classmethod
    def fancy_menu_name(self):
        return 'by XYZ'

    @classmethod
    def fancy_tree_name(self):
        return 'Point'


pdata = (('surf', VtableElement('surf', type='string',
                                guilabel='Surface',
                                default="",
                                tip="surfaces")),
         ('u_coord', VtableElement('uarr', type='array',
                                   guilabel='U',
                                   default='0.',
                                   tip="U coords")),
         ('v_coord', VtableElement('varr', type='array',
                                   guilabel='V',
                                   default='0.',
                                   tip="V coords.")),)


class PointByUV(GeomPB):
    vt = Vtable(pdata)

    @classmethod
    def fancy_menu_name(self):
        return 'on Surface'

    @classmethod
    def fancy_tree_name(self):
        return 'PointOnSurface'


pdata = (('edge', VtableElement('edge', type='string',
                                guilabel='Line',
                                default="",
                                tip="Edge to put point")),
         ('u_coord', VtableElement('uarr', type='array',
                                   guilabel='U_n',
                                   default='0.',
                                   tip="U coords")),)


class PointOnEdge(GeomPB):
    vt = Vtable(pdata)

    @classmethod
    def fancy_menu_name(self):
        return 'on Line'

    @classmethod
    def fancy_tree_name(self):
        return "Point"


pdata = (('edge', VtableElement('edge', type='string',
                                guilabel='Line(conic)',
                                default="",
                                tip="Edge to find center")),
         ('use_focus', VtableElement('use_focus', type='bool',
                                     guilabel='Use focus (ellipse/hyperbola/parabola)',
                                     default=False,
                                     tip="Plcae point on focus (ellipse/hyperbola/parabola) ")), )


class PointCircleCenter(GeomPB):
    vt = Vtable(pdata)

    @classmethod
    def fancy_menu_name(self):
        return 'on Center/Focus of Conic'

    @classmethod
    def fancy_tree_name(self):
        return "CharacteristicPoint"


pdata = (('points1', VtableElement('points1', type='string',
                                   guilabel='1st Points',
                                   default="",
                                   tip="1st points (must be the same lenght as 2nd points)")),
         ('points2', VtableElement('points2', type='string',
                                   guilabel='2nd Points',
                                   default="",
                                   tip="1st points (must be the same lenght as 1srt points)")),)


class PointCenter(GeomPB):
    vt = Vtable(pdata)

    @classmethod
    def fancy_menu_name(self):
        return 'Center of 2 Points'

    @classmethod
    def fancy_tree_name(self):
        return "Point"


pdata = (('cpoint', VtableElement('cpoint', type='string',
                                  guilabel='Center point',
                                  default="",
                                  tip="center of circle")),
         ('point', VtableElement('points', type='string',
                                 guilabel='Point on circle',
                                 default="",
                                 tip="point on circle")),
         ('fill_circle', VtableElement('fill_circle', type='bool',
                                       guilabel='Fill circle',
                                       default=True,
                                       tip="Make surface ")), )


class Circle2DCenterOnePoint(GeomPB):
    vt = Vtable(pdata)

    @classmethod
    def fancy_menu_name(self):
        return 'Center and Point'

    @classmethod
    def fancy_tree_name(self):
        return "Circle"


pdata = (('tlines', VtableElement('tlines', type='string',
                                  guilabel='Tangent lines',
                                  default="",
                                  tip="two tangent lines of cirlce")),
         ('radius', VtableElement('radius', type='float',
                                  guilabel='Radius',
                                  default=1.0,
                                  tip="radius of circle")),
         ('fill_circle', VtableElement('fill_circle', type='bool',
                                       guilabel='Fill circle',
                                       default=False,
                                       tip="Make surface ")), )


class Circle2DRadiusTwoTangentCurve(GeomPB):
    vt = Vtable(pdata)

    @classmethod
    def fancy_menu_name(self):
        return 'Two Tangents and Radius'

    @classmethod
    def fancy_tree_name(self):
        return "Circle"


pdata = (('points', VtableElement('points', type='string',
                                  guilabel='Ends of diameter',
                                  default="",
                                  tip="two points to define diameter")),
         ('fill_circle', VtableElement('fill_circle', type='bool',
                                       guilabel='Fill circle',
                                       default=True,
                                       tip="Make surface ")), )


class Circle2DByDiameter(GeomPB):
    vt = Vtable(pdata)

    @classmethod
    def fancy_menu_name(self):
        return 'Diamter'

    @classmethod
    def fancy_tree_name(self):
        return "Circle"


cdata = (('center', VtableElement('center', type='float',
                                  guilabel='Center',
                                  suffix=('x', 'y', 'z'),
                                  default=[0, 0, 0],
                                  tip="Center of Circle")),
         ('ax1', VtableElement('ax1', type='float',
                               guilabel='Axis1',
                               suffix=('x', 'y', 'z'),
                               default=[1, 0, 0],
                               tip="axis 1")),
         ('ax2', VtableElement('ax2', type='float',
                               guilabel='Axis2',
                               suffix=('x', 'y', 'z'),
                               default=[0, 1, 0],
                               tip="axis 2")),
         ('radius', VtableElement('radius', type='float',
                                  guilabel='r',
                                  default=1.0,
                                  tip="radius")),
         ('fill_circle', VtableElement('fill_circle', type='bool',
                                       guilabel='Fill circle',
                                       default=True,
                                       tip="Make surface ")), )


class Circle(GeomPB):
    vt = Vtable(cdata)

    @classmethod
    def fancy_menu_name(self):
        return 'Circle'

    @classmethod
    def fancy_tree_name(self):
        return 'Circle'


cdata = (('center', VtableElement('center', type='float',
                                  guilabel='Center',
                                  suffix=('x', 'y', 'z'),
                                  default=[0, 0, 0],
                                  tip="Center of Circle")),
         ('ax1', VtableElement('ax1', type='float',
                               guilabel='Axis1',
                               suffix=('x', 'y', 'z'),
                               default=[1, 0, 0],
                               tip="axis 1")),
         ('ax2', VtableElement('ax2', type='float',
                               guilabel='Axis2',
                               suffix=('x', 'y', 'z'),
                               default=[0, 1, 0],
                               tip="axis 2")),
         ('radius', VtableElement('radius', type='float',
                                  guilabel='r',
                                  default=1.0,
                                  tip="radius")),
         ('num_points', VtableElement('num_points', type='int',
                                      guilabel='#points on circle (>0)',
                                      default=2,
                                      tip="numbe of points on circle")),
         ('fill_circle', VtableElement('fill_circle', type='bool',
                                       guilabel='Fill circle',
                                       default=True,
                                       tip="Make surface ")), )


class CircleOCC(GeomPB):
    vt = Vtable(cdata)

    @classmethod
    def fancy_menu_name(self):
        return 'Two Axes and Radius'

    @classmethod
    def fancy_tree_name(self):
        return 'Circle'


cdata = (('axis_by_points', VtableElement('axis_by_points', type='string',
                                          guilabel='Points on axis',
                                          default="",
                                          tip="Points to define axis")),
         ('point_on_cirlce', VtableElement('point_on_circle', type='string',
                                           guilabel='Point on circle',
                                           default="",
                                           tip="Point on circle")),
         ('num_points', VtableElement('num_points', type='int',
                                      guilabel='#points on circle (>0)',
                                      default=2,
                                      tip="numbe of points on circle")),
         ('fill_circle', VtableElement('fill_circle', type='bool',
                                       guilabel='Fill circle',
                                       default=True,
                                       tip="Make surface ")), )


class CircleByAxisPoint(GeomPB):
    vt = Vtable(cdata)

    @classmethod
    def fancy_menu_name(self):
        return 'Axis and Point'

    @classmethod
    def fancy_tree_name(self):
        return 'Circle'


cdata = (('axis_by_points', VtableElement('axis_by_points', type='string',
                                          guilabel='Axis (edge or two points)',
                                          default="",
                                          tip="Points to define axis")),
         ('center', VtableElement('center', type='string',
                                  guilabel='Center point',
                                  default="",
                                          tip="Points to define axis")),
         ('radius', VtableElement('point_on_circle', type='string',
                                  guilabel='Radius',
                                  default="1.0",
                                  tip="Radius")),
         ('num_points', VtableElement('num_points', type='int',
                                      guilabel='#points on circle (>0)',
                                      default=2,
                                      tip="numbe of points on circle")),
         ('fill_circle', VtableElement('fill_circle', type='bool',
                                       guilabel='Fill circle',
                                       default=True,
                                       tip="Make surface ")), )


class CircleByAxisCenterRadius(GeomPB):
    vt = Vtable(cdata)

    @classmethod
    def fancy_menu_name(self):
        return 'Axis/Center/Radius'

    @classmethod
    def fancy_tree_name(self):
        return 'Circle'


cdata = (('point_on_cirlce', VtableElement('point_on_circle', type='string',
                                           guilabel='Points on circle',
                                           default="",
                                           tip="Points on circle")),
         ('fill_circle', VtableElement('fill_circle', type='bool',
                                       guilabel='Fill circle',
                                       default=True,
                                       tip="Make surface ")), )


class CircleBy3Points(GeomPB):
    vt = Vtable(cdata)

    @classmethod
    def fancy_menu_name(self):
        return '3 Points'

    @classmethod
    def fancy_tree_name(self):
        return 'Circle'


rdata = (('corner', VtableElement('corner', type='float',
                                  guilabel='Corner',
                                  suffix=('x', 'y', 'z'),
                                  default=[0, 0, 0],
                                  tip="Center of Circle")),
         ('edge1', VtableElement('edge1', type='float',
                                 guilabel='Edge(1)',
                                 suffix=('x', 'y', 'z'),
                                 default=[1, 0, 0],
                                 tip="Edge of rectangle")),
         ('edge2', VtableElement('edge2', type='float',
                                 guilabel='Edge(2)',
                                 suffix=('x', 'y', 'z'),
                                 default=[0, 1, 0],
                                 tip="Edge of rectangle")),)


class Rect(GeomPB):
    vt = Vtable(rdata)


rdata = (('corner', VtableElement('corner', type='float',
                                  guilabel='Corner',
                                  suffix=('x', 'y', 'z'),
                                  default=[0, 0, 0],
                                  tip="Center of Circle")),
         ('edge1', VtableElement('edge1', type='float',
                                 guilabel='Edge(1)',
                                 suffix=('x', 'y', 'z'),
                                 default=[1, 0, 0],
                                 tip="Edge of rectangle")),
         ('edge2', VtableElement('edge2', type='float',
                                 guilabel='Edge(2)',
                                 suffix=('x', 'y', 'z'),
                                 default=[0, 1, 0],
                                 tip="Edge of rectangle")),
         ('edge3', VtableElement('edge3', type='float',
                                 guilabel='Edge(3)',
                                 suffix=('x', 'y', 'z'),
                                 default=[0, 0, 1],
                                 tip="Edge of rectangle")),)


class Box(GeomPB):
    vt = Vtable(rdata)


vtdata = (('center', VtableElement('center', type='float',
                                   guilabel='Center',
                                   suffix=('x', 'y', 'z'),
                                   default=[0, 0, 0],
                                   tip="Center of Circle")),
          ('x_radius', VtableElement('x_radius', type='float',
                                     guilabel='radius (X)',
                                     default=1.0,
                                     tip="radius in X direction")),
          ('y_radius', VtableElement('y_radius', type='float',
                                     guilabel='radius (Y)',
                                     default=1.0,
                                     tip="radius in Y direction")),
          ('z_rarius', VtableElement('z_radius', type='float',
                                     guilabel='radius (Z)',
                                     default=1.0,
                                     tip="radius in Z direction")),
          ('angle1', VtableElement('angle1', type='float',
                                   guilabel='polar opening(1)',
                                   default=-90,
                                   tip="poloar opening angle (start)")),
          ('angle2', VtableElement('angle2', type='float',
                                   guilabel='polar opening(2)',
                                   default=90.,
                                   tip="polar opening angle (stop)")),
          ('angle3', VtableElement('angle3', type='float',
                                   guilabel='azimuthal opening',
                                   default=360.,
                                   tip="azimuthal opening")),)


class Ball(GeomPB):
    vt = Vtable(vtdata)


vtdata = (('center', VtableElement('center', type='float',
                                   guilabel='Center',
                                   suffix=('x', 'y', 'z'),
                                   default=[0, 0, 0],
                                   tip="Center of Circle")),
          ('axis', VtableElement('axis', type='float',
                                 guilabel='Axis',
                                 suffix=('x', 'y', 'z'),
                                 default=[0, 0, 1],
                                 tip="Center of Circle")),
          ('r1', VtableElement('r1', type='float',
                               guilabel='radius1',
                               default=1.0,
                               tip="r1")),
          ('r2', VtableElement('r2', type='float',
                               guilabel='radius2',
                               default=0.0,
                               tip="r2")),
          ('angle', VtableElement('angle', type='float',
                                  guilabel='Angle',
                                  default=360,
                                  tip="angle")),)


class Cone(GeomPB):
    vt = Vtable(vtdata)


vtdata = (('center', VtableElement('center', type='float',
                                   guilabel='Center',
                                   suffix=('x', 'y', 'z'),
                                   default=[0, 0, 0],
                                   tip="Center of Circle")),
          ('axis', VtableElement('axis', type='float',
                                 guilabel='Axis',
                                 suffix=('x', 'y', 'z'),
                                 default=[0, 0, 1],
                                 tip="Center of Circle")),
          ('radius', VtableElement('radius', type='float',
                                   guilabel='Radius',
                                   default=1.0,
                                   tip="radius")),
          ('angle', VtableElement('angle', type='float',
                                  guilabel='Angle',
                                  default=360,
                                  tip="angle")),)


class Cylinder(GeomPB):
    vt = Vtable(vtdata)


vtdata = (('corner', VtableElement('corner', type='float',
                                   guilabel='Corner',
                                   suffix=('x', 'y', 'z'),
                                   default=[0, 0, 0],
                                   tip="Center of Circle")),
          ('dxdydz', VtableElement('dxdydz', type='float',
                                   guilabel='Size',
                                   suffix=('x', 'y', 'z'),
                                   default=[1, 1, 0.1],
                                   tip="Size of Wedge")),
          ('ltx', VtableElement('ltx', type='float',
                                guilabel='Top Extend',
                                default=0.0,
                                tip="r1")),)


class Wedge(GeomPB):
    vt = Vtable(vtdata)


vtdata = (('center', VtableElement('center', type='float',
                                   guilabel='Center',
                                   suffix=('x', 'y', 'z'),
                                   default=[0, 0, 0],
                                   tip="Center of Circle")),
          ('r1', VtableElement('r1', type='float',
                               guilabel='radius1',
                               default=1.0,
                               tip="r1")),
          ('r2', VtableElement('r2', type='float',
                               guilabel='radius2',
                               default=0.0,
                               tip="r2")),
          ('angle', VtableElement('angle', type='float',
                                  guilabel='Angle',
                                  default=360,
                                  tip="angle")),
          ('keep_interior', VtableElement('keep_interior', type='bool',
                                          guilabel='keep interior surfaces',
                                          default=True,
                                          tip="Keep Intrior Surfaces ")), )


class Torus(GeomPB):
    vt = Vtable(vtdata)


pdata = (('xarr', VtableElement('xarr', type='array',
                                guilabel='X',
                                default='0.0',
                                tip="X")),
         ('yarr', VtableElement('yarr', type='array',
                                guilabel='Y',
                                default='0.0',
                                tip="Y")),
         ('zarr', VtableElement('zarr', type='array',
                                guilabel='Z',
                                default='0.0',
                                tip="Z")),)


class Line(GeomPB):
    vt = Vtable(pdata)

    @classmethod
    def fancy_menu_name(self):
        return 'Line'

    @classmethod
    def fancy_tree_name(self):
        return 'Line'

    def attribute_set(self, v):
        v = super(GeomPB, self).attribute_set(v)
        self.vt.attribute_set(v)
        v["make_spline"] = False
        v["periodic"] = False
        return v

    def panel1_param(self):
        ll = GeomPB.panel1_param(self)
        ll.append(["Spline",
                   self.make_spline, 3, {"text": ""}])
        ll.append(["Close loop(OCC only)",
                   self.periodic, 3, {"text": ""}])
        return ll

    def get_panel1_value(self):
        v = GeomPB.get_panel1_value(self)
        return v + [self.make_spline, self.periodic]

    def preprocess_params(self, engine):
        self.vt.preprocess_params(self)
        return

    def import_panel1_value(self, v):
        GeomPB.import_panel1_value(self, v[:-2])
        self.make_spline = v[-2]
        self.periodic = v[-1]

    def panel1_tip(self):
        tip = GeomPB.panel1_tip(self)
        return tip + ['make spline curve', 'close loop']

    def _make_value_or_expression(self):
        #xarr, yarr, zarr,  lcar = self.vt.make_value_or_expression(self)
        return self.vt.make_value_or_expression(self)

    def add_geom_sequence(self, geom):
        gui_name = self.fullname()
        self.vt.preprocess_params(self)
        gui_param = list(self._make_value_or_expression()) + \
            [self.make_spline, self.periodic]
        geom_name = self.__class__.__name__
        geom.add_sequence(gui_name, gui_param, geom_name)


class LineOCC(Line):
    @classmethod
    def fancy_menu_name(self):
        return 'by XYZ'

    @classmethod
    def fancy_tree_name(self):
        return 'Line'


class Polygon(GeomPB):
    vt = Vtable(pdata)


class Polygon2(Polygon):
    vt = Vtable(pdata)

    @classmethod
    def fancy_menu_name(self):
        return 'by XYZ'

    @classmethod
    def fancy_tree_name(self):
        return 'Polygon'


ldata = (('lines', VtableElement('lines', type='string',
                                 guilabel='Lines',
                                 default="",
                                 tip="lines to be extended")),
         ('ratio', VtableElement('ratio', type='array',
                                 guilabel='Ratio',
                                 default='0.1, 0.1',
                                 tip="Normalized ratio (based on U)")),
         ('num_resample', VtableElement('num_resample', type='int',
                                        guilabel='Resample#',
                                        default='30',
                                        tip="Number of resample points on curve")),)


class ExtendedLine(GeomPB):
    vt = Vtable(ldata)

    @classmethod
    def fancy_menu_name(self):
        return 'from Existing Lines'

    @classmethod
    def fancy_tree_name(self):
        return 'Line'


ldata = (('line', VtableElement('lines', type='string',
                                guilabel='Lines',
                                default="",
                                tip="lines to be extended")),
         ('u_n', VtableElement('u_n', type='float',
                               guilabel='U_n',
                               default='0.0',
                               tip="Normalized position (based on U)")),
         ('length', VtableElement('num_resample', type='array',
                                  guilabel='length',
                                  default='1.0',
                                  tip="Length of line")),
         ('rev_dir', VtableElement('rev_dir', type='bool',
                                   guilabel='Reverse',
                                   default=False,
                                   tip="reverse direction ")), )


class NormalLine2D(GeomPB):
    vt = Vtable(ldata)

    @classmethod
    def fancy_menu_name(self):
        return 'Normal to Existing Lines'

    @classmethod
    def fancy_tree_name(self):
        return 'Line'


ldata = (('points', VtableElement('pts', type='string',
                                  guilabel='Points',
                                  default="",
                                  tip="points to be connected")), )


class OCCPolygon(GeomPB):
    vt = Vtable(ldata)

    @classmethod
    def fancy_menu_name(self):
        return 'by Points'

    @classmethod
    def fancy_tree_name(self):
        return 'Polygon'


class Spline(GeomPB):
    vt = Vtable(ldata)


class CreateLine(GeomPB):
    vt = Vtable(ldata)

    @classmethod
    def fancy_menu_name(self):
        return 'Line'

    @classmethod
    def fancy_tree_name(self):
        return 'Line'


ldata = (('lines', VtableElement('lines', type='string',
                                 guilabel='Lines',
                                 default="",
                                 tip="lines to be connected")), )


class LineLoop(GeomPB):
    vt = Vtable(ldata)


ldata = (('loop1', VtableElement('loop1', type='string',
                                 guilabel='Edges1',
                                 default="",
                                 tip="edges to define the 1st loop")),
         ('loop2', VtableElement('loop2', type='string',
                                 guilabel='Edges2',
                                 default="",
                                 tip="edges to define the 2nd loop")),
         ('solid', VtableElement('solid', type='bool',
                                 guilabel='Solid',
                                 default=False,
                                 tip="create solid ")),
         ('ruled', VtableElement('ruled', type='bool',
                                 guilabel='Ruled surface',
                                 default=False,
                                 tip="create ruled surface ")),
         ('rev1', VtableElement('rev1', type='bool',
                                guilabel='Reverse edges1',
                                default=False,
                                tip="revesre edges 1 direction ")),
         ('rev2', VtableElement('rev2', type='bool',
                                guilabel='Reverse edges2',
                                default=False,
                                tip="reverse edge 2 direction ")), )


class ThruSection(GeomPB):
    vt = Vtable(ldata)


ldata = (('lines', VtableElement('lines', type='string',
                                 guilabel='Lines',
                                 default="",
                                 tip="lines to be connected")),
         ('points', VtableElement('points', type='string',
                                  guilabel='Points on surface',
                                  default="",
                                  tip="points as constraints")),
         ('isplane', VtableElement('isplane_org', type='bool',
                                   guilabel='Use surface filling',
                                   default=False,
                                   tip="Surface filling")), )


class CreateSurface(GeomPB):
    vt = Vtable(ldata)

    @classmethod
    def fancy_menu_name(self):
        return 'Surface'

    @classmethod
    def fancy_tree_name(self):
        return 'Surface'


ldata = (('surfs', VtableElement('surfs', type='string',
                                 guilabel='Surfaces',
                                 default="",
                                 tip="surfacess to be connected")), )


class SurfaceLoop(GeomPB):
    vt = Vtable(ldata)


class CreateVolume(GeomPB):
    vt = Vtable(ldata)

    @classmethod
    def fancy_menu_name(self):
        return 'Volume'

    @classmethod
    def fancy_tree_name(self):
        return 'Volume'


ldata = (('volume', VtableElement('volume', type='string',
                                  guilabel='Volume',
                                  default="",
                                  tip="Volume to generate shell")),
         ('opening', VtableElement('opening', type='string',
                                   guilabel='OpeningFaces',
                                   default="",
                                   tip="Shell opening")),
         ('thickness', VtableElement('thikcness', type='float',
                                     guilabel='Thickness',
                                     default='1.0',
                                     tip="Shell thickness")),
         ('round_conner', VtableElement('round_conner', type='bool',
                                        guilabel='Fillet outer edges',
                                        default=False,
                                        tip="Use rounded conner ")), )


class CreateShell(GeomPB):
    vt = Vtable(ldata)

    @classmethod
    def fancy_menu_name(self):
        return 'Shell'

    @classmethod
    def fancy_tree_name(self):
        return 'Shell'


ldata = (('target', VtableElement('target', type='string',
                                  guilabel='Volume',
                                  default="",
                                  tip="volume to be defeatured")),
         ('surfs', VtableElement('surfs', type='string',
                                 guilabel='Surfaces',
                                 default="",
                                 tip="surfacess to be removed")), )


class RemoveFaces(GeomPB):
    vt = Vtable(ldata)


edata = (('ex_target', VtableElement('ex_target', type='string',
                                     guilabel='Targets (v/f/l/p)',
                                     default="",
                                     tip="extrusion target")),
         ('taxis', VtableElement_Direction('taxis', type='float',
                                           guilabel='Translation',
                                           suffix=('x', 'y', 'z'),
                                           default=[0, 0, 1],
                                           tip="translation axis")),
         ('ex_len', VtableElement('exlen', type='array',
                                  guilabel='Length',
                                  default='1.0',
                                  tip="Extrusion length")), )


class Extrude(GeomPB):
    vt = Vtable(edata)


edata = (('ex_target', VtableElement('ex_target', type='string',
                                     guilabel='Targets (v/f/l/p)',
                                     default="",
                                     tip="extrusion target")),
         ('taxis', VtableElement_Rotation('taxis', type='array',
                                          guilabel='Axis',
                                          default='0, 0, 1',
                                          tip="translation axis")),
         ('angle', VtableElement('angel', type='array',
                                 guilabel='angle',
                                 default='90',
                                 tip="Extrusion length")), )


class Revolve(GeomPB):
    vt = Vtable(edata)


edata = (('ex_target', VtableElement('ex_target', type='string',
                                     guilabel='Targets (v/f/l/p)',
                                     default="",
                                     tip="extrusion target")),
         ('path', VtableElement('path', type='string',
                                guilabel='Path (Lines)',
                                default="",
                                tip="sweep path")),)


class Sweep(GeomPB):
    vt = Vtable(edata)


'''
 objection transformations
'''
data0 = (('target_object', VtableElement('target_object', type='string',
                                         guilabel='Objects (v/f/l/p)',
                                         default="",
                                         tip="object to move")),
         ('dx', VtableElement('dx', type='float',
                              guilabel='dx',
                              default=0.0,
                              tip="x-displacement")),
         ('dy', VtableElement('dy', type='float',
                              guilabel='dy',
                              default=0.0,
                              tip="x-displacement")),
         ('dz', VtableElement('dz', type='float',
                              guilabel='dz',
                              default=0.0,
                              tip="z-displacement")),
         ('keep_org', VtableElement('kepp_org', type='bool',
                                    guilabel='Copy',
                                    default=True,
                                    tip="Keep original")), )


class Move(GeomPB):  # tanslate in gmsh
    vt = Vtable(data0)


data0 = (('target_object', VtableElement('target_object', type='string',
                                         guilabel='Objects (v/f/l/p)',
                                         default="",
                                         tip="object to move")),
         ('point1', VtableElement('point1', type='string',
                                  guilabel='Point (from)',
                                  default="",
                                  tip="first point to define translation")),
         ('point2', VtableElement('point2', type='string',
                                  guilabel='Point (to)',
                                  default="",
                                  tip="2nd point to define translation")),
         ('distance', VtableElement('distance', type='float',
                                    guilabel='Distance',
                                    default="1.0",
                                    tip="Distance")),
         ('scale_d', VtableElement('scale_d', type='bool',
                                   guilabel='Use distance between pionts + scaler',
                                   default=True,
                                   tip="Scale or give the distance for translation")),
         ('keep_org', VtableElement('kepp_org', type='bool',
                                    guilabel='Copy',
                                    default=False,
                                    tip="Keep original")), )


class MoveByPoints(GeomPB):
    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'Move (by Points)'

    @classmethod
    def fancy_tree_name(self):
        return 'Move'


class MovePoint(GeomPB):
    data0 = (('target_object', VtableElement('target_object', type='string',
                                             guilabel='Objects (v/f/l/p)',
                                             default="",
                                             tip="object to move")),
             ('point1', VtableElement('point1', type='string',
                                      guilabel='Point1',
                                      default="",
                                      tip="point to be moved")),
             ('point2', VtableElement('point2', type='string',
                                      guilabel='Point2)',
                                      default="",
                                      tip="destination of move")),)

    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'MovePoint'

    @classmethod
    def fancy_tree_name(self):
        return 'MovePoint'


class SplitHairlineFace(GeomPB):
    data0 = (('target_object', VtableElement('target_object', type='string',
                                             guilabel='Volume',
                                             default="",
                                             tip="object to heal (split face algorithm)")),
             ('area_limit', VtableElement('area_limit', type='float',
                                          guilabel='Area thr.',
                                          default="1e-5",
                                          tip="relative area to check if it is hairline")),
             ('dist_limit', VtableElement('dist_limit', type='float',
                                          guilabel='DistanceLimit',
                                          default="1",
                                          tip="distance limit to split")),
             ('wrong_object', VtableElement('wrong_object', type='string',
                                            guilabel='Faces (optinal)',
                                            default="",
                                            tip="addtional faces to fix")),
             ('recursive', VtableElement('recursive', type='bool',
                                         guilabel='Recursive',
                                         default=False,
                                         tip="apply recursively")), )

    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'SplitHairlineFace'

    @classmethod
    def fancy_tree_name(self):
        return 'SplitHairlineFace'


class CapFaces(GeomPB):
    data0 = (('target_object', VtableElement('target_object', type='string',
                                             guilabel='Volume',
                                             default="",
                                             tip="object to heal (split face algorithm)")),
             ('face_object', VtableElement('face_object', type='string',
                                           guilabel='Faces',
                                           default="",
                                           tip="addtional faces to fix")),
             ('use_filling', VtableElement('use_filling', type='bool',
                                           guilabel='use Surface filling',
                                           default=True,
                                           tip="use surface filling to make a cap face")), )

    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'CapFaces'

    @classmethod
    def fancy_tree_name(self):
        return 'CapFaces'


class ReplaceFaces(GeomPB):
    data0 = (('target_object', VtableElement('target_object', type='string',
                                             guilabel='Volume',
                                             default="",
                                             tip="object to heal (split face algorithm)")),
             ('face_object', VtableElement('face_object', type='string',
                                           guilabel='Faces to replace',
                                           default="",
                                           tip="addtional faces to fix")),
             ('face_object2', VtableElement('face_object2', type='string',
                                            guilabel='New faces',
                                            default="",
                                            tip="faces for replacement")),)

    vt = Vtable(data0)


data0 = (('target_object', VtableElement('target_object', type='string',
                                         guilabel='Objects (v/f/l/p)',
                                         default="",
                                         tip="object to move")),
         ('ctr_rot', VtableElement('ctr_rot', type='float',
                                   guilabel='Center',
                                   suffix=('x', 'y', 'z'),
                                   default=[0, 0, 0],
                                   tip="point on revolustion axis")),
         ('ax_rot', VtableElement('ax_rot', type='float',
                                  guilabel='Axis',
                                  suffix=('x', 'y', 'z'),
                                  default=[0., 0., 1.0],
                                  tip="direction of revolustion axis")),
         ('angle', VtableElement('angle', type='float',
                                 guilabel='angle',
                                 default=180.0,
                                 tip="angle of revoluiton")),
         ('keep_org', VtableElement('kepp_org', type='bool',
                                    guilabel='Copy',
                                    default=True,
                                    tip="Keep original")), )


class Rotate(GeomPB):
    vt = Vtable(data0)


data0 = (('target_object', VtableElement('target_object', type='string',
                                         guilabel='Objects (v/f/l/p)',
                                         default="",
                                         tip="object to move")),
         ('cpoint', VtableElement('cpoint', type='string',
                                  guilabel='Center',
                                  default="",
                                  tip="center of rotation")),
         ('twopoints', VtableElement('twopoints', type='string',
                                     guilabel='Point Pair',
                                     default="",
                                     tip="pair of points to match")),
         ('use_sup', VtableElement('use_sup', type='bool',
                                   guilabel='Use supplementary angle',
                                   default=False,
                                   tip="Use 180 - angle (deg)")),
         ('keep_org', VtableElement('kepp_org', type='bool',
                                    guilabel='Copy',
                                    default=True,
                                    tip="Keep original")), )


class RotateCenterPoints(GeomPB):
    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'Rotate (by Points)'

    @classmethod
    def fancy_tree_name(self):
        return 'Rotate'


data0 = (('target_object', VtableElement('target_object', type='string',
                                         guilabel='Objects (v/f/l/p)',
                                         default="",
                                         tip="object to move")),
         ('ctr_scale', VtableElement('ctr_scale', type='float',
                                     guilabel='Center',
                                     suffix=('x', 'y', 'z'),
                                     default=[0, 0, 0],
                                     tip="center of scale")),
         ('size_scale', VtableElement('size_scale', type='float',
                                      guilabel='Scale',
                                      suffix=('x', 'y', 'z'),
                                      default=[1., 1., 1.],
                                      tip="scale size")),
         ('keep_org', VtableElement('kepp_org', type='bool',
                                    guilabel='Copy',
                                    default=True,
                                    tip="Keep original")), )


class Scale(GeomPB):  # Dilate in gmsh
    vt = Vtable(data0)


class Array(GeomPB):
    data0 = (('target_object', VtableElement('target_object', type='string',
                                             guilabel='Objects (v/f/l/p)',
                                             default="",
                                             tip="object to move")),
             ('array_count', VtableElement('array_count', type='int',
                                           guilabel='Count', default=1,
                                           tip="Center of Circle")),
             ('displacement', VtableElement('displacement', type='float',
                                            guilabel='displacement',
                                            suffix=('x', 'y', 'z'),
                                            default=[1, 0, 0],
                                            tip="displacemnt")),)
    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'Linear (by Displacement)'

    @classmethod
    def fancy_tree_name(self):
        return 'Array'


class ArrayByPoints(GeomPB):
    data0 = (('target_object', VtableElement('target_object', type='string',
                                             guilabel='Objects (v/f/l/p)',
                                             default="",
                                             tip="object to move")),
             ('array_count', VtableElement('array_count', type='int',
                                           guilabel='Count', default=1,
                                           tip="Center of Circle")),
             ('ref_ptx', VtableElement('ref_ptx', type='string',
                                       guilabel='Points',
                                       default="",
                                       tip="Reference points to define linear translation")),)
    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'Linear (by Points)'

    @classmethod
    def fancy_tree_name(self):
        return 'Array'


class ArrayRot(GeomPB):
    data0 = (('target_object', VtableElement('target_object', type='string',
                                             guilabel='Objects (v/f/l/p)',
                                             default="",
                                             tip="object to move")),
             ('array_count', VtableElement('array_count', type='int',
                                           guilabel='Count', default=1,
                                           tip="Center of Circle")),
             ('ctr_rot', VtableElement('ctr_rot', type='float',
                                       guilabel='Center',
                                       suffix=('x', 'y', 'z'),
                                       default=[0, 0, 0],
                                       tip="point on revolustion axis")),
             ('ax_rot', VtableElement('ax_rot', type='float',
                                      guilabel='Axis',
                                      suffix=('x', 'y', 'z'),
                                      default=[0., 0., 1.0],
                                      tip="direction of revolustion axis")),
             ('angle', VtableElement('angle', type='float',
                                     guilabel='angle',
                                     default=180.0,
                                     tip="angle of revoluiton")),)
    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'Angular (by Angle)'

    @classmethod
    def fancy_tree_name(self):
        return 'ArrayRot'


class ArrayRotByPoints(GeomPB):
    data0 = (('target_object', VtableElement('target_object', type='string',
                                             guilabel='Objects (v/f/l/p)',
                                             default="",
                                             tip="object to move")),
             ('array_count', VtableElement('array_count', type='int',
                                           guilabel='Count', default=1,
                                           tip="Center of Circle")),
             ('ctr_rot', VtableElement('ctr_rot', type='float',
                                       guilabel='Center',
                                       suffix=('x', 'y', 'z'),
                                       default=[0, 0, 0],
                                       tip="point on revolustion axis")),
             ('ax_rot', VtableElement('ax_rot', type='float',
                                      guilabel='Axis',
                                      suffix=('x', 'y', 'z'),
                                      default=[0., 0., 1.0],
                                      tip="direction of revolustion axis")),
             ('ref_ptx', VtableElement('ref_ptx', type='string',
                                       guilabel='Points',
                                       default="",
                                       tip="Reference points to define angular translation")),)
    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'Angular (by Points)'

    @classmethod
    def fancy_tree_name(self):
        return 'ArrayRot'


class ArrayPath(GeomPB):
    data0 = (('target_object', VtableElement('target_object', type='string',
                                             guilabel='Objects (v/f/l/p)',
                                             default="",
                                             tip="object to move")),
             ('ref_pnt', VtableElement('ref_pnt', type='string',
                                       default='',
                                       guilabel='Reference Point',
                                       tip="Reference point of object")),
             ('pathlines', VtableElement('pathlines', type='string',
                                         default='',
                                         guilabel='Path(lines)',
                                         tip="Path to crate array")),
             ('array_count', VtableElement('array_count', type='int',
                                           guilabel='Count',
                                           default=1,
                                           tip="Center of Circle")),
             ('margin_start', VtableElement('margin_start',
                                            type='float',
                                            guilabel='Margin(start)',
                                            default=1,
                                            tip="Margin at the path beginning")),
             ('margin_end', VtableElement('margin_end',
                                          type='float',
                                          guilabel='Margin(end)',
                                          default=1,
                                          tip="Margin at the path end")),
             ('ignore_angle', VtableElement('ignore_angle',
                                            type='bool',
                                            guilabel='Translation only',
                                            default=False,
                                            tip="Ignore rotation along the path")), )

    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'Path'

    @classmethod
    def fancy_tree_name(self):
        return 'ArrayPath'


data0 = (('target_object', VtableElement('target_object',
                                         type='string',
                                         guilabel='Objects (v/f/l/p)',
                                         default="",
                                         tip="object to move")),
         ('flip_plane', VtableElement_Plane('flip_plane', type='float',
                                            guilabel='flip plane',
                                            suffix=('a', 'b', 'c', 'd'),
                                            default=[1, 0, 0, 0],
                                            tip="Flipping Plane")),
         ('keep_org', VtableElement('kepp_org', type='bool',
                                    guilabel='Copy',
                                    default=True,
                                    tip="Keep original")), )


class Flip(GeomPB):
    vt = Vtable(data0)


data0 = (('target_object', VtableElement('target_object',
                                         type='string',
                                         guilabel='Volume',
                                         default="",
                                         tip="object to add fillet")),
         ('curves', VtableElement('curves', type='string',
                                  guilabel='Curves',
                                  default="",
                                  tip="curves to add fillet")),
         ('radius', VtableElement('radisu', type='array',
                                  guilabel='Radii',
                                  default=1.0,
                                  tip="radisu")),)


class Fillet(GeomPB):
    vt = Vtable(data0)


data0 = (('target_object', VtableElement('target_object', type='string',
                                         guilabel='Volume',
                                         default="",
                                         tip="object to add chamfer")),
         ('curves', VtableElement('curves', type='string',
                                  guilabel='Curves',
                                  default="",
                                  tip="curves to add chamfer")),
         ('distance', VtableElement('distance', type='array',
                                    guilabel='Distances',
                                    default="1.0",
                                    tip="distance")),
         ('surfaces', VtableElement('surfaces', type='string',
                                    guilabel='Surfaces',
                                    default="",
                                    tip="distance")),)


class Chamfer(GeomPB):
    vt = Vtable(data0)


class Fillet2D(GeomPB):
    data0 = (('target_object', VtableElement('target_object',
                                             type='string',
                                             guilabel='Surface',
                                             default="",
                                             tip="object to add fillet")),
             ('corner', VtableElement('corner', type='string',
                                      guilabel='Corner points',
                                      default="",
                                      tip="corner to add fillet")),
             ('radius', VtableElement('radisu', type='float',
                                      guilabel='Radius',
                                      default=1.0,
                                      tip="radisu")),)

    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'Fillet'

    @classmethod
    def fancy_tree_name(self):
        return 'Fillet'


class Chamfer2D(GeomPB):
    data0 = (('target_object', VtableElement('target_object', type='string',
                                             guilabel='Surface',
                                             default="",
                                             tip="object to add chamfer")),
             ('edges', VtableElement('edges', type='string',
                                     guilabel='Pair of edges',
                                     default="",
                                     tip="Coner to add chamfer")),
             ('distance1', VtableElement('distance1', type='float',
                                         guilabel='1st distances',
                                         default=1.0,
                                         tip="distance")),
             ('distance2', VtableElement('distance2', type='string',
                                         guilabel='2d distances (optional)',
                                         default='',
                                         tip="distance")),)

    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'Chamfer'

    @classmethod
    def fancy_tree_name(self):
        return 'Chamfer'


data0 = (('target_object', VtableElement('target_object',
                                         type='string',
                                         guilabel='Objects (v/f/l/p)',
                                         default="",
                                         tip="object to move")), )


class Copy(GeomPB):
    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'Copy'

    @classmethod
    def fancy_tree_name(self):
        return 'Copy'


data0 = (('target_object', VtableElement('target_object', type='string',
                                         guilabel='Objects (v/f/l/p)',
                                         default="",
                                         tip="object to move")),
         ('recursive', VtableElement('recursive', type='bool',
                                     guilabel='Recursive',
                                     default=True,
                                     tip="delete recursively")), )


class Remove(GeomPB):
    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'Delete'

    @classmethod
    def fancy_tree_name(self):
        return 'Delete'


data0 = (('target_object', VtableElement('target_object', type='string',
                                         guilabel='Objects (v/f/l/p)',
                                         default="",
                                         tip="object to move")),)


class Remove2(GeomPB):
    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'Delete Rest'

    @classmethod
    def fancy_tree_name(self):
        return 'DeleteRest'


data0 = (('target_object', VtableElement('target_object', type='string',
                                         guilabel='Surfaces',
                                         default="",
                                         tip="surfaces")),
         ('use_unifier', VtableElement('use_unifier', type='bool',
                                       guilabel='Merge edge',
                                       default=True,
                                       tip="Use unifier to merge edges if possible")), )


class MergeFace(GeomPB):
    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'Merge Faces'

    @classmethod
    def fancy_tree_name(self):
        return "MergeFace"


class GeomPB_Bool(GeomPB):
    def attribute_set(self, v):
        v = super(GeomPB, self).attribute_set(v)
        self.vt.attribute_set(v)
        v["delete_input"] = True
        v["delete_tool"] = True
        v["keep_highest"] = False
        v["tol_scale"] = 1.0
        return v

    def panel1_param(self):
        ll = GeomPB.panel1_param(self)
        ll.append(["Delete Input",
                   self.delete_input, 3, {"text": ""}])
        ll.append(["Delete Tool",
                   self.delete_tool, 3, {"text": ""}])
        ll.append(["Keep highest dim only",
                   self.keep_highest, 3, {"text": ""}])
        ll.append(["Tolelance scalar",
                   self.tol_scale, 300, {"text": ""}])

        return ll

    def get_panel1_value(self):
        v = GeomPB.get_panel1_value(self)
        return v + [self.delete_input,
                    self.delete_tool, self.keep_highest, self.tol_scale]

    def preprocess_params(self, engine):
        self.vt.preprocess_params(self)
        return

    def import_panel1_value(self, v):
        GeomPB.import_panel1_value(self, v[:-4])
        self.delete_input = v[-4]
        self.delete_tool = v[-3]
        self.keep_highest = v[-2]
        self.tol_scale = v[-1]

    def panel1_tip(self):
        tip = GeomPB.panel1_tip(self)
        return tip + ['delete input objects'] + \
            ['delete tool objects'] + \
            ['keep highest dim. objects'] + ['tolelance scalar']

    def add_geom_sequence(self, geom):
        gui_name = self.fullname()
        gui_param = (list(self.vt.make_value_or_expression(self)) +
                     [self.delete_input,
                      self.delete_tool, self.keep_highest, self.tol_scale])
        geom_name = self.__class__.__name__
        geom.add_sequence(gui_name, gui_param, geom_name)


class GeomPB_Bool_with_UP(GeomPB):
    def attribute_set(self, v):
        v = super(GeomPB, self).attribute_set(v)
        self.vt.attribute_set(v)
        v["delete_input"] = True
        v["delete_tool"] = True
        v["keep_highest"] = False
        v["use_upgrade"] = False
        v["tol_scale"] = 1.0
        return v

    def panel1_param(self):
        ll = GeomPB.panel1_param(self)
        ll.append(["Delete Input",
                   self.delete_input, 3, {"text": ""}])
        ll.append(["Delete Tool",
                   self.delete_tool, 3, {"text": ""}])
        ll.append(["Keep highest dim only",
                   self.keep_highest, 3, {"text": ""}])
        ll.append(["Use domain unifier",
                   self.use_upgrade, 3, {"text": ""}])
        ll.append(["Tolelance scalar",
                   self.tol_scale, 300, {"text": ""}])

        return ll

    def get_panel1_value(self):
        v = GeomPB.get_panel1_value(self)
        return v + [self.delete_input, self.delete_tool,
                    self.keep_highest, self.use_upgrade, self.tol_scale]

    def preprocess_params(self, engine):
        self.vt.preprocess_params(self)
        return

    def import_panel1_value(self, v):
        GeomPB.import_panel1_value(self, v[:-5])
        self.delete_input = v[-5]
        self.delete_tool = v[-4]
        self.keep_highest = v[-3]
        self.use_upgrade = v[-2]
        self.tol_scale = v[-1]

    def panel1_tip(self):
        tip = GeomPB.panel1_tip(self)
        return tip + ['delete input objects', 'delete tool objects',
                      'keep highest dim. objects', 'unify the same domains',
                      'scale tolerance of boolear operaion']

    def add_geom_sequence(self, geom):
        gui_name = self.fullname()
        gui_param = (list(self.vt.make_value_or_expression(self)) +
                     [self.delete_input, self.delete_tool,
                      self.keep_highest, self.use_upgrade, self.tol_scale])
        geom_name = self.__class__.__name__
        geom.add_sequence(gui_name, gui_param, geom_name)


ddata = (('objplus', VtableElement('objplus', type='string',
                                   guilabel='+ (v/f/l/p)',
                                   default="",
                                   tip="added objects")),
         ('objminus', VtableElement('objminus', type='string',
                                    guilabel='- (v/f/l/p)',
                                    default="",
                                    tip="objects to be subtracted")),)


class Difference(GeomPB_Bool_with_UP):
    vt = Vtable(ddata)


udata = (('objplus', VtableElement('obj1', type='string',
                                   guilabel='Input (v/f/l/p)',
                                   default="",
                                   tip="objects")),
         ('tool_object', VtableElement('tool_object', type='string',
                                       guilabel='Tool Obj. (v/f/l/p)',
                                       default="",
                                       tip="object to move")),)


class Union(GeomPB_Bool_with_UP):
    vt = Vtable(udata)

    @classmethod
    def fancy_menu_name(self):
        return 'Union'

    @classmethod
    def fancy_tree_name(self):
        return "Union"


class Union2(Union):
    vt = Vtable(udata)

    def attribute_set(self, v):
        v = super(Union, self).attribute_set(v)
        self.vt.attribute_set(v)
        v["use_upgrade"] = False
        return v


class Union2D(GeomPB_Bool):
    vt = Vtable(udata)

    @classmethod
    def fancy_menu_name(self):
        return 'Union'

    @classmethod
    def fancy_tree_name(self):
        return 'Union'


class Intersection(GeomPB_Bool_with_UP):
    vt = Vtable(udata)


class Fragments(GeomPB_Bool):
    vt = Vtable(udata)


pdata = (('objp', VtableElement("obj1p", type='string',
                                guilabel='Faces/Edges/Points (l/p)',
                                default="",
                                tip="objects")),
         ('fill_wire', VtableElement('fill_wire', type='bool',
                                     guilabel='Fill',
                                     default=False,
                                     tip='Fill projected faces')),)


class ProjectOnWP(GeomPB):
    vt = Vtable(pdata)

    @classmethod
    def fancy_menu_name(self):
        return 'Projection'

    @classmethod
    def fancy_tree_name(self):
        return 'Projection'


pdata = (('xarr', VtableElement('xarr', type='array',
                                guilabel='X',
                                default='0.0',
                                tip="X")),
         ('yarr', VtableElement('yarr', type='array',
                                guilabel='Y',
                                default='0.0',
                                tip="Y")),)


class Point2D(GeomPB):
    vt = Vtable(pdata)

    @classmethod
    def fancy_menu_name(self):
        return 'Point'

    @classmethod
    def fancy_tree_name(self):
        return 'Point'


pdata = (('xarr', VtableElement('xarr', type='array',
                                guilabel='X',
                                default='0.0',
                                tip="X")),
         ('yarr', VtableElement('yarr', type='array',
                                guilabel='Y',
                                default='0.0',
                                tip="Y")),)


class Line2D(Line):
    vt = Vtable(pdata)

    def _make_value_or_expression(self):
        #xarr, yarr, zarr = self.vt.make_value_or_expression(self)
        xarr, yarr = self.vt.make_value_or_expression(self)
        zarr = [0.0 for x in yarr]
        return xarr, yarr, zarr

    @classmethod
    def fancy_menu_name(self):
        return 'Line'

    @classmethod
    def fancy_tree_name(self):
        return 'Line'


cdata = (('center', VtableElement('center', type='float',
                                  guilabel='Center',
                                  suffix=('x', 'y', ),
                                  default=[0, 0, ],
                                  tip="Center of Circle")),
         ('ax1', VtableElement('ax1', type='float',
                               guilabel='axis1',
                               suffix=('x', 'y',),
                               default=[1, 0, ],
                               tip="axis 1")),
         ('ax2', VtableElement('ax2', type='float',
                               guilabel='axis2',
                               suffix=('x', 'y', ),
                               default=[0, 1, ],
                               tip="axis 2")),
         ('radius', VtableElement('radius', type='float',
                                  guilabel='r',
                                  default=1.0,
                                  tip="radius")),)


class Circle2D(GeomPB):
    vt = Vtable(cdata)

    @classmethod
    def fancy_menu_name(self):
        return 'Circle'

    @classmethod
    def fancy_tree_name(self):
        return 'Circle'


cdata = (('center', VtableElement('center', type='float',
                                  guilabel='Center',
                                  suffix=('x', 'y', ),
                                  default=[0, 0, ],
                                  tip="Center of Circle")),
         ('ax1', VtableElement('ax1', type='float',
                               guilabel='axis1',
                               suffix=('x', 'y',),
                               default=[1, 0, ],
                               tip="axis 1")),
         ('ax2', VtableElement('ax2', type='float',
                               guilabel='axis2',
                               suffix=('x', 'y', ),
                               default=[0, 1, ],
                               tip="axis 2")),
         ('radius', VtableElement('radius', type='float',
                                  guilabel='r',
                                  default=1.0,
                                  tip="radius")),
         ('angle1', VtableElement('angle1', type='float',
                                  guilabel='angle1',
                                  default=0.0,
                                  tip="radius")),
         ('angle2', VtableElement('angle2', type='float',
                                  guilabel='angle2',
                                  default=90.0,
                                  tip="radius")),
         ('fillarc', VtableElement('fillarc', type='bool',
                                   guilabel='fill',
                                   default=True,
                                   tip="fill arc")), )


class Arc2D(GeomPB):
    vt = Vtable(cdata)

    @classmethod
    def fancy_menu_name(self):
        return 'Arc by Coords'

    @classmethod
    def fancy_tree_name(self):
        return 'Arc'


cdata = (('point_on_cirlce', VtableElement('point_on_circle', type='string',
                                           guilabel='Points(3) on arc',
                                           default="",
                                           tip="Points on arc")),
         ('fillarc', VtableElement('fillarc', type='bool',
                                   guilabel='fill',
                                   default=True,
                                   tip="fill arc")), )


class Arc2DBy3Points(GeomPB):
    vt = Vtable(cdata)

    @classmethod
    def fancy_menu_name(self):
        return 'Arc by 3 Points'

    @classmethod
    def fancy_tree_name(self):
        return 'Arc'


cdata = (('point_on_cirlce', VtableElement('point_on_circle', type='string',
                                           guilabel='Points(2) on arc',
                                           default="",
                                           tip="Points on arc")),
         ('angle2', VtableElement('angle2', type='float',
                                  guilabel='Angle',
                                  default=90.0,
                                  tip="Angle of arc")),
         ('fillarc', VtableElement('fillarc', type='bool',
                                   guilabel='fill',
                                   default=True,
                                   tip="fill arc")), )


class Arc2DBy2PointsAngle(GeomPB):
    vt = Vtable(cdata)

    @classmethod
    def fancy_menu_name(self):
        return 'Arc by 2 Points and Angle'

    @classmethod
    def fancy_tree_name(self):
        return 'Arc'


rdata = (('corner', VtableElement('corner', type='float',
                                  guilabel='Corner',
                                  suffix=('x', 'y'),
                                  default=[0, 0],
                                  tip="Center of Circle")),
         ('edge1', VtableElement('edge1', type='float',
                                 guilabel='Edge(1)',
                                 suffix=('x', 'y'),
                                 default=[1, 0],
                                 tip="Edge of rectangle")),
         ('edge2', VtableElement('edge2', type='float',
                                 guilabel='Edge(2)',
                                 suffix=('x', 'y'),
                                 default=[0, 1],
                                 tip="Edge of rectangle")),)


class Rect2D(GeomPB):
    vt = Vtable(rdata)

    @classmethod
    def fancy_menu_name(self):
        return 'Rect'

    @classmethod
    def fancy_tree_name(self):
        return 'Rect'


cdata = (('corner_of_rect', VtableElement('corner_of_rect', type='string',
                                          guilabel='Points (2) at corner',
                                          default="",
                                          tip="Points at two corners")),)


class Rect2DByCorners(GeomPB):
    vt = Vtable(cdata)

    @classmethod
    def fancy_menu_name(self):
        return 'Rect by 2 Corners'

    @classmethod
    def fancy_tree_name(self):
        return 'Rect'


pdata = (('xarr', VtableElement('xarr', type='array',
                                guilabel='X',
                                default='0.0',
                                tip="X")),
         ('yarr', VtableElement('yarr', type='array',
                                guilabel='Y',
                                default='0.0',
                                tip="Y")),)


class Polygon2D(GeomPB):
    vt = Vtable(pdata)

    @classmethod
    def fancy_menu_name(self):
        return 'Polygon'


class Spline2D(Spline):

    @classmethod
    def fancy_menu_name(self):
        return 'Spline'

    @classmethod
    def fancy_tree_name(self):
        return 'Spline'


data0 = (('target_object', VtableElement('target_object', type='string',
                                         guilabel='Object',
                                         default="",
                                         tip="object to move")),
         ('dx', VtableElement('dx', type='float',
                              guilabel='dx',
                              default=0.0,
                              tip="x-displacement")),
         ('dy', VtableElement('dy', type='float',
                              guilabel='dy',
                              default=0.0,
                              tip="y-displacement")),
         ('keep_org', VtableElement('kepp_org', type='bool',
                                    guilabel='Copy',
                                    default=True,
                                    tip="Keep original")), )


class Move2D(GeomPB):  # tanslate in gmsh
    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'Move'

    @classmethod
    def fancy_tree_name(self):
        return 'Move'


data0 = (('target_object', VtableElement('target_object', type='string',
                                         guilabel='Object',
                                         default="",
                                         tip="object to move")),
         ('ctr_rot', VtableElement('ctr_rot', type='float',
                                   guilabel='Center',
                                   suffix=('x', 'y',),
                                   default=[0, 0],
                                   tip="point on revolustion axis")),
         ('angle', VtableElement('angle', type='float',
                                 guilabel='angle',
                                 default=180.0,
                                 tip="angle of revoluiton")),
         ('keep_org', VtableElement('kepp_org', type='bool',
                                    guilabel='Copy',
                                    default=True,
                                    tip="Keep original")), )


class Rotate2D(GeomPB):
    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'Rotate'

    @classmethod
    def fancy_tree_name(self):
        return 'Rotate'


data0 = (('target_object', VtableElement('target_object', type='string',
                                         guilabel='Object',
                                         default="",
                                         tip="object to move")),
         ('flip_ax', VtableElement('flip_ax', type='float',
                                   guilabel='Flip Axis X',
                                   default=0.0,
                                   tip="direction on flip axis")),
         ('flip_ay', VtableElement('flip_ay', type='float',
                                   guilabel='Flip Axis Y',
                                   default=1.0,
                                   tip="direction on flip axis")),
         ('flip_d', VtableElement('flip_d', type='float',
                                  guilabel='Offset',
                                  default=0.0,
                                  tip="direction of revolustion axis")),
         ('keep_org', VtableElement('kepp_org', type='bool',
                                    guilabel='Copy',
                                    default=True,
                                    tip="Keep original")), )


class Flip2D(GeomPB):
    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'Flip'

    @classmethod
    def fancy_tree_name(self):
        return 'Flip'


data0 = (('target_object', VtableElement('target_object', type='string',
                                         guilabel='Object',
                                         default="",
                                         tip="object to move")),
         ('ctr_scale', VtableElement('ctr_scale', type='float',
                                     guilabel='Center',
                                     suffix=('x', 'y'),
                                     default=[0., 0.],
                                     tip="center of scale")),
         ('size_scale', VtableElement('size_scale', type='float',
                                      guilabel='Scale',
                                      suffix=('x', 'y'),
                                      default=[1., 1.],
                                      tip="scale size")),
         ('keep_org', VtableElement('kepp_org', type='bool',
                                    guilabel='Copy',
                                    default=True,
                                    tip="Keep original")), )


class Scale2D(GeomPB):  # Dilate in gmsh
    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'Scale'

    @classmethod
    def fancy_tree_name(self):
        return 'Scale'


class CreateOffset(GeomPB):
    data0 = (('target_object', VtableElement('target_object', type='string',
                                             guilabel='Face/Edges (f/l)',
                                             default="",
                                             tip="object from which offset is created")),
             ('displacements', VtableElement('displacement', type='array',
                                             guilabel='Displacements',
                                             tip="displacemnts. ex) 1, 2, 3")),
             ('altitudes', VtableElement('altitudes', type='array',
                                         guilabel='Altitudes',
                                         tip="altitude (displacement normal to plane). scalar or array with the same length of displacements")),
             ('round_conner', VtableElement('round_conner', type='bool',
                                            guilabel='Fillet conner',
                                            default=False,
                                            tip="Use rounded conner")),
             ('close_wire', VtableElement('close_wire', type='bool',
                                          guilabel='Close lines',
                                          default=False,
                                          tip="Close offset lines")),
             ('fill_wire', VtableElement('fill_wire', type='bool',
                                         guilabel='Fill',
                                         default=False,
                                         tip="Fill when it is closed wire")), )

    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'OffsetLines'

    @classmethod
    def fancy_Tree_name(self):
        return 'OffsetLines'


class CreateOffsetFace(GeomPB):
    data0 = (('target_object', VtableElement('target_object', type='string',
                                             guilabel='Vol/Faces (v/f)',
                                             default="",
                                             tip="object from which offset is created")),
             ('displacements', VtableElement('displacement', type='array',
                                             guilabel='Displacements',
                                             tip="displacemnts. ex) 1, 2, 3")),

             ('round_conner', VtableElement('round_conner', type='bool',
                                            guilabel='Fillet conner',
                                            default=False,
                                            tip="Use rounded conner")),
             ('fill_shell', VtableElement('fill_shell', type='bool',
                                          guilabel='Fill',
                                          default=False,
                                          tip="Fill when it is closed shell")), )

    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'OffsetFaces'

    @classmethod
    def fancy_Tree_name(self):
        return 'OffsetFaces'


class CreateProjection(GeomPB):
    data0 = (('target_object', VtableElement('target_object', type='string',
                                             guilabel='Faces/Edges',
                                             default="",
                                             tip="edges to be projected")),
             ('taxis', VtableElement_Plane('taxis', type='float',
                                           guilabel='cut plane',
                                           suffix=('a', 'b', 'c', 'd'),
                                           default=[1, 0, 0, 0],
                                           tip="Cutting Plane")),
             ('offset', VtableElement('offset', type='float',
                                      guilabel='offset',
                                      default=0.0,
                                      tip="offset in normal direction")),
             ('fill_edges', VtableElement('fill_edges', type='bool',
                                          guilabel='Fill',
                                          default=False,
                                          tip="Fill projected faces")), )

    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'EdgeProjection'

    @classmethod
    def fancy_Tree_name(self):
        return 'Projection (Edge)'


class Array2D(GeomPB):
    data0 = (('target_object', VtableElement('target_object', type='string',
                                             guilabel='Object',
                                             default="",
                                             tip="object to move")),
             ('array_count', VtableElement('array_count', type='int',
                                           guilabel='Count', default=1,
                                           tip="Center of Circle")),
             ('displacement', VtableElement('displacement', type='float',
                                            guilabel='displacement',
                                            suffix=('x', 'y',),
                                            default=[1, 0, ],
                                            tip="displacemnt")),)
    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'Array'

    @classmethod
    def fancy_Tree_name(self):
        return 'Array'


class Simplifiers(GeomPB):
    data0 = (('target', VtableElement('target', type='string',
                                      guilabel='Volume',
                                      default="",
                                      tip="target object")),
             ('use_unifier', VtableElement('use_unifier', type='bool',
                                           guilabel='Domain Unifier',
                                           default=False,
                                           tip="Merge faces/edges using DomainUnifier")),
             ('use_rm_internal', VtableElement('use_rm_internal', type='bool',
                                               guilabel='Remove internal wire',
                                               default=False,
                                               tip="Use RemoveInternalWires")),
             ('rm_size', VtableElement('rm_size', type='float',
                                       guilabel='displacement',
                                       default=0.1,
                                       tip="Size of minimum area")),)

    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'Simplifers'

    @classmethod
    def fancy_tree_name(self):
        return 'Simplifers'


edata = (('ex_target', VtableElement('ex_target', type='string',
                                     guilabel='Target',
                                     default="",
                                     tip="target object")),
         ('taxis', VtableElement_Plane('taxis', type='float',
                                       guilabel='cut plane',
                                       suffix=('a', 'b', 'c', 'd'),
                                       default=[1, 0, 0, 0],
                                       tip="Cutting Plane")),
         ('offset', VtableElement('offset', type='float',
                                  guilabel='offset',
                                  default=0.0,
                                  tip="offset in normal direction")), )


class SplitByPlane(GeomPB):
    vt = Vtable(edata)


data0 = (('center', VtableElement('center', type='float',
                                  guilabel='Center',
                                  suffix=('x', 'y', 'z'),
                                  default=[0, 0, 0],
                                  tip="Center of WP")),
         ('ax1', VtableElement('ax1', type='float',
                               guilabel='1st Axis',
                               suffix=('x', 'y', 'z'),
                               default=[0, 1, 0],
                               tip="1st Axis")),
         ('ax2', VtableElement('ax2', type='float',
                               guilabel='2nd Axis',
                               suffix=('x', 'y', 'z'),
                               default=[0, 0, 1],
                               tip="2nd Axis")),)


class WPBase(GeomPB):
    isWP = True

    def get_possible_child(self):
        return [Point2D, PointCircleCenter, Line2D, NormalLine2D,
                Circle2D, CircleBy3Points, Circle2DCenterOnePoint,
                Circle2DByDiameter, Circle2DRadiusTwoTangentCurve,
                Arc2D, Arc2DBy3Points, Arc2DBy2PointsAngle,
                Rect2D, Rect2DByCorners,
                Polygon2D, Spline2D,
                Move2D, Rotate2D, Flip2D, Scale2D, Array2D,
                Union2D, Intersection, Difference, Fragments, Copy, Remove,
                Fillet2D, Chamfer2D,
                CreateLine, CreateSurface, ProjectOnWP, OCCPolygon]

    def get_possible_child_menu(self):
        return [("Add Point...", Point2D), ("", PointCenter), ("", PointOnEdge),
                ("!", PointCircleCenter),
                ("Add Line/Arc", Line2D), ("", NormalLine2D),
                ("", Arc2D), ("", Arc2DBy3Points),
                ("!", Arc2DBy2PointsAngle),
                ("Add Rect", Rect2D), ("!", Rect2DByCorners),
                ("Add Circle...", Circle2D), ("", CircleBy3Points),
                # ("", Circle2DRadiusTwoTangentCurve),
                ("", Circle2DCenterOnePoint),
                ("!", Circle2DByDiameter),
                ("", Spline2D),
                ("Create...", CreateLine), ("", CreateSurface),
                ("", OCCPolygon), ("", CreateOffset), ("!", ProjectOnWP),
                ("Fillet/Chamfer", Fillet2D), ("!", Chamfer2D),
                ("Copy/Remove...", Copy), ("!", Remove),
                ("Translate...", Move2D), ("", Rotate2D),
                ("", Flip2D), ("", Scale2D), ("!", Array2D),
                ("Boolean...", Union2D),
                ("", Intersection), ("", Difference), ("!", Fragments), ]

    def add_geom_sequence_wp_start(self, geom):
        gui_name = self.fullname()
        self.vt.preprocess_params(self)
        gui_param = self.vt.make_value_or_expression(self)
        geom_name = self.__class__.__name__ + "Start"
        geom.add_sequence(gui_name, gui_param, geom_name)

    def add_geom_sequence_wp_end(self, geom):
        gui_name = self.fullname()
        self.vt.preprocess_params(self)
        gui_param = self.vt.make_value_or_expression(self)
        geom_name = "WorkPlaneEnd"
        geom.add_sequence(gui_name, gui_param, geom_name)


class WorkPlane(WPBase):
    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'WP by Coords'

    @classmethod
    def fancy_tree_name(self):
        return 'WorkPlane'


class Subsequence(GeomPB):
    vt = Vtable(tuple())
    can_rename = True

    @classmethod
    def fancy_menu_name(self):
        return 'Sequence'

    @classmethod
    def fancy_tree_name(self):
        return 'Sequence'

    @property
    def ns_name(self):
        return self.fullname()

    @ns_name.setter
    def ns_name(self, value):
        pass

    def get_default_ns(self):
        return self.seq_values

    def attribute_set(self, v):
        v = super(Subsequence, self).attribute_set(v)
        v["seq_values"] = {}
        return v

    def get_possible_child(self):
        return [PointOCC, LineOCC, CircleOCC, Polygon2,
                Point, PointCenter, PointOnEdge, PointByUV, PointCircleCenter,
                Line, Circle, CircleByAxisPoint, CircleBy3Points,
                CircleByAxisCenterRadius,
                Rect, Polygon,  OCCPolygon, Spline, Box,
                Ball, Cone, Wedge, Cylinder,
                Torus, CreateLine, CreateSurface, CreateVolume, LineLoop, SurfaceLoop,
                Extrude, Revolve, Sweep, Union, Union2, MergeFace,
                Intersection, Difference, Fragments, SplitByPlane, Copy, Remove,
                Remove2, RemoveFaces, Move, Rotate,
                Flip, Scale, WorkPlane, WorkPlaneByPoints, WPParallelToPlane,
                WPNormalToPlane,
                healCAD, CADImport, BrepImport,
                Fillet, Chamfer, Array, ArrayRot, ArrayByPoints, ArrayRotByPoints,
                ArrayPath,
                ThruSection, CreateShell, RotateCenterPoints, MoveByPoints, ExtendedLine,
                CreateOffset, CreateOffsetFace, CreateProjection, Simplifiers, MovePoint,
                SplitHairlineFace, CapFaces, ReplaceFaces, Subsequence]

    def get_possible_child_menu(self):
        return [("Geometry Element...", None),
                ("Points...", PointOCC), ("", PointCenter), ("", PointOnEdge),
                ("", PointCircleCenter), ("!", PointByUV),
                ("Lines...", LineOCC), ("!", ExtendedLine),
                ("Polygon...", Polygon2), ("!", OCCPolygon),
                ("Circle...", CircleOCC), ("", CircleByAxisPoint),
                ("", CircleByAxisCenterRadius), ("!", CircleBy3Points),
                ("", Rect),
                ("", Spline),
                ("!", None),
                ("3D shape...", Box),
                ("", Ball), ("", Cone), ("", Wedge), ("", Cylinder),
                ("!", Torus),
                ("Create...", CreateLine), ("", CreateSurface), ("", CreateVolume),
                ("", ThruSection), ("", CreateOffset), ("",
                                                        CreateOffsetFace), ("", CreateShell),
                ("!", CreateProjection),
                ("Protrude...", Extrude), ("", Revolve), ("!", Sweep),
                ("Fillet/Chamfer", Fillet), ("!", Chamfer),
                ("Copy/Remove...", Copy), ("",
                                           Remove), ("", Remove2), ("!", RemoveFaces),
                ("Translate...", Move,), ("", MoveByPoints), ("",
                                                              Rotate), ("", RotateCenterPoints),
                ("", Flip), ("!", Scale),
                ("Array...", Array), ("", ArrayRot), ("",
                                                      ArrayByPoints), ("", ArrayRotByPoints),
                ("!", ArrayPath),
                ("Boolean...", Union), ("", MergeFace), ("", Intersection),
                ("", Difference), ("", Fragments), ("!", SplitByPlane),
                ("WorkPlane...", WorkPlane), ("", WorkPlaneByPoints),
                ("", WPParallelToPlane), ("!", WPNormalToPlane),
                ("Import...", BrepImport), ("", CADImport), ("", healCAD),
                ("Extra(under Dev,)...", Simplifiers), ("", MovePoint),
                ("", CapFaces), ("", ReplaceFaces), ("!", SplitHairlineFace),
                ("!", None),
                ("Subsequence", Subsequence),
                ("!", None),
                ]

    def panel1_param(self):
        ll = super(Subsequence, self).panel1_param()
        from petram.pi.widget_parameters import WidgetParameters
        ll.append([None, None, 99, {"UI": WidgetParameters, "span": (1, 2)}])
        return ll

    def get_panel1_value(self):
        values = super(Subsequence, self).get_panel1_value()
        values.append(self.seq_values)
        return values

    def import_panel1_value(self, v):
        self.seq_values = v[-1]
        v = super(Subsequence, self).import_panel1_value(v[:-1])


data0 = (('center', VtableElement('pts1', type='string',
                                  guilabel='Center point',
                                  default="",
                                  tip="center of WP")),
         ('ax1', VtableElement('ax1', type='string',
                               guilabel='Point on 1st axis',
                               default="",
                               tip="point on the 1st axis")),
         ('ax2', VtableElement('ax2', type='string',
                               guilabel='Point on surface',
                               default="",
                               tip="point on the surface")),
         ('flip1', VtableElement('flip1', type='bool',
                                 guilabel='Flip 1st axis',
                                 default=False,
                                 tip="flip 1st axis")),
         ('flip2', VtableElement('flip2', type='bool',
                                 guilabel='Flip 2nd axis',
                                 default=False,
                                 tip="flip 2nd axis")),
         ('offset', VtableElement('offset', type='float',
                                  guilabel='Offset',
                                  default=0.0,
                                  tip="offset in normal direction")), )


class WorkPlaneByPoints(WPBase):
    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'WP by Points'

    @classmethod
    def fancy_tree_name(self):
        return 'WorkPlane'


data0 = (('surface', VtableElement('surface', type='string',
                                   guilabel='Surface',
                                   default="",
                                   tip="surface to which WP is paralle")),
         ('center', VtableElement('pts1', type='string',
                                  guilabel='Center point',
                                  default="",
                                  tip="center of WP")),
         ('ax1', VtableElement('ax1', type='string',
                               guilabel='Point on 1st axis',
                               default="",
                               tip="point on 1st axis")),
         ('flip1', VtableElement('flip1', type='bool',
                                 guilabel='Flip 1st axis',
                                 default=False,
                                 tip="flip 1st axis")),
         ('flip2', VtableElement('flip2', type='bool',
                                 guilabel='Flip 2nd axis',
                                 default=False,
                                 tip="flip 2nd axis")),
         ('offset', VtableElement('offset', type='float',
                                  guilabel='Offset',
                                  default=0.0,
                                  tip="offset in normal direction")), )


class WPParallelToPlane(WPBase):
    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'Parallel to Plane'

    @classmethod
    def fancy_tree_name(self):
        return 'WorkPlane'


data0 = (('wp_normal', VtableElement_Normal('wp_normal', type='float',
                                            guilabel='normal',
                                            suffix=('x', 'y', 'z'),
                                            default=[0, 0, 1],
                                            tip="wp_normal")),
         ('center', VtableElement('pts1', type='string',
                                  guilabel='Center point',
                                  default="",
                                  tip="center of WP")),
         ('ax1', VtableElement('ax1', type='string',
                               guilabel='Point on 1st axis',
                               default="",
                               tip="point on 1st axis")),
         ('flip1', VtableElement('flip1', type='bool',
                                 guilabel='Flip 1st axis',
                                 default=False,
                                 tip="flip 1st axis")),
         ('flip2', VtableElement('flip2', type='bool',
                                 guilabel='Flip 2nd axis',
                                 default=False,
                                 tip="flip 2nd axis")),
         ('offset', VtableElement('offset', type='float',
                                  guilabel='Offset',
                                  default=0.0,
                                  tip="offset in normal direction")), )


class WPNormalToPlane(WPBase):
    vt = Vtable(data0)

    @classmethod
    def fancy_menu_name(self):
        return 'Normal to Plane'

    @classmethod
    def fancy_tree_name(self):
        return 'WorkPlane'


cad_fix_cb = [None, None, 36, {"col": 4,
                               "labels": ["Degenerated",
                                          "SmallEdges",
                                          "SmallFaces",
                                          "sewFaces",
                                          "makeSolid"]}]
cad_fix_tol = ["Tolerance", 1e-8, 300, None]
cad_fix_rescale = ["Rescale", 1.0, 300, None]
cad_fix_elp0 = [cad_fix_cb, cad_fix_tol, cad_fix_rescale]

cad_fix_elp = [None, None, 27, [
    {"text": "import fixer"}, {"elp": cad_fix_elp0}]]


class ImportBase(GeomPB):
    def attribute_set(self, v):
        v = super(GeomPB, self).attribute_set(v)
        v["use_fix_param"] = [False] * 5
        v["use_fix_tol"] = 1e-8
        v["use_fix_rescale"] = 1.0
        return v

    def get_importfix_value(self):
        value = [self.use_fix_param, self.use_fix_tol, self.use_fix_rescale]
        return value

    def import_importfix_value(self, v):
        self.use_fix_param = [x[1] for x in v[0]]
        self.use_fix_tol = float(v[1])
        self.use_fix_rescale = float(v[2])


class healCAD(ImportBase):
    vt = Vtable(tuple())

    def panel1_param(self):
        from wx import BU_EXACTFIT
        # b1 = {"label": "S", "func": self.onBuildBefore,
        #      "noexpand": True, "style": BU_EXACTFIT}
        b2 = {"label": "R", "func": self.onBuildAfter,
              "noexpand": True, "style": BU_EXACTFIT}
        wc = "ANY|*|STEP|*.stp|IGES|*.igs"

        ll = [[None, None, 241, {'buttons': [b2, ],
                                 'alignright':True,
                                 'noexpand': True}, ],
              ["Entity (v/f/l/p)", "", 0, {}], ]
        ll.extend(cad_fix_elp0)

        return ll

    def attribute_set(self, v):
        v = super(healCAD, self).attribute_set(v)
        v["fix_entity"] = ""
        return v

    def get_panel1_value(self):
        value = self.get_importfix_value()
        return [None, self.fix_entity] + value

    def preprocess_params(self, engine):
        return

    def import_panel1_value(self, v):
        self.fix_entity = str(v[1])
        self.import_importfix_value(v[2:])

    def panel1_tip(self):
        return [None, None, None, None]

    def add_geom_sequence(self, geom):
        gui_name = self.fullname()
        gui_param = (
            self.fix_entity,
            self.use_fix_param,
            self.use_fix_tol,
            self.use_fix_rescale)
        geom_name = self.__class__.__name__
        geom.add_sequence(gui_name, gui_param, geom_name)

    @classmethod
    def fancy_menu_name(self):
        return "fixCAD"

    @classmethod
    def fancy_tree_name(self):
        return "fixCAD"

# we make BREP import separately so that we can add Brep specific
# interface later....


class CADImport(ImportBase):
    vt = Vtable(tuple())

    def panel1_param(self):
        from wx import BU_EXACTFIT
        # b1 = {"label": "S", "func": self.onBuildBefore,
        #      "noexpand": True, "style": BU_EXACTFIT}
        b2 = {"label": "R", "func": self.onBuildAfter,
              "noexpand": True, "style": BU_EXACTFIT}
        wc = "ANY|*|STEP|*.stp|IGES|*.igs"
        ll = [[None, None, 241, {'buttons': [b2, ],
                                 'alignright':True,
                                 'noexpand': True}, ],
              ["File(STEP/IGES)", None, 45, {'wildcard': wc, }],
              cad_fix_elp,
              [None, True, 3, {'text': "Highest Dim Only"}],
              ["Unit", None, 0, {}],
              [None, "Unit: "", M, MM, INCH, KM, CM, FT, MIL, UM...", 2, None],
              ]

        return ll

    def attribute_set(self, v):
        v = super(CADImport, self).attribute_set(v)
        v["cad_file"] = ""
        v["use_fix"] = False
        v["highestdimonly"] = True
        v["import_unit"] = ""
        return v

    def get_panel1_value(self):
        value0 = self.get_importfix_value()
        value = [self.use_fix, value0]
        return [None, self.cad_file, value,
                self.highestdimonly, self.import_unit, None]

    def preprocess_params(self, engine):
        return

    def import_panel1_value(self, v):
        self.cad_file = str(v[1])
        self.use_fix = v[2][0]
        self.import_importfix_value(v[2][1])
        self.highestdimonly = bool(v[3])
        self.import_unit = v[4]

    def panel1_tip(self):
        return [None, None, None]

    def add_geom_sequence(self, geom):
        gui_name = self.fullname()
        gui_param = (self.cad_file, self.use_fix, self.use_fix_param,
                     self.use_fix_tol, self.use_fix_rescale,
                     self.highestdimonly, self.import_unit)
        geom_name = self.__class__.__name__
        geom.add_sequence(gui_name, gui_param, geom_name)

    @classmethod
    def fancy_menu_name(self):
        return "STEP/IGS"

# we make BREP import separately so that we can add Brep specific
# interface later....


class BrepImport(ImportBase):
    vt = Vtable(tuple())

    def panel1_param(self):
        from wx import BU_EXACTFIT
        # b1 = {"label": "S", "func": self.onBuildBefore,
        #      "noexpand": True, "style": BU_EXACTFIT}
        b2 = {"label": "R", "func": self.onBuildAfter,
              "noexpand": True, "style": BU_EXACTFIT}
        wc = "ANY|*|Brep|*.brep"
        ll = [[None, None, 241, {'buttons': [b2],
                                 'alignright':True,
                                 'noexpand': True}, ],
              ["File(.brep)", None, 45, {'wildcard': wc, }],
              cad_fix_elp,
              [None, True, 3, {'text': "Highest Dim Only"}],
              ]
        return ll

    def attribute_set(self, v):
        v = super(BrepImport, self).attribute_set(v)
        v["cad_file"] = ""
        v["use_fix"] = False
        v["highestdimonly"] = True
        return v

    def get_panel1_value(self):
        value0 = self.get_importfix_value()
        value = [self.use_fix, value0]
        return [None, self.cad_file, value, self.highestdimonly]

    def preprocess_params(self, engine):
        return

    def import_panel1_value(self, v):
        self.cad_file = str(v[1])
        self.use_fix = v[2][0]
        self.import_importfix_value(v[2][1])
        self.highestdimonly = bool(v[3])

    def panel1_tip(self):
        return [None, None, None]

    def add_geom_sequence(self, geom):
        gui_name = self.fullname()
        gui_param = (self.cad_file, self.use_fix, self.use_fix_param,
                     self.use_fix_tol, self.use_fix_rescale,
                     self.highestdimonly,)
        geom_name = self.__class__.__name__
        geom.add_sequence(gui_name, gui_param, geom_name)

    @classmethod
    def fancy_menu_name(self):
        return 'Brep'
