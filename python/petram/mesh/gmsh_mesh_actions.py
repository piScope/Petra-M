from petram.mesh.gmsh_mesh_model import GmshMeshActionBase
from petram.phys.vtable import VtableElement, Vtable
import numpy as np

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('GmshMeshActions')


data = (('geom_id', VtableElement('geom_id', type='string',
                                  guilabel='Line#',
                                  default="remaining",
                                  tip="Line ID")),
        ('num_seg', VtableElement('num_seg', type='int',
                                  guilabel='Number of segments',
                                  default=5,
                                  tip="Number of segments")),
        ('progression', VtableElement('progression', type='float',
                                      guilabel='Progression (> 0)',
                                      default=1.0,
                                      tip="Progression")),
        ('bump', VtableElement('bump', type='float',
                               guilabel='Bump (> 0)',
                               default=1.0,
                               tip="Bump")),)


class TransfiniteLine(GmshMeshActionBase):
    dim = 1
    vt = Vtable(data)

    def add_meshcommand(self, mesher):
        gid, nseg, p, b = self.vt.make_value_or_expression(self)
        gid = self.eval_entity_id(gid)

        mesher.add('transfinite_edge', gid, nseg=nseg,
                   progression=p, bump=b)

    def get_element_selection(self):
        self.vt.preprocess_params(self)
        ret, mode = self.element_selection_empty()
        gid = self.eval_entity_id2(self.geom_id)
        try:
            ret['edge'] = [int(x) for x in gid.split(',')]
        except BaseException:
            pass
        return ret, 'edge'


data = (('geom_id', VtableElement('geom_id', type='string',
                                  guilabel='Surface#',
                                  default="",
                                  tip="Surface ID")),
        ('edge1', VtableElement('edge1', type='string',
                                guilabel='1st corner',
                                default="",
                                tip="1st Edge")),
        ('edge2', VtableElement('edge2', type='string',
                                guilabel='2nd corner',
                                default="",
                                tip="2ndp Edge")),
        ('edge3', VtableElement('edge3', type='string',
                                guilabel='3rd corner',
                                default="",
                                tip="3rd Edge")),
        ('edge4', VtableElement('edge4', type='string',
                                guilabel='4th corner',
                                default="",
                                tip="4th Edge")),)


class TransfiniteSurface(GmshMeshActionBase):
    dim = 2
    vt = Vtable(data)

    def add_meshcommand(self, mesher):
        gid, e1, e2, e3, e4 = self.vt.make_value_or_expression(self)
        gid = self.eval_entity_id(gid)

        c = [int(x) for x in (e1, e2, e3, e4) if x.strip() != '']
        mesher.add('transfinite_surface', gid,
                   corner=c,
                   arrangement=self.transfinite_arrangement)

    def get_element_selection(self):
        self.vt.preprocess_params(self)
        ret, mode = self.element_selection_empty()
        gid = self.eval_entity_id2(self.geom_id)
        try:
            ret['face'] = [int(x) for x in gid.split(',')]
        except BaseException:
            pass
        return ret, 'face'

    def attribute_set(self, v):
        v = super(TransfiniteSurface, self).attribute_set(v)
        self.vt.attribute_set(v)
        v["transfinite_arrangement"] = "Left"
        return v

    def panel1_param(self):
        import wx
        transfinite_arrangement_cb = [None, "Left", 4,
                                      {"style": wx.CB_READONLY,
                                       "choices": ["Left",
                                                   "Right",
                                                   "AlternateLeft",
                                                   "AlternateRight"]}]

        ll = super(TransfiniteSurface, self).panel1_param()
        ll.append(transfinite_arrangement_cb)
        return ll

    def get_panel1_value(self):
        v = super(TransfiniteSurface, self).get_panel1_value()
        return v + [self.transfinite_arrangement]

    def preprocess_params(self, engine):
        self.vt.preprocess_params(self)
        return

    def import_panel1_value(self, v):
        super(TransfiniteSurface, self).import_panel1_value(v[:-1])
        self.transfinite_arrangement = v[-1]

    def panel1_tip(self):
        tip = super(TransfiniteSurface, self).panel1_tip()
        return tip + ['arrangement of triangles', ]


data = (('geom_id', VtableElement('geom_id', type='string',
                                  guilabel='Volume#',
                                  default="",
                                  tip="Surface ID")),
        ('corners', VtableElement('corners', type='string',
                                  guilabel='Corners (1/6/8pts)',
                                  default="",
                                  tip="Corners (1 or 6 or 8 points)")),)


class TransfiniteVolume(GmshMeshActionBase):
    dim = 3
    vt = Vtable(data)

    def add_meshcommand(self, mesher):
        gid, pts = self.vt.make_value_or_expression(self)
        gid = self.eval_entity_id(gid)

        c = [int(x) for x in pts.split(',') if x.strip() != '']
        mesher.add('transfinite_volume', gid, corner=c)

    def get_element_selection(self):
        self.vt.preprocess_params(self)
        ret, mode = self.element_selection_empty()
        gid = self.eval_entity_id2(self.geom_id)
        try:
            ret['volume'] = [int(x) for x in gid.split(',')]
        except BaseException:
            pass
        return ret, 'volume'


data = (('geom_id', VtableElement('geom_id', type='string',
                                  guilabel='Point#',
                                  default="remaining",
                                  tip="Point ID")),
        ('cl', VtableElement('cl', type='float',
                             guilabel='Size)',
                             default=1.0,
                             tip="CharacteristicLength")))


class CharacteristicLength(GmshMeshActionBase):
    dim = 0
    vt = Vtable(data)

    def add_meshcommand(self, mesher):
        gid, cl = self.vt.make_value_or_expression(self)
        gid = self.eval_entity_id(gid)
        mesher.add('cl', gid, cl)

    def get_element_selection(self):
        self.vt.preprocess_params(self)
        ret, mode = self.element_selection_empty()
        gid = self.eval_entity_id2(self.geom_id)
        try:
            ret['point'] = [int(x) for x in gid.split(',')]
        except BaseException:
            pass
        return ret, 'point'


data = (('geom_id', VtableElement('geom_id', type='string',
                                  guilabel='Element#',
                                  default="remaining",
                                  tip="Element ID")),
        ('clmax', VtableElement('clmax', type='float',
                                guilabel='Max size)',
                                default_txt='',
                                default=0.,
                                tip="CharacteristicLengthMax")),
        ('clmin', VtableElement('clmin', type='float',
                                guilabel='Min size',
                                default_txt='',
                                default=0.0,
                                tip="CharacteristicLengthMin")),
        ('resolution', VtableElement('resolution', type='int',
                                     guilabel='Resolution',
                                     default=5.,
                                     tip="Edge Resolution")),
        ('growth', VtableElement('growth', type='float',
                                 guilabel='Growth',
                                 default=1.0,
                                 tip="Mesh size growth")),
        ('embed_s', VtableElement('embed_s', type='string',
                                  guilabel='Surface#',
                                  default="",
                                  tip="Surface number")),
        ('embed_l', VtableElement('embed_l', type='string',
                                  guilabel='Line#',
                                  default="",
                                  tip="Line number")),
        ('embed_p', VtableElement('embed_p', type='string',
                                  guilabel='Point#',
                                  default="",
                                  tip="Point number")),)


class FreeVolume(GmshMeshActionBase):
    dim = 3
    vt = Vtable(data)

    def add_meshcommand(self, mesher):
        values = self.vt.make_value_or_expression(self)
        gid, clmax, clmin, res, growth, embed_s, embed_l, embed_p = values
        gid, embed_s, embed_l, embed_p = self.eval_entity_id(
            gid, embed_s, embed_l, embed_p)

        mesher.add('freevolume', gid,
                   maxsize=clmax,
                   minsize=clmin,
                   resolution=res,
                   sizegrowth=growth,
                   embed_s=embed_s,
                   embed_l=embed_l,
                   embed_p=embed_p,
                   alg2d=self.alg_2d,
                   alg3d=self.alg_3d)

    def get_element_selection(self):
        self.vt.preprocess_params(self)
        ret, mode = self.element_selection_empty()
        gid = self.eval_entity_id2(self.geom_id)
        try:
            ret['volume'] = [int(x) for x in gid.split(',')]
        except BaseException:
            pass
        return ret, 'volume'

    def attribute_set(self, v):
        v = super(FreeVolume, self).attribute_set(v)
        self.vt.attribute_set(v)
        v["alg_2d"] = "default"
        v["alg_3d"] = "default"
        return v

    def panel1_param(self):
        from petram.mesh.gmsh_mesh_wrapper import Algorithm2D, Algorithm3D

        c1 = list(Algorithm2D)
        c2 = list(Algorithm3D)

        from wx import CB_READONLY
        setting1 = {"style": CB_READONLY, "choices": c1}
        setting2 = {"style": CB_READONLY, "choices": c2}

        ll = super(FreeVolume, self).panel1_param()
        ll.extend([["2D Algorithm", c1[-1], 4, setting1],
                   ["3D Algorithm", c2[-1], 4, setting2], ])
        return ll

    def get_panel1_value(self):
        v = super(FreeVolume, self).get_panel1_value()
        return v + [self.alg_2d, self.alg_3d]

    def preprocess_params(self, engine):
        self.vt.preprocess_params(self)
        return

    def import_panel1_value(self, v):
        super(FreeVolume, self).import_panel1_value(v[:-2])
        self.alg_2d = v[-2]
        self.alg_3d = v[-1]

    def panel1_tip(self):
        tip = super(FreeVolume, self).panel1_tip()
        return tip + ['mesh algorithm for surface',
                      'mesh algorithm for volume']


data = (('geom_id', VtableElement('geom_id', type='string',
                                  guilabel='Element#',
                                  default="remaining",
                                  tip="Element ID")),
        ('clmax', VtableElement('clmax', type='float',
                                guilabel='Max size)',
                                default_txt='',
                                default=0.,
                                tip="CharacteristicLengthMax")),
        ('clmin', VtableElement('clmin', type='float',
                                guilabel='Min size',
                                default=0.,
                                tip="CharacteristicLengthMin")),
        ('resolution', VtableElement('resolution', type='int',
                                     guilabel='Resolution',
                                     default=5.,
                                     tip="Edge Resolution")),
        ('growth', VtableElement('growth', type='float',
                                 guilabel='Growth',
                                 default=1.0,
                                 tip="Mesh size growth")),
        ('embed_l', VtableElement('embed_l', type='string',
                                  guilabel='Line#',
                                  default="",
                                  tip="Line number")),
        ('embed_p', VtableElement('embed_p', type='string',
                                  guilabel='Point#',
                                  default="",
                                  tip="Point number")),)


class FreeFace(GmshMeshActionBase):
    dim = 2
    vt = Vtable(data)

    def add_meshcommand(self, mesher):
        gid, clmax, clmin, res, growth, embed_l, embed_p = self.vt.make_value_or_expression(
            self)
        gid, embed_l, embed_p = self.eval_entity_id(gid, embed_l, embed_p)
        mesher.add('freeface', gid,
                   maxsize=clmax,
                   minsize=clmin,
                   resolution=res,
                   sizegrowth=growth,
                   embed_l=embed_l,
                   embed_p=embed_p,
                   alg2d=self.alg_2d)

    def get_element_selection(self):
        self.vt.preprocess_params(self)
        ret, mode = self.element_selection_empty()
        gid = self.eval_entity_id2(self.geom_id)
        try:
            ret['face'] = [int(x) for x in gid.split(',')]
        except BaseException:
            pass
        return ret, 'face'

    def get_embed(self):
        gid, clmax, clmin, embed_l, embed_p = self.vt.make_value_or_expression(
            self)
        ll = [str(x) for x in embed_l.split(',')]
        pp = [str(x) for x in embed_p.split(',')]
        return [], ll, pp

    def attribute_set(self, v):
        v = super(FreeFace, self).attribute_set(v)
        self.vt.attribute_set(v)
        v["alg_2d"] = "default"
        return v

    def panel1_param(self):
        from petram.mesh.gmsh_mesh_wrapper import Algorithm2D

        c1 = list(Algorithm2D)

        from wx import CB_READONLY
        setting1 = {"style": CB_READONLY, "choices": c1}

        ll = super(FreeFace, self).panel1_param()
        ll.extend([["2D Algorithm", c1[-1], 4, setting1], ])

        return ll

    def get_panel1_value(self):
        v = super(FreeFace, self).get_panel1_value()
        return v + [self.alg_2d, ]

    def preprocess_params(self, engine):
        self.vt.preprocess_params(self)
        return

    def import_panel1_value(self, v):
        super(FreeFace, self).import_panel1_value(v[:-1])
        self.alg_2d = v[-1]

    def panel1_tip(self):
        tip = super(FreeFace, self).panel1_tip()
        return tip + ['mesh algorithm for surface']


data = (('geom_id', VtableElement('geom_id', type='string',
                                  guilabel='Line#',
                                  default="remaining",
                                  tip="Line number")),
        ('clmax', VtableElement('clmax', type='float',
                                guilabel='Max size',
                                default_txt='',
                                default=0.0,
                                tip="CharacteristicLengthMax")),
        ('clmin', VtableElement('clmin', type='float',
                                guilabel='Min size',
                                default_txt='', default=0.0,
                                tip="CharacteristicLengthMin")),
        ('resolution', VtableElement('resolution', type='int',
                                     guilabel='Resolution',
                                     default_txt='5',
                                     default=5.,
                                     tip="Edge Resolution")),)


class FreeEdge(GmshMeshActionBase):
    dim = 1
    vt = Vtable(data)

    def add_meshcommand(self, mesher):
        gid, clmax, clmin, res = self.vt.make_value_or_expression(self)
        gid = self.eval_entity_id(gid)

        mesher.add('freeedge', gid,
                   maxsize=clmax,
                   minsize=clmin,
                   resolution=res)

    def get_element_selection(self):
        self.vt.preprocess_params(self)
        ret, mode = self.element_selection_empty()
        gid = self.eval_entity_id2(self.geom_id)
        try:
            ret['edge'] = [int(x) for x in gid.split(',')]
        except BaseException:
            pass
        return ret, 'edge'

# Transform Hint (extrude)
#    d     : dx, dy, dz : 3 float
#    l1, l2,,,,: set of edges : end points determines d (N.I.)


def process_hint_ex(text):
    try:
        text = str(text)
        values = [x.strip() for x in text.split(',') if len(x.strip()) != 0]
        values = [float(x) for x in values]
        if len(values) == 3:
            return {'axan': (values, 0.0)}
        else:
            assert False, "enter direction of translation"
    except BaseException:
        pass
    return {}

# Transform Hint (revolve)
#    ax an : ax_x, ax_y, ax_z, angle(deg): 4 float
#    l1, angle  : l1 direction of axis, angle (deg) (N.I.)
#    s1, angle  : normal to face s1, angle (deg) (N.I.)


def process_hint_rv(text):
    try:
        text = str(text)
        values = [x.strip() for x in text.split(',') if len(x.strip()) != 0]
        values = [float(x) for x in values]
        if len(values) == 4:
            return {'axan': (values[0:3], values[-1])}
    except BaseException:
        pass
    return {}


data = (('geom_id', VtableElement('geom_id', type='string',
                                  guilabel='Surface# (To)',
                                  default="",
                                  tip="Surface number")),
        ('src_id', VtableElement('src_id', type='string',
                                 guilabel='Source # (From)',
                                 default="",
                                 tip="Surface number")),
        ('mapper', VtableElement('mapper', type='string',
                                 guilabel='Transform Hint',
                                 default="",
                                 tip="Coordinate transformatin ")),
        ('cp_cl', VtableElement('cp_cl', type='bool',
                                guilabel='Copy CL',
                                default=True,
                                tip="Copy Characteristic Length")), )


class CopyFace(GmshMeshActionBase):
    dim = 2
    vt = Vtable(data)

    def add_meshcommand(self, mesher):
        gid, src_id, hint, cp_cl = self.vt.make_value_or_expression(self)
        gid = self.eval_entity_id(gid)

        kwargs = process_hint_ex(hint)
        kwargs['copy_cl'] = cp_cl
        print('adding here', src_id, gid,)
        mesher.add('copyface', src_id, gid,
                   **kwargs)

    def get_element_selection(self):
        self.vt.preprocess_params(self)
        ret, mode = self.element_selection_empty()
        gid = self.eval_entity_id(self.geom_id)
        sid = self.eval_entity_id(self.src_id)
        try:
            dest = [int(x) for x in gid.split(',')]
            src = [int(x) for x in sid.split(',')]
            ret['face'] = dest + src
        except BaseException:
            pass
        return ret, 'face'


class CopyFaceRotate(GmshMeshActionBase):
    dim = 2
    vt = Vtable(data)

    def add_meshcommand(self, mesher):
        gid, src_id, hint, cp_cl = self.vt.make_value_or_expression(self)
        gid = self.eval_entity_id(gid)

        kwargs = process_hint_rv(hint)
        kwargs['copy_cl'] = cp_cl
        kwargs['revolve'] = True

        mesher.add('copyface', src_id, gid,
                   **kwargs)

    def get_element_selection(self):
        self.vt.preprocess_params(self)
        ret, mode = self.element_selection_empty()
        gid = self.eval_entity_id(self.geom_id)
        sid = self.eval_entity_id(self.src_id)

        try:
            dest = [int(x) for x in gid.split(',')]
            src = [int(x) for x in sid.split(',')]
            ret['face'] = dest + src
        except BaseException:
            pass
        return ret, 'face'


data = (('geom_id', VtableElement('geom_id', type='string',
                                  guilabel='Curves',
                                  default="",
                                  tip="Curves to generate boundary layer")),
        ('surface_id', VtableElement('surface_id', type='string',
                                     guilabel='Surfaces',
                                     default="auto",
                                     tip="Surfaces where boundary layer is generated")),
        ('thickness', VtableElement('thckness', type='float',
                                    guilabel='Thickness',
                                    default_txt="0.1",
                                    default=0.1,
                                    tip="total thickness of layer")),
        ('growth', VtableElement('growth', type='float',
                                 guilabel='Growth',
                                 default_txt="1.1",
                                 default=0.1,
                                 tip="growth of layer thickness")),
        ('nlayer', VtableElement('nlayer', type='int',
                                 guilabel='# of layer',
                                 default_txt='5',
                                 default=5.,
                                 tip="number of boundary layer")),
        ('fanpoint', VtableElement('fanpoint', type='string',
                                   guilabel='Fan points',
                                   tip="number of boundary layer")),
        ('fanpointsizes', VtableElement('fanpointsizes', type='string',
                                        guilabel='# of size at fan points',
                                        tip="number of element for fan points")),
        ('use_quad', VtableElement('use_quad', type='bool',
                                   guilabel='Quad element',
                                   default=False,
                                   tip="Generate boundary layer using quad elements")), )


class BoundaryLayer(GmshMeshActionBase):
    vt = Vtable(data)

    def add_meshcommand(self, mesher):
        gid, sid, thickness, growth, nlayer, fanpoints, fanpointssize, use_quad = self.vt.make_value_or_expression(
            self)
        gid = self.eval_entity_id(gid)
        sid = self.eval_entity_id(sid)

        fanpoints = self.eval_entity_id(fanpoints)
        fanpointssize = [int(x)
                         for x in fanpointssize.split(',') if len(x) > 0]

        mesher.add('boundary_layer', gid, sid,
                   thickness=thickness,
                   growth=growth,
                   nlayer=nlayer,
                   fanpoints=fanpoints,
                   fanpointssize=fanpointssize,
                   use_quad=use_quad)

    def get_element_selection(self):
        self.vt.preprocess_params(self)
        ret, mode = self.element_selection_empty()
        gid = self.eval_entity_id2(self.geom_id)
        try:
            ret['edge'] = [int(x) for x in gid.split(',')]
        except BaseException:
            pass
        return ret, 'edge'


merge_loc = [None, None, 36, {"col": 4,
                              "labels": ["0D", "1D", "2D", "3D"]}]


class MergeText(GmshMeshActionBase):
    vt = Vtable(tuple())

    def panel1_param(self):
        ll = super(MergeText, self).panel1_param()

        ll.extend([["Text", "", 235, {"nlines": 15}],
                   merge_loc, ])

        return ll

    def attribute_set(self, v):
        v = super(MergeText, self).attribute_set(v)
        v["merge_txt"] = ""
        v["merge_dim"] = [True] + [False] * 3
        return v

    def get_panel1_value(self):
        v = super(MergeText, self).get_panel1_value()
        v.extend([self.merge_txt, self.merge_dim])
        return v

    def preprocess_params(self, engine):
        return

    def import_panel1_value(self, v):
        super(MergeText, self).import_panel1_value(v[:-2])
        self.merge_txt = str(v[-2])
        self.merge_dim = [x[1] for x in v[-1]]

    def panel1_tip(self):
        return [None, None, None, None]

    def add_meshcommand(self, mesher):
        self.vt.preprocess_params(self)
        mesher.add('mergetxt', text=self.merge_txt, dim=self.merge_dim)


data = (('geom_id', VtableElement('geom_id', type='string',
                                  guilabel='Surfaces',
                                  default="",
                                  tip="Entity number")),)


class CompoundSurface(GmshMeshActionBase):
    dim = -1  # this element does not make mesh
    vt = Vtable(data)

    def add_meshcommand(self, mesher):
        gid = self.vt.make_value_or_expression(self)[0]
        gid = self.eval_entity_id(gid)

        # generate something like... Compound Surface{1, 5, 10};
        text = "Compound Surface{ " + gid + "};"
        mesher.add('mergetxt', text=text, dim=[True, False, False, False])


data = (('geom_id', VtableElement('geom_id', type='string',
                                  guilabel='Curves',
                                  default="",
                                  tip="Entity number")),)


class CompoundCurve(GmshMeshActionBase):
    dim = -1  # this element does not make mesh
    vt = Vtable(data)

    def add_meshcommand(self, mesher):
        gid = self.vt.make_value_or_expression(self)[0]
        gid = self.eval_entity_id(gid)

        # generate something like... Compound Curve{1, 5, 10};
        text = "Compound Curve{ " + gid + "};"
        mesher.add('mergetxt', text=text, dim=[True, False, False, False])


rsdata = (('geom_id', VtableElement('geom_id', type='string',
                                    guilabel='Surfaces',
                                    default="",
                                    tip="surfacess to be recombined")), )
#           ('max_angle', VtableElement('max_angle', type='float',
#                                guilabel = 'Max size)',
#                                default = 45,
#                                tip = "Maximum differend of angle" )),)


class RecombineSurface(GmshMeshActionBase):
    dim = -1  # this element does not make mesh
    vt = Vtable(rsdata)

    def add_meshcommand(self, mesher):
        gid = self.vt.make_value_or_expression(self)[0]
        gid = self.eval_entity_id(gid)

        mesher.add('recombine_surface', gid)

    def get_element_selection(self):
        self.vt.preprocess_params(self)
        ret, mode = self.element_selection_empty()
        gid = self.eval_entity_id(self.geom_id)
        try:
            dest = [int(x) for x in gid.split(',')]
            ret['face'] = dest
        except BaseException:
            pass
        return ret, 'face'


edata = (('ex_target', VtableElement('ex_target', type='string',
                                     guilabel='Volume',
                                     default="",
                                     tip="extrusion target")),
         ('dst_id', VtableElement('dst_id', type='string',
                                  guilabel='Surface# (To)',
                                  default="",
                                  tip="Surface number")),
         ('src_id', VtableElement('src_id', type='string',
                                  guilabel='Source # (From)',
                                  default="",
                                  tip="Surface number")),
         ('nlayer', VtableElement('nlayer', type='int',
                                  guilabel='Num. Layers',
                                  default=5,
                                  tip="Number of Layers")),
         ('mapper', VtableElement('mapper', type='string',
                                  guilabel='Transform Hint',
                                  default="",
                                  tip="Coordinate transformatin (ax, an), (dx,dy,dz), ")),
         ('use_recombine', VtableElement('use_recombine', type='bool',
                                         guilabel='Recombine(Hex/Prism)',
                                         default=False,
                                         tip="recombine extruded mesh to Hex/Prism")), )


class ExtrudeMesh(GmshMeshActionBase):
    dim = (2, 3)
    vt = Vtable(edata)

    def add_meshcommand(self, mesher):
        gid, dst_id, src_id, nlayers, hint, use_recombine = self.vt.make_value_or_expression(
            self)
        gid, dst_id, src_id = self.eval_entity_id(gid, dst_id, src_id)

        kwargs = process_hint_ex(hint)
        mesher.add('extrude_face', gid, src_id, dst_id,
                   nlayers=nlayers, use_recombine=use_recombine, **kwargs)

    def get_element_selection(self):
        self.vt.preprocess_params(self)
        ret, mode = self.element_selection_empty()
        did = self.eval_entity_id(self.dst_id)
        sid = self.eval_entity_id(self.src_id)

        try:
            dest = [int(x) for x in did.split(',')]
            src = [int(x) for x in sid.split(',')]
            ret['face'] = dest + src
        except BaseException:
            pass
        return ret, 'face'


class RevolveMesh(GmshMeshActionBase):
    dim = (2, 3)
    vt = Vtable(edata)

    def add_meshcommand(self, mesher):
        gid, dst_id, src_id, nlayers, hint, use_recombine = self.vt.make_value_or_expression(
            self)
        gid, dst_id, src_id = self.eval_entity_id(gid, dst_id, src_id)

        kwargs = process_hint_rv(hint)
        mesher.add('revolve_face', gid, src_id, dst_id,
                   nlayers=nlayers, use_recombine=use_recombine, **kwargs)

    def get_element_selection(self):
        self.vt.preprocess_params(self)
        ret, mode = self.element_selection_empty()
        did = self.eval_entity_id(self.dst_id)
        sid = self.eval_entity_id(self.src_id)

        try:
            dest = [int(x) for x in did.split(',')]
            src = [int(x) for x in sid.split(',')]
            ret['face'] = dest + src
        except BaseException:
            pass
        return ret, 'face'
