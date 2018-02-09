import  petram.mfem_config as mfem_config
mfem_config.use_parallel = False

debug_level = 0

from numpy.lib.type_check import real
from numpy import cos
from numpy import log2
from numpy import log
from numpy.core.multiarray import vdot
from numpy.core.multiarray import dot
from numpy.lib.type_check import imag
from numpy.core.multiarray import array
from numpy import arctan2
from numpy import arctan
from numpy import abs
from numpy import sqrt
from numpy import exp
from numpy.core.numeric import cross
from numpy import tan
from numpy import log10
from numpy import sin
from numpy import conj

from petram.mfem_model import MFEM_ModelRoot
from petram.mfem_model import MFEM_GeneralRoot
from petram.geom.geom_model import MFEM_GeomRoot
from petram.geom.gmsh_geom_model import GmshGeom
from petram.geom.gmsh_primitives import Point
from petram.geom.gmsh_primitives import Polygon
from petram.mfem_model import MFEM_MeshRoot
from petram.mesh.gmsh_mesh_model import GmshMesh
from petram.mesh.gmsh_mesh_actions import CharacteristicLength
from petram.mesh.gmsh_mesh_actions import FreeFace
from petram.mesh.mesh_model import MeshGroup
from petram.mesh.mesh_model import MeshFile
from petram.mesh.mesh_model import UniformRefinement
from petram.mfem_model import MFEM_PhysRoot
from petram.phys.coeff2d.coeff2d_model import Coeff2D
from petram.phys.coeff2d.coeff2d_model import Coeff2D_DefDomain
from petram.phys.coeff2d.coeff2d_domains import Coeff2D_Diffusion
from petram.phys.coeff2d.coeff2d_domains import Coeff2D_Source
from petram.phys.coeff2d.coeff2d_model import Coeff2D_DefBdry
from petram.phys.coeff2d.coeff2d_bdries import Coeff2D_Zero
from petram.phys.coeff2d.coeff2d_model import Coeff2D_DefPoint
from petram.phys.coeff2d.coeff2d_points import Coeff2D_PointSource
from petram.phys.coeff2d.coeff2d_points import Coeff2D_PointValue
from petram.phys.coeff2d.coeff2d_model import Coeff2D_DefPair
from petram.mfem_model import MFEM_InitRoot
from petram.mfem_model import MFEM_SolverRoot
from petram.solver.std_solver_model import StdSolver
from petram.solver.mumps_model import MUMPS
from petram.solver.gmres_model import GMRES

def make_model():
    obj1 = MFEM_ModelRoot()
    obj1.root_path = '/var/folders/pn/_lgs5ycs6vs0y2j8v0y1fw0r0000gn/T/piscope_shiraiwa/pid13991/.###ifigure_Users_shiraiwa_piscope_projects_PetraM_example_gmres.pfz/proj/model1/mfem'
    obj2 = obj1.add_node(name = "General", cls = MFEM_GeneralRoot)
    obj2.ns_name = "global"
    obj3 = obj1.add_node(name = "Geometry", cls = MFEM_GeomRoot)
    obj4 = obj3.add_node(name = "GmshGeom1", cls = GmshGeom)
    obj5 = obj4.add_node(name = "Point1", cls = Point)
    obj5.lcar = '0.0'
    obj6 = obj4.add_node(name = "Polygon1", cls = Polygon)
    obj6.zarr_txt = 'star_z'
    obj6.xarr_txt = 'star_x'
    obj6.lcar = '0.0'
    obj6.zarr = 'star_z'
    obj6.xarr = 'star_x'
    obj6.yarr_txt = 'star_y'
    obj6.yarr = 'star_y'
    obj6.lcar_txt = '0.0'
    obj7 = obj1.add_node(name = "Mesh", cls = MFEM_MeshRoot)
    obj8 = obj7.add_node(name = "GmshMesh1", cls = GmshMesh)
    obj8.clmin = '1.0'
    obj8.geom_group = 'GmshGeom1'
    obj8.clmax = '1.0'
    obj8.clmin_txt = '1.0'
    obj8.clmax_txt = '1.0'
    obj9 = obj8.add_node(name = "CharacteristicLength1", cls = CharacteristicLength)
    obj9.cl = '0.15'
    obj9.geom_id_txt = '1'
    obj9.geom_id = '1'
    obj9.cl_txt = '0.15'
    obj10 = obj8.add_node(name = "FreeFace1", cls = FreeFace)
    obj10.embed_p_txt = '1'
    obj10.embed_p = '1'
    obj10.clmax = '0.1'
    obj10.geom_id_txt = '12'
    obj10.geom_id = '12'
    obj10.clmax_txt = '0.1'
    obj11 = obj7.add_node(name = "MeshGroup1", cls = MeshGroup)
    obj12 = obj11.add_node(name = "MeshFile1", cls = MeshFile)
    obj12.path = 'GmshMesh1.msh'
    obj13 = obj11.add_node(name = "UniformRefinement1", cls = UniformRefinement)
    obj13.num_refine = '1'
    obj14 = obj1.add_node(name = "Phys", cls = MFEM_PhysRoot)
    obj15 = obj14.add_node(name = "Coeff2D1", cls = Coeff2D)
    obj15.dep_vars_base_txt = 'T'
    obj15.sel_index = 'all'
    obj15.order = 2L
    obj16 = obj15.add_node(name = "Domain", cls = Coeff2D_DefDomain)
    obj17 = obj16.add_node(name = "Diffusion1", cls = Coeff2D_Diffusion)
    obj17.c_yy = '1.0'
    obj17.T_init_txt = '0.0'
    obj17.c_yx = '0.0'
    obj17.sel_index = ['1']
    obj17.c_xx = '1.0'
    obj17.c_xy = '0.0'
    obj17.use_m_c = True
    obj17.c_m_txt = '=c_diff'
    obj17.c_m = '=c_diff'
    obj17.T_init = '0.0'
    obj18 = obj16.add_node(name = "Source1", cls = Coeff2D_Source)
    obj18.f = '1.0'
    obj18.enabled = False
    obj18.sel_index = ['1']
    obj18.f_txt = '1.0'
    obj18.T_init = '0.0'
    obj19 = obj15.add_node(name = "Boundary", cls = Coeff2D_DefBdry)
    obj20 = obj19.add_node(name = "Zero1", cls = Coeff2D_Zero)
    obj20.T_init_txt = '0.0'
    obj20.sel_index = ['remaining']
    obj20.T_init = '0.0'
    obj21 = obj15.add_node(name = "Point", cls = Coeff2D_DefPoint)
    obj22 = obj21.add_node(name = "PointSource1", cls = Coeff2D_PointSource)
    obj22.T_init = '0.0'
    obj22.y_delta_txt = '0.0'
    obj22.enabled = False
    obj22.s_delta_txt = '1.0'
    obj22.x_delta = '0.0'
    obj22.sel_index = ['1']
    obj22.x_delta_txt = '0.0'
    obj22.y_delta = '0.0'
    obj22.s_delta = '1.0'
    obj22.T_init_txt = '0.0'
    obj23 = obj21.add_node(name = "PointValue1", cls = Coeff2D_PointValue)
    obj23.y_delta_txt = '[0.3*sin(i/5.*2*pi) for i in range(5)]'
    obj23.T_init_txt = '0.0'
    obj23.value_txt = '[1.0, 1.0, 1.0, 0.5, 0.5]'
    obj23.x_delta = '[0.3*cos(i/5.*2*pi) for i in range(5)]'
    obj23.value = '[1.0, 1.0, 1.0, 0.5, 0.5]'
    obj23.sel_index = 'all'
    obj23.x_delta_txt = '[0.3*cos(i/5.*2*pi) for i in range(5)]'
    obj23.y_delta = '[0.3*sin(i/5.*2*pi) for i in range(5)]'
    obj23.T_init = '0.0'
    obj24 = obj15.add_node(name = "Pair", cls = Coeff2D_DefPair)
    obj25 = obj1.add_node(name = "InitialValue", cls = MFEM_InitRoot)
    obj26 = obj1.add_node(name = "Solver", cls = MFEM_SolverRoot)
    obj27 = obj26.add_node(name = "StdSolver1", cls = StdSolver)
    obj27.phys_model = 'Coeff2D1'
    obj27.assemble_real = True
    obj28 = obj27.add_node(name = "MUMPS1", cls = MUMPS)
    obj28.log_level = 1L
    obj28.enabled = False
    obj28.write_mat = True
    obj29 = obj27.add_node(name = "GMRES1", cls = GMRES)
    obj29.reltol = 1e-10
    obj29.abstol = 1e-06
    obj29.maxiter = 8000L
    obj29.kdim = 200L
    return obj1

if __name__ == "__main__":
    if mfem_config.use_parallel:
        from petram.engine import ParallelEngine as Eng
    else:
        from petram.engine import SerialEngine as Eng
    
    import petram.debug as debug
    debug.set_debug_level(debug_level)
    
    model = make_model()
    
    eng = Eng(model = model)
    
    solvers = eng.preprocess_modeldata()
    
    for s in solvers:
        s.run(eng)