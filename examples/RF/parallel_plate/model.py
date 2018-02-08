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
from petram.mfem_model import MFEM_MeshRoot
from petram.mesh.mesh_model import MeshGroup
from petram.mesh.mesh_model import MeshFile
from petram.mesh.mesh_model import UniformRefinement
from petram.mfem_model import MFEM_PhysRoot
from petram.phys.em3d.em3d_model import EM3D
from petram.phys.em3d.em3d_model import EM3D_DefDomain
from petram.phys.em3d.em3d_vac import EM3D_Vac
from petram.phys.em3d.em3d_model import EM3D_DefBdry
from petram.phys.em3d.em3d_pec import EM3D_PEC
from petram.phys.em3d.em3d_port import EM3D_Port
from petram.phys.em3d.em3d_model import EM3D_DefPair
from petram.phys.em3d.em3d_floquet import EM3D_Floquet
from petram.mfem_model import MFEM_InitRoot
from petram.init_model import InitSetting
from petram.mfem_model import MFEM_SolverRoot
from petram.solver.std_solver_model import StdSolver
from petram.solver.mumps_model import MUMPS
from petram.solver.gmres_model import GMRES
from petram.solver.parametric import Parametric

def make_model():
    obj1 = MFEM_ModelRoot()
    obj1.root_path = '/var/folders/pn/_lgs5ycs6vs0y2j8v0y1fw0r0000gn/T/piscope_shiraiwa/pid12345/.###ifigure_Users_shiraiwa_piscope_projects_PetraM_example_simple_waveguide_periodic.pfz/proj/model1/mfem'
    obj2 = obj1.add_node(name = "General", cls = MFEM_GeneralRoot)
    obj2.ns_name = "global"
    obj3 = obj1.add_node(name = "Geometry", cls = MFEM_GeomRoot)
    obj4 = obj1.add_node(name = "Mesh", cls = MFEM_MeshRoot)
    obj5 = obj4.add_node(name = "MeshGroup1", cls = MeshGroup)
    obj6 = obj5.add_node(name = "MeshFile1", cls = MeshFile)
    obj6.path = 'waveguide_hex.mesh'
    obj7 = obj5.add_node(name = "UniformRefinement1", cls = UniformRefinement)
    obj7.num_refine = '1'
    obj7.enabled = False
    obj8 = obj4.add_node(name = "MeshGroup2", cls = MeshGroup)
    obj8.enabled = False
    obj9 = obj1.add_node(name = "Phys", cls = MFEM_PhysRoot)
    obj10 = obj9.add_node(name = "EM3D1", cls = EM3D)
    obj10.freq_txt = '5000000000.0'
    obj10.sel_index = 'all'
    obj11 = obj10.add_node(name = "Domain", cls = EM3D_DefDomain)
    obj11.epsilonr = '1.0'
    obj11.mur = '1.0'
    obj11.Einit_m = '[0, 0, 0]'
    obj11.Einit_y = '0'
    obj11.Einit_x = '0'
    obj11.sel_index = ['remaining']
    obj11.sigma = '0.0'
    obj11.Einit_z = '0'
    obj12 = obj11.add_node(name = "Vac1", cls = EM3D_Vac)
    obj12.epsilonr = '1'
    obj12.use_Einit = True
    obj12.mur = '1.0'
    obj12.Einit_m = '[0, 0, 0]'
    obj12.Einit_y = '1'
    obj12.mur_txt = '1.0'
    obj12.Einit_x = '0'
    obj12.sel_index = ['1']
    obj12.nl_config = [False, {'epsilonr': []}]
    obj12.Einit_y_txt = '1'
    obj12.epsilonr_txt = '1'
    obj12.sigma = '0.0'
    obj12.sigma_txt = '0.0'
    obj12.Einit_z = '0'
    obj13 = obj10.add_node(name = "Boundary", cls = EM3D_DefBdry)
    obj13.Einit_m = '[0, 0, 0]'
    obj13.Einit_y = '0'
    obj13.Einit_x = '0'
    obj13.Einit_z = '0'
    obj14 = obj13.add_node(name = "PEC1", cls = EM3D_PEC)
    obj14.Einit_m = '[0, 0, 0]'
    obj14.sel_index = ['remaining']
    obj14.Einit_y = '0'
    obj14.Einit_x = '0'
    obj14.Einit_z = '0'
    obj15 = obj13.add_node(name = "Port1", cls = EM3D_Port)
    obj15.inc_phase_txt = 'ph'
    obj15.sel_index = ['2']
    obj15.port_idx = '1'
    obj15.mn = [1L, 0L]
    obj15.mode = u'TEM'
    obj16 = obj13.add_node(name = "Port2", cls = EM3D_Port)
    obj16.sel_index = ['5']
    obj16.inc_amp_txt = '0.0'
    obj16.port_idx = '2'
    obj16.mn = [1L, 0L]
    obj16.inc_amp = 0.0
    obj16.mode = u'TEM'
    obj17 = obj10.add_node(name = "Pair", cls = EM3D_DefPair)
    obj18 = obj17.add_node(name = "Floquet1", cls = EM3D_Floquet)
    obj18.src_index = ['4']
    obj18.sel_index = ['3']
    obj19 = obj1.add_node(name = "InitialValue", cls = MFEM_InitRoot)
    obj20 = obj19.add_node(name = "InitSetting1", cls = InitSetting)
    obj20.init_value_txt = '1.0'
    obj20.phys_model = 'EM3D1'
    obj20.init_path = '~/'
    obj21 = obj1.add_node(name = "Solver", cls = MFEM_SolverRoot)
    obj22 = obj21.add_node(name = "StdSolver1", cls = StdSolver)
    obj22.phys_model = 'EM3D1'
    obj22.init_setting = 'InitSetting1'
    obj22.assemble_real = True
    obj23 = obj22.add_node(name = "MUMPS1", cls = MUMPS)
    obj23.log_level = 1L
    obj23.out_of_core = True
    obj23.enabled = False
    obj23.use_blr = True
    obj24 = obj22.add_node(name = "GMRES1", cls = GMRES)
    obj24.reltol = 1e-15
    obj24.abstol = 1e-06
    obj24.maxiter = 10000L
    obj24.kdim = 2000L
    obj25 = obj21.add_node(name = "Parametric1", cls = Parametric)
    obj25.scanner = 'Scan("ph", [0, 120, 240])'
    obj25.save_separate_mesh = False
    obj25.enabled = False
    obj26 = obj25.add_node(name = "StdSolver1", cls = StdSolver)
    obj26.phys_model = 'EM3D1'
    obj27 = obj26.add_node(name = "MUMPS1", cls = MUMPS)
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