'''

   evaluators

    This module provides objects to evaluate solutions
    in various ways. 


      BdrNodalEvaluator : evaulation on boundary surfaces

         eval:
         evalinteg:

      (plan)
      NodalEvaluator : evaulation on volume

         eval:
         evalinteg:


'''

import numpy as np
import six
import weakref
import os
from weakref import WeakKeyDictionary as WKD
from weakref import WeakValueDictionary as WVD


from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem


def evaluator_cls():
    from petram.sol.bdr_nodal_evaluator import BdrNodalEvaluator
    from petram.sol.slice_evaluator import SliceEvaluator
    from petram.sol.edge_nodal_evaluator import EdgeNodalEvaluator
    from petram.sol.ncface_evaluator import NCFaceEvaluator
    from petram.sol.ncedge_evaluator import NCEdgeEvaluator
    from petram.sol.probe_evaluator import ProbeEvaluator
    from petram.sol.pointcloud_evaluator import PointcloudEvaluator
    from petram.sol.integral_evaluator import IntegralEvaluator

    return {'BdrNodal': BdrNodalEvaluator,
            'EdgeNodal': EdgeNodalEvaluator,
            'NCFace':   NCFaceEvaluator,
            'NCEdge':   NCEdgeEvaluator,
            'Slice': SliceEvaluator,
            'Pointcloud': PointcloudEvaluator,
            'Integral': IntegralEvaluator,
            'Probe': ProbeEvaluator, }


class Evaluator(object):
    '''
    Evaluator is a base class. Mostly empty.
    '''

    def __init__(self):
        object.__init__(self)
        self.failed = False

    def make_agents(self, name, params, **kwargs):
        self._agent_params = (name, params)
        self._agent_kwargs = kwargs.copy()

    '''
    check if evauators are build with current need
    '''

    def validate_param(self, params):
        return self._agent_params[1] == params

    def validate_cls(self, name):
        return self._agent_params[0] == name

    def validate_evaluator(self, name, params, *args, **kwargs):
        val = self.validate_param(params) and self.validate_cls(name)
        if not val:
            return False
        # print 'comparing', self._agent_kwargs, kwargs

        for key in six.iterkeys(kwargs):
            if not key in self._agent_kwargs:
                return False

            # print("comparing")
            # print(kwargs[key],self._agent_kwargs[key])
            #print(kwargs[key] != self._agent_kwargs[key])

            chk = (kwargs[key] != self._agent_kwargs[key])
            if isinstance(chk, bool):
                if chk:
                    return False
            else:
                if chk.any():
                    return False
        return True

    def set_model(self, model):
        '''
        single
        MP:    save file in a temprary directoy and send signal
        server: save file, transfer it, send signal
        '''
        raise NotImplementedError('subclass needs to implement this')

    def set_solfiles(self, solfiles):
        '''
        single
        MP:     send signal
        server: send signal
        '''
        raise NotImplementedError('subclass needs to implement this')

    def check_sol_new(self, client_status):
        '''
        return True if it needs to load model file
        on single : use client_status
        on MP     : use client_status
        on server : time stamp of solfile
        '''
        raise NotImplementedError('subclass needs to implement this')

    def load_solfiles(self, mfem_mode=None):
        raise NotImplementedError('subclass needs to implement this')

    def set_phys_path(self, phys_path):
        raise NotImplementedError('subclass needs to implement this')

    def eval(self, expr):
        raise NotImplementedError('subclass needs to implement this')

    def make_probe_agents(self, *args, **kwargs):
        raise NotImplementedError('subclass needs to implement this')


class EvaluatorCommon(Evaluator):
    '''
    EvaluatorCommon provide pertial implementation of methods
    '''

    def set_model(self, model):
        self.mfem_model = weakref.ref(model)

    def load_solfiles(self, mfem_mode=None):
        if self.solfiles is None:
            return

        if self.solfiles in self.solvars:
            return self.solvars[self.solfiles]

        from petram.sol.solsets import Solsets

        solsets = Solsets(self.solfiles)

        print("reading sol variables")
        if self.solfiles.has_parametic_data:
            self.set_parametric_data()

        phys_root = self.mfem_model()["Phys"]
        solvars = phys_root.make_solvars(solsets.set)

        # Setting _emesh_idx from emesh_idx in GridFunction
        for phys in phys_root:
            vnames = phys_root[phys].dep_vars
            gf_var = None
            for name in vnames:
                if name not in solvars[0]:
                    continue
                gf_var = solvars[0][name]
                break

            if gf_var is not None and hasattr(gf_var, "deriv_args"):
                gf_real, gf_imag = gf_var.deriv_args
                eidx = gf_real._emesh_idx if gf_real is not None else gf_imag._emesh_idx
                phys_root[phys]._emesh_idx = eidx
            else:
                # For this variable, correnspoinding GridFunction does not exists.
                # This happens when solution does not exists for a particular variable.
                # We are not plotting this anyway. But we need to set something
                # instead of stopping it here.
                phys_root[phys]._emesh_idx = -1

        pp_root = self.mfem_model()["PostProcess"]
        solvars = pp_root.add_solvars(solsets.set, solvars)
        print("=== solvars= ===", solvars[0])

        self.solvars[self.solfiles] = solvars

        for key in six.iterkeys(self.agents):
            for a, vv in zip(self.agents[key], solsets):
                a.forget_knowns()
        self.solsets = solsets
        return solvars

    def set_parametric_data(self):
        model = self.mfem_model()
        for name, data in self.solfiles.parametric_data:
            parametric = model[name]
            scanner = parametric.get_scanner(nosave=True)
            solvers = parametric.set_scanner_physmodel(scanner)
            scanner.apply_param(data)

    def make_agents(self, name, params, **kwargs):
        if name == 'Probe':
            self.make_probe_agents(name, params, **kwargs)
        elif name == 'Pointcloud':
            self.make_pointcloud_agents(name, params, **kwargs)
        elif name == 'Integral':
            self.make_integral_agents(name, params, **kwargs)
        else:
            self.make_nodalslice_agents(name, params, **kwargs)

    def make_nodalslice_agents(self, name, params, **kwargs):
        print("making new agents", name, params, kwargs)
        super(EvaluatorCommon, self).make_agents(name, params, **kwargs)
        self.agents = {}
        cls = evaluator_cls()[name]
        solsets = self.solsets
        for param in params:
            self.agents[param] = []
            for m, void in solsets:
                a = cls([param], **kwargs)
                a.set_mesh(m)
                self.agents[param].append(a)

    def make_pointcloud_agents(self, name, params, **kwargs):
        print("making new pc agents", name, params, kwargs)
        super(EvaluatorCommon, self).make_agents(name, params, **kwargs)
        self.agents = {}
        cls = evaluator_cls()[name]
        solsets = self.solsets

        params = tuple(params)
        self.agents[params] = []

        for m, void in solsets:
            a = cls(params, **kwargs)
            a.set_mesh(m)
            self.agents[params].append(a)

    def make_integral_agents(self, name, params, **kwargs):
        print("making new integral agents", name, params, kwargs)
        super(EvaluatorCommon, self).make_agents(name, params, **kwargs)
        self.agents = {}
        cls = evaluator_cls()[name]
        solsets = self.solsets

        params = tuple(params)
        self.agents[params] = []

        for m, void in solsets:
            a = cls(params, **kwargs)
            a.set_mesh(m)
            self.agents[params].append(a)

    def make_probe_agents(self, name, params, **kwargs):
        print("making new probe agents", name, params, kwargs)
        super(EvaluatorCommon, self).make_agents(name, params, **kwargs)
        self.agents = {}
        cls = evaluator_cls()[name]
        for param in params:
            self.agents[param] = [cls([param], **kwargs)]

    def eval_probe(self, expr, **kwargs):
        raise NotImplementedError('subclass needs to implement this')
        '''
        phys_path = self.phys_path
        phys = self.mfem_model()[phys_path]
        solvars = self.load_solfiles()
        
        if solvars is None: return None, None
        '''

    def terminate_all(self):
        pass


def_config = {'use_mp': False,
              'use_cs': False,
              'mp_worker': 2,
              'mp_debug': False,
              'cs_worker': 4,
              'cs_server': 'localhost',
              'cs_soldir': '',
              'cs_solsubdir': '',
              'cs_user': ''}


def build_evaluator(params,
                    mfem_model,
                    solfiles,
                    name='BdrNodal',
                    config=def_config,
                    **kwargs):

    from petram.sol.evaluator_mp import EvaluatorMP
    from petram.sol.evaluator_single import EvaluatorSingle
    from petram.sol.evaluator_cs import EvaluatorClient

    from petram.debug import debug_evaluator_mp
    config["mp_debug"] = debug_evaluator_mp

    print("making new evaluator", config, params)

    if not config['use_mp'] and not config['use_cs']:
        evaluator = EvaluatorSingle()
    elif config['use_mp']:
        logfile = 'log' if config['mp_debug'] else False
        evaluator = EvaluatorMP(nproc=config['mp_worker'],
                                logfile=logfile)
    elif config['use_cs']:
        solpath = os.path.join(config['cs_soldir'],
                               config['cs_solsubdir'])
        evaluator = EvaluatorClient(nproc=config['cs_worker'],
                                    host=config['cs_server'],
                                    soldir=solpath,
                                    user=config['cs_user'],
                                    ssh_opts=config['cs_ssh_opts'],
                                    mp_debug=config['mp_debug'])
    else:
        raise ValueError("Unknown evaluator mode")
    evaluator.set_model(mfem_model)

    if name != 'Probe':  # probe evaluator does not load solfiles.
        evaluator.set_solfiles(solfiles)
        evaluator.load_solfiles()

    evaluator.make_agents(name, params, **kwargs)

    return evaluator


def area_tri(vertices):
    # area of triangles
    v = vertices
    p1 = v[:, 0, :]-v[:, 1, :]
    p2 = v[:, 0, :]-v[:, 2, :]
    return (np.sqrt(np.sum(np.cross(p1, p2)**2, 1)))/2.0
