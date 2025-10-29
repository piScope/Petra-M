from __future__ import print_function

import os
import numpy as np
import time
import tempfile

import six
if six.PY2:
    import cPickle as pickle
else:
    import pickle

from six.moves.queue import Empty as QueueEmpty

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('GeomSequenceOperator')

test_thread = False


class MeshSequenceOperator():
    def __init__(self, **kwargs):
        self.mesh_sequence = []
        #self._prev_sequence = []

    # def __del__(self):
    #    self.terminate_child()

    def clean_queue(self, p):
        if self.use_mp:
            p.task_q.close()
            p.q.close()
            p.task_q.cancel_join_thread()
            p.q.cancel_join_thread()

    def terminate_child(self, p):
        if p.is_alive():
            if self.use_mp:
                self.clean_queue(p)
                p.terminate()
            else:
                p.task_q.put((-1, None))
                p.task_q.join()

    def add(self, name, *gids, **kwargs):
        '''
        add mesh command
        '''

        if name == 'extrude_face':
            self.mesh_sequence.append(['copyface', (gids[1], gids[2]), kwargs])
        elif name == 'revolve_face':

            kwargs['revolve'] = True
            kwargs['volume_hint'] = gids[0]
            self.mesh_sequence.append(['copyface', (gids[1], gids[2]), kwargs])
        else:
            pass
        self.mesh_sequence.append([name, gids, kwargs])

    def count_sequence(self):
        return len(self.mesh_sequence)

    def clear(self):
        self.mesh_sequence = []

    def run_generater(self, brep_input, msh_file, kwargs,
                      finalize=False, dim=3, progressbar=None):
        '''        
        kwargs = {'CharacteristicLengthMax': self.clmax,
                  'CharacteristicLengthMin': self.clmin,
                  'EdgeResolution': self.res,
                  'MeshAlgorithm': self.algorithm,
                  'MeshAlgorithm3D': self.algorithm3d,
                  'MeshAlgorithmR': self.algorithmr,
                  'MaxThreads': self.maxthreads,
                  'use_profiler': self.use_profiler,
                  'use_expert_mode': self.use_expert_mode,
                  'gen_all_phys_entity': self.gen_all_phys_entity,
                  'trash': self.trash,
                  'edge_tss': edge_tss}
        '''
        from petram.mesh.gmsh_mesh_wrapper import (GMSHMeshGenerator,
                                                   GMSHMeshGeneratorTH)

        if progressbar is None or globals()['test_thread']:
            self.use_mp = False
            p = GMSHMeshGeneratorTH()
        else:
            self.use_mp = True
            p = GMSHMeshGenerator()
        p.start()

        args = (brep_input, msh_file, self.mesh_sequence, dim,
                finalize, kwargs)

        p.task_q.put((1, args))

        istep = 0

        while True:
            try:
                ret = p.q.get(True, 1)
                if ret[0]:
                    break
                if progressbar is not None:
                    istep += 1
                    rng = progressbar.GetRange()
                    progressbar.Update(min([istep, rng-1]),
                                       newmsg=ret[1])
                else:
                    print("Mesh Generator : Step = " +
                          str(istep) + " : " + ret[1])

            except QueueEmpty:
                if not p.is_alive():
                    if progressbar is not None:
                        progressbar.Destroy()
                    p.q.close()
                    p.q.cancel_join_thread()
                    assert False, "Child Process Died"
                    break
                time.sleep(1.)
                if progressbar is not None:
                    import wx
                    wx.Yield()
                    if progressbar.WasCancelled():
                        self.terminate_child(p)
                        progressbar.Destroy()
                        assert False, "Mesh Generation Aborted"

            time.sleep(0.01)

        self.terminate_child(p)

        try:
            max_dim, done, msh_output = ret[1]
            assert msh_output is not None, "failed to generate mesh"
            from petram.geom.read_gmsh import read_pts_groups, read_loops

            if progressbar is not None:
                rng = progressbar.GetRange()
                progressbar.Update(min([istep, rng]),
                                   newmsg="Reading mesh file for rendering")
            else:
                print("Reading mesh file for rendering")

            import gmsh
            gmsh.open(msh_output)

            ptx, cells, cell_data = read_pts_groups(gmsh,
                                                    finished_lines=done[1],
                                                    finished_faces=done[2])

            data = ptx, cells, {}, cell_data, {}

        except:
            if progressbar is not None:
                progressbar.Destroy()
            raise

        if progressbar is not None:
            progressbar.Destroy()

        return max_dim, done, data, msh_output
