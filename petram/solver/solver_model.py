from petram.model import Model
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('Solver')

class Solver(Model):
    def get_phys(self):
        names = self.phys_model.split(',')
        names = [n.strip() for n in names if n.strip() != '']        
        return [self.root()['Phys'][n] for n in names]
    def get_init_setting(self):
        names = self.init_setting.split(',')
        names = [n.strip() for n in names if n.strip() != '']        
        return [self.root()['InitialValue'][n] for n in names]
    
    def assemble(self, engine):
        raise NotImplementedError(
             "you must specify this method in subclass")
        
    def run(self, engine):
        raise NotImplementedError(
             "you must specify this method in subclass")
    
    def postprocess(self, engine):
        raise NotImplementedError(
             "you must specify this method in subclass")

    def set_parameters(self, names, params):
        raise NotImplementedError(
             "you must specify this method in subclass")
    
    def get_matrix_weight(self, timestep_config, timestep_weight):
        raise NotImplementedError(
             "you must specify this method in subclass")
        
    def compute_A_rhs(self, M, B, X):
        '''
        called from an engine to compute linear system from matrices/solutions.
        '''
        raise NotImplementedError(
             "you must specify this method in subclass")
        
    def get_active_solver(self, mm = None):
        for x in self.iter_enabled(): return x

    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('phys', self)
        
