#
#
#  variables loaded on both client and server sides where wx is not installed
#
#  these parameters are set here, in order to avoid an error when importing wx
#  on a server
#
#

values = ['1', '1', '1', '00:10:00', 'regular(PROJ_19700521)', '', '',
          '', '', "None", 'sbatch', '', '', False, False, ]

keys = ['num_nodes', 'num_cores', 'num_openmp', 'walltime',
        'queue', 'petramver', 'rwdir',
        'log_txt', 'log_keywords', 'notification',
        'cmd', 'adv_opts', 'env_opts', 'skip_mesh',
        'retrieve_files']

def_queues = {'type': 'SLURM',
              'queus': [{'name': 'debug',
                         'maxnode': 1}]}


def get_defaults():
    return values[:], keys[:]


default_remote = {x: y for x, y in zip(keys, values)}

def get_model_remote(param):
    remote = param.getvar('remote')
    if remote is None:
        return None

    for k in default_remote:
        if k not in remote:
            remote[k] = default_remote[k]
    return remote


