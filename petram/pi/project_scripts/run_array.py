from petram.pi.run_petram import run_parallel

import os

values = kwargs.pop("values")
folder = kwargs.pop("folder")

for array_id in values:
    path = os.path.join(folder.owndir(), 'case_'+str(array_id))
    os.mkdir(path)
    kwargs["path"] = path
    kwargs["array_id"] = array_id

    run_parallel(model, *args, **kwargs)

ans()
