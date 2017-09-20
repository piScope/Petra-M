def init_dataset(nsname, datasetnames):
    data = model.datasets.get_child(name = nsname + '_data')
    for n in datasetnames:
           if not data.hasvar(n): 
                data.setvar(n, None)
ans(init_dataset(*args))