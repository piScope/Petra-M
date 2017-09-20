def create_ns(txt):
    ns_folder = model.namespaces
    data_folder = model.datasets
    ns_script = ns_folder.get_child(name = txt+'_ns')
    if ns_script is None:
        ns_folder.onAddNewScript(name=txt+'_ns')
    data = data_folder.get_child(name = txt+'_data')
    if data is None:
        data_folder.add_data(txt+'_data')
    from ifigure.events import SendChangedEvent
    SendChangedEvent(model)

ans(create_ns(*args))
