def make_new_sol():
    if not model.has_child('solutions'): model.add_folder('solutions')

    folder = model.solutions.add_folder('sol')
    folder.mk_owndir()
    param = model.param
    param.setvar('sol', '='+folder.get_full_path())
    return folder

ans(make_new_sol())
