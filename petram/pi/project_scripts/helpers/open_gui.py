from petram.mfem_viewer import MFEMViewer

if not obj.get_pymodel().has_child('mfembook'):
    from ifigure.interactive import figure
    v = figure()
    v.book.rename('mfembook')
    v.book.set_keep_data_in_tree(True)
    v.isec(0)
    v.threed('on')
    v.view('noclip')

    v.book.Close()
    v.book.move(obj.get_pymodel())
obj.get_pymodel().mfembook.Open(MFEMViewer)
