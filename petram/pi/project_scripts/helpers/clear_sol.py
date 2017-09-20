def clear_sol(w = proj.app):
      for name, child in model.solutions.get_children():
           child.destroy()
      model.param.delvar('sol')   
      from ifigure.events import SendChangedEvent
      SendChangedEvent(model, w=w)
ans(clear_sol())
