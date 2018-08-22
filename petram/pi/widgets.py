import wx
from ifigure.utils.edit_list import TextCtrlCopyPaste

class TextCtrlCallBack(TextCtrlCopyPaste):
    def __init__(self, *args, **kwargs):
        self._previous_txt = ''
        setting = kwargs.pop('setting', {})
        self._callback = setting.pop('callback_method', None)
        args = list(args)
        if len(args) == 1: args.append(wx.ID_ANY)
        if len(args) == 2: args.append('')
        #kwargs['changing_event'] = True
        TextCtrlCopyPaste.__init__(self, *args, **kwargs)

    def onKillFocus(self, evt):
        print ("onKillFocus")
        TextCtrlCopyPaste.onKillFocus(self, evt)
        if self._callback is not None:
            self._callback(evt)

    '''        
    def onKeyPressed(self, evt):
        print ("onKeyPress")
        TextCtrlCopyPaste.onKeyPressed(self, evt)
        if self._callback is not None:
            self._callback(evt)
    '''
