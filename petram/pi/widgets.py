import wx
from ifigure.utils.edit_list import TextCtrlCopyPaste

class TextCtrlCallBack(TextCtrlCopyPaste):
    def __init__(self, *args, **kwargs):
        self._previous_txt = ''
        setting = kwargs.pop('setting', {})
        self._callback = setting.pop('callback_method', None)
        callback_on_enter = setting.pop('callback_on_enter', False)
        args = list(args)
        if len(args) == 1: args.append(wx.ID_ANY)
        if len(args) == 2: args.append('')
        #kwargs['changing_event'] = True
        kwargs['style'] = wx.TE_PROCESS_ENTER
        TextCtrlCopyPaste.__init__(self, *args, **kwargs)

        if callback_on_enter:
            self.Bind(wx.EVT_TEXT_ENTER, self.onEnter)            
        
    def onKillFocus(self, evt):
        TextCtrlCopyPaste.onKillFocus(self, evt)
        if self._callback is not None:
            self._callback(evt)

    def onKeyPressed(self, evt):
        super(TextCtrlCallBack, self).onKeyPressed(evt)
        
    def onEnter(self, evt):
        super(TextCtrlCallBack, self).onEnter(evt)

    '''        
    def onKeyPressed(self, evt):
        print ("onKeyPress")
        TextCtrlCopyPaste.onKeyPressed(self, evt)
        if self._callback is not None:
            self._callback(evt)
    '''
