import os
import wx
import sys
import subprocess as sp

from ifigure.utils.edit_list import EditListPanel, EDITLIST_CHANGED
from ifigure.utils.cbook import BuildPopUpMenu

font_h = None
font_w = None
font = None
font_label = None


def set_default_font():
    size = 12
    font = wx.Font(pointSize=size, family=wx.DEFAULT,
                   style=wx.NORMAL,  weight=wx.NORMAL,
                   faceName='Consolas')
    globals()['font_label'] = wx.Font(pointSize=size, family=wx.DEFAULT,
                                      style=wx.NORMAL,  weight=wx.BOLD,
                                      faceName='Consolas')
    dc = wx.ScreenDC()
    dc.SetFont(font)
    w, h = dc.GetTextExtent('A')
    globals()['font_h'] = h*1.5
    globals()['font_w'] = w
    globals()['font'] = font


def install_from_github(url, update=False, parent=None):
    """
    Installs a Python library from a GitHub repo.

    """
    if not url.startswith("git+"):
        url = "git+" + url
    return pip_install(url, update=update, parent=parent)


def pip_uninstall(package):
    command = [sys.executable, "-m", "pip", "uninstall", "-y", package]
    try:
        # python -m pip install
        sp.check_call(command,
                      # stdout=sp.PIPE,
                      # stderr=sp.STDOUT,
                      )
    except sp.CalledProcessError as e:
        print(f"Uninstallation failed. Error: {e}")
        if e.output is not None:
            print(f"Output: {e.output.decode()}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def pip_install(url, update=False, parent=None):
    pip_args = run_package_env_checker(url, parent=parent)
    if pip_args is None:
        return False
    command = [sys.executable, "-m", "pip", "install"]
    if update:
        command.append("-U")
    command.extend(pip_args)
    command.append(url)
    try:
        # python -m pip install
        sp.check_call(command,
                      # stdout=sp.PIPE,
                      # stderr=sp.STDOUT,
                      )
        return True
    except sp.CalledProcessError as e:
        print(f"Installation failed. Error: {e}")
        if e.output is not None:
            print(f"Output: {e.output.decode()}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return False


class pkglist_popup(wx.Menu):
    def __init__(self, parent):
        super(pkglist_popup, self).__init__()
        self.parent = parent
        pkg = parent.packages[parent.selected_row]

        self.target_pkg = pkg
        if pkg["html_url"] == "?":

            menus = [('Uninstall', self.onUninstall, None),
                     ('Recheck Repos...', self.onRecheck, None),
                     ('** can not update from here ** ', None, None),]
        elif pkg["latest"] == "?":
            menus = [('Uninstall', self.onUninstall, None),
                     ('Recheck Repos...', self.onRecheck, None),
                     ('** can not update from here ** ', None, None),]
        elif pkg["installed"] == "no":
            menus = [('Install', self.onInstall, None),
                     ('Recheck Repos...', self.onRecheck, None),]
        else:
            menus = [('Update', self.onUpdate, None),
                     ('Uninstall', self.onUninstall, None),
                     ('Recheck Repos...', self.onRecheck, None),]

        BuildPopUpMenu(self, menus)

    def onInstall(self, evt):
        if install_from_github(self.target_pkg["html_url"], parent=self.parent):
            self.parent.update_done = True
            self.parent.do_recheck()

    def onUpdate(self, evt):
        if install_from_github(self.target_pkg["html_url"], update=True, parent=self.parent):
            self.parent.update_done = True
            self.parent.do_recheck()

    def onUninstall(self, evt):
        pip_uninstall(self.target_pkg["module"])
        self.parent.do_recheck()

    def onRecheck(self, evt):
        self.parent.do_recheck()


class repolist_popup(wx.Menu):
    def __init__(self, parent):
        super(repolist_popup, self).__init__()
        self.parent = parent

        k = self.parent.selected_row
        if k < 0:
            menus = [('Adds...', parent.onAddRepo, None),]
        else:
            k = self.parent.selected_row
            name = self.parent.grid.GetCellValue(k, 0)
            if name.strip() == "":
                menus = [('Adds...', parent.onAddRepo, None),]
            else:
                menus = [('Edit...', parent.onEditRepo, None),
                         ('Adds...', parent.onAddRepo, None),
                         ('Remove', parent.onRmRepo, None),]

        BuildPopUpMenu(self, menus)


class dlg_repos(wx.Dialog):
    def __init__(self, parent, repos, id=wx.ID_ANY, title='repositories'):
        set_default_font()

        wx.Dialog.__init__(self, parent, wx.ID_ANY, title,
                           style=wx.STAY_ON_TOP | wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)

        self.repos = ([x['name'] for x in repos],
                      [x['url'] for x in repos],
                      ['github' if 'protocol' not in x else x['protocol'] for x in repos],)

        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)

        self.grid = wx.grid.Grid(self)
        self.grid.CreateGrid(10, 3)
        self.grid.HideRowLabels()
        self.grid.SetSelectionMode(wx.grid.Grid.SelectRows)
        self.grid.SetDefaultCellFont(font)
        self.grid.SetLabelFont(font_label)
        self.grid.SetColLabelSize(int(font_h))
        self.grid.SetDefaultRowSize(int(font_h), True)
        self.grid.EnableDragColSize(True)
        self.grid.SetColLabelValue(0, "name")
        self.grid.SetColLabelValue(1, "url")
        self.grid.SetColLabelValue(2, "protocol")

        self.grid.EnableEditing(False)

        sizer.Add(self.grid, 1, wx.EXPAND, 0)

        # buttons
        sizer0 = wx.BoxSizer(wx.HORIZONTAL)
        okbutton = wx.Button(self, wx.ID_OK, "OK")
        sizer0.AddStretchSpacer()
        sizer0.Add(okbutton, 0, wx.ALIGN_CENTER | wx.ALL, 1)
        okbutton.Bind(wx.EVT_BUTTON, self.onOK)

        sizer.Add(sizer0, 0, wx.EXPAND | wx.ALL, 10)

        # if add_palette:
        wx.GetApp().add_palette(self)

        #
        self.fill_grid()
        #
        self.grid.Bind(wx.grid.EVT_GRID_SELECT_CELL, self.onCellSelected)
        self.grid.Bind(wx.grid.EVT_GRID_CELL_RIGHT_CLICK, self.onRightRelease)
        #
        self.selected_row = -1
        self.update_done = False

        self.Show()
        wx.CallAfter(self._myRefresh)

    def fill_grid(self):
        min_len = 3
        nrow = self.grid.GetNumberRows()

        ldif = len(self.repos[0])+min_len - nrow

        if ldif > 0:
            self.grid.AppendRows(ldif)
        elif ldif < 0:
            self.grid.DeleteRows(0, -ldif)
        else:
            pass

        for k, p in enumerate(zip(*self.repos)):
            self.grid.SetCellValue(k, 0, p[0])
            self.grid.SetCellValue(k, 1, p[1])
            self.grid.SetCellValue(k, 2, p[2])
        self.grid.AutoSizeColumns()

    def onCellSelected(self, evt):
        self.selected_row = evt.GetRow()
        evt.Skip()

    def onRightRelease(self, evt):
        if self.selected_row < 0:
            return
        m = repolist_popup(self)
        self.PopupMenu(m,
                       evt.GetPosition())
        m.Destroy()

    def onEditRepo(self, evt):
        repo = (self.repos[0][self.selected_row],
                self.repos[1][self.selected_row],
                self.repos[2][self.selected_row])

        ll = (["name", repo[0], 0],
              ["url", repo[1], 0],
              ["protocol", repo[2], 2])

        from ifigure.utils.edit_list import DialogEditList
        ret = DialogEditList(ll, modal=True, parent=self,
                             title='Repository', size=(-1, -1),
                             style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        if not ret[0]:
            return

        self.repos[0][self.selected_row] = ret[1][0]
        self.repos[1][self.selected_row] = ret[1][1]
        self.repos[2][self.selected_row] = 'github'

        self.fill_grid()

    def onAddRepo(self, evt):
        repo = ('', '', 'gihhub')
        ll = (["name", repo[0], 0],
              ["url", repo[1], 0],
              ["protocol", repo[2], 2])

        from ifigure.utils.edit_list import DialogEditList
        ret = DialogEditList(ll, modal=True, parent=self,
                             title='Repository', size=(600, -1),
                             style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        if not ret[0]:
            return

        self.repos[0].append(ret[1][0])
        self.repos[1].append(ret[1][1])
        self.repos[2].append('github')

        self.fill_grid()

    def onRmRepo(self, evt):
        if self.selected_row < 0:
            return

        self.repos[0].pop(self.selected_row)
        self.repos[1].pop(self.selected_row)
        self.repos[2].pop(self.selected_row)

        self.fill_grid()

    def onOK(self, evt):
        self.Close()

    def get_repo_list(self):

        value = []

        nrow = self.grid.GetNumberRows()
        for k in range(nrow):
            n = self.grid.GetCellValue(k, 0)
            u = self.grid.GetCellValue(k, 1)
            p = self.grid.GetCellValue(k, 2)

            if n.strip() == "":
                continue

            value.append({"name": n, "url": u, "protocol": p})

        return value

    def _myRefresh(self):
        self.Fit()
        self.Layout()


class dlg_packages(wx.Dialog):
    def __init__(self, parent, id=wx.ID_ANY, title='packages'):
        from petram.utils import get_user_config

        config = get_user_config()
        self.repos = config["repos"]

        set_default_font()

        wx.Dialog.__init__(self, parent, wx.ID_ANY, title,
                           style=wx.STAY_ON_TOP | wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        sizer = wx.BoxSizer(wx.VERTICAL)

        # grid
        self.SetSizer(sizer)

        self.grid = wx.grid.Grid(self)
        self.grid.CreateGrid(10, 7)
        self.grid.HideRowLabels()
        self.grid.SetSelectionMode(wx.grid.Grid.SelectRows)
        self.grid.SetDefaultCellFont(font)
        self.grid.SetLabelFont(font_label)
        self.grid.SetColLabelSize(int(font_h))
        self.grid.SetDefaultRowSize(int(font_h), True)
        self.grid.EnableDragColSize(True)
        self.grid.SetColLabelValue(0, "package")
        self.grid.SetColLabelValue(1, "installed")
        self.grid.SetColLabelValue(2, "version")
        self.grid.SetColLabelValue(3, "description")
        self.grid.SetColLabelValue(4, "package url")
        self.grid.SetColLabelValue(5, "lastest")
        self.grid.SetColLabelValue(6, "public")

        self.grid.EnableEditing(False)

        sizer.Add(self.grid, 1, wx.EXPAND, 0)

        # buttons
        sizer0 = wx.BoxSizer(wx.HORIZONTAL)
        rpbutton = wx.Button(self, wx.ID_OK, "Repositories...")
        ifbutton = wx.Button(self, wx.ID_OK, "Install from...")
        okbutton = wx.Button(self, wx.ID_OK, "OK")
        sizer0.Add(rpbutton, 0, wx.ALIGN_CENTER | wx.ALL, 1)
        sizer0.Add(ifbutton, 0, wx.ALIGN_CENTER | wx.ALL, 1)
        sizer0.AddStretchSpacer()
        sizer0.Add(okbutton, 0, wx.ALIGN_CENTER | wx.ALL, 1)
        okbutton.Bind(wx.EVT_BUTTON, self.onOK)
        rpbutton.Bind(wx.EVT_BUTTON, self.onRepo)
        ifbutton.Bind(wx.EVT_BUTTON, self.onInstallFrom)

        sizer.Add(sizer0, 0, wx.EXPAND | wx.ALL, 10)

        self.fill_grid()

        # if add_palette:
        wx.GetApp().add_palette(self)

        #
        self.grid.Bind(wx.grid.EVT_GRID_SELECT_CELL, self.onCellSelected)
        self.grid.Bind(wx.grid.EVT_GRID_CELL_RIGHT_CLICK, self.onRightRelease)
        #
        self.selected_row = -1
        self.update_done = False

        self.Show()
        wx.CallAfter(self._myRefresh)

    def fill_grid(self):
        from petram.remote.get_repo_info import (get_local_packages,
                                                 get_repo_info)

        urls = [x['url'] for x in self.repos]

        self.packages = get_repo_info(urls=urls)

        def_l = 3
        nrow = self.grid.GetNumberRows()
        ldif = len(self.packages)+def_l - nrow

        if ldif > 0:
            self.grid.AppendRows(ldif)
        elif ldif < 0:
            self.grid.DeleteRows(0, -ldif)
        else:
            pass

        for k, p in enumerate(self.packages):
            self.grid.SetCellValue(k, 0, p['module'])
            self.grid.SetCellValue(k, 1, "yes" if p['version'] != "" else "no")
            self.grid.SetCellValue(k, 2, p['version'])
            self.grid.SetCellValue(k, 3, p['description'])
            self.grid.SetCellValue(k, 4, p['html_url'])
            self.grid.SetCellValue(k, 5, p['latest'])
            self.grid.SetCellValue(k, 6, p['public'])
        self.grid.AutoSizeColumns()

    def onCellSelected(self, evt):
        self.selected_row = evt.GetRow()
        evt.Skip()

    def onRightRelease(self, evt):
        if self.selected_row < 0:
            return
        m = pkglist_popup(self)
        self.PopupMenu(m,
                       evt.GetPosition())
        m.Destroy()

    def do_recheck(self):
        self.fill_grid()
        self.selected_row = -1
        self.grid.ClearSelection()
        wx.CallAfter(self._myRefresh)

    def onOK(self, evt):
        self.Close()

    def onRepo(self, evt):
        dlg = dlg_repos(self, self.repos)

        def close_dlg(evt, dlg=dlg):
            repos = dlg.get_repo_list()

            self.repos = repos

            from petram.utils import get_user_config, update_user_config

            config = get_user_config()
            config["repos"] = repos
            update_user_config(config)

            dlg.Destroy()

        dlg.Bind(wx.EVT_CLOSE, close_dlg)

    def onInstallFrom(self, evt):
        ll = (["url", "", 0],
              ["example 1 (local)", "/home/user/venvs/src/Petra-M", 0,
               {"style": wx.TE_READONLY | wx.BORDER_NONE}],
              ["example 2 (remote file)", "https://github.com/piScope/Petra-M--RF/archive/refs/tags/v_26_2_7.tar.gz", 0,
               {"style": wx.TE_READONLY | wx.BORDER_NONE}],
              ["example 3 (github)", "git+https://github.com/piScope/Petra-M--RF", 0,
               {"style": wx.TE_READONLY | wx.BORDER_NONE}],
              ["example 4 (github)", "git+ssh://git@github.com/piScope/Petra-M--RF.git", 0,
               {"style": wx.TE_READONLY | wx.BORDER_NONE}],
              [None, " ", 2, {}])

        from ifigure.utils.edit_list import DialogEditList
        ret = DialogEditList(ll, modal=True, parent=self,
                             title='Repository', size=(600, -1),
                             style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        if not ret[0]:
            return

        url = ret[1][0]

        if pip_install(url, parent=self):
            self.update_done = True
            self.do_recheck()

    def _myRefresh(self):
        self.Fit()
        self.Layout()


def check_packages(parent):
    dlg = dlg_packages(parent)

    def close_dlg(evt, dlg=dlg):
        if dlg.update_done:
            from ifigure.widgets.dialog import message
            wx.CallAfter(message, parent,
                         "Packages are updated. Restart piScope, in order to use updated modules.",
                         style=0,
                         title="Update recommended")

        dlg.Destroy()
    dlg.Bind(wx.EVT_CLOSE, close_dlg)
    return dlg


def run_package_env_checker(package_loc, parent=None):
    """
    Run an optional package environment check and ask whether to install.

    Return pip arguments after confirmation, [] when no checker exists, or
    None when the user cancels or checking fails. Supports local directories
    and GitHub HTTPS/SSH repository URLs, optionally prefixed with git+ and
    followed by @revision. GitHub's git@github.com: form is also accepted.
    HTTP(S) source archive URLs skip the check.
    """
    import json
    from pathlib import Path
    import tempfile
    from urllib.error import HTTPError
    from urllib.parse import quote, urlsplit
    from urllib.request import urlopen
    from ifigure.utils.edit_list import DialogEditListWithWindowList

    dialog_style = wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.STAY_ON_TOP

    try:
        with tempfile.TemporaryDirectory(prefix="petram-install-check-") as directory:
            source = Path(package_loc).expanduser()
            if source.is_dir():
                script = source.resolve() / "scripts" / "install_env_check.py"
                if not script.is_file():
                    return []
            else:
                address = package_loc[4:] if package_loc.startswith(
                    "git+") else package_loc
                if address.startswith("git@github.com:"):
                    address = "ssh://git@github.com/" + address[len("git@github.com:"):]
                parsed = urlsplit(address)

                if (not package_loc.startswith("git+") and
                        parsed.scheme in ("http", "https") and
                        parsed.path.lower().endswith((".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".zip"))):
                    # if package is source archive, skip the environment check
                    return []

                ssh = parsed.scheme == "ssh" and parsed.netloc == "git@github.com"
                if not ssh and (parsed.scheme != "https" or parsed.netloc != "github.com"):
                    raise ValueError(
                        "Use a local source directory or a GitHub HTTPS/SSH repository URL.")

                repository, separator, revision = parsed.path.strip(
                    "/").partition("@")
                parts = repository.removesuffix(".git").split("/")
                if len(parts) != 2 or not all(parts) or (separator and not revision):
                    raise ValueError(
                        "Invalid GitHub repository URL: " + package_loc)
                revision = revision if separator else "HEAD"
                script = Path(directory) / "install_env_check.py"
                if ssh:
                    repository_dir = Path(directory) / "repository"
                    remote = "ssh://git@github.com/" + repository
                    sp.run(["git", "clone", "--filter=blob:none", "--no-checkout", "--depth=1",
                            remote, str(repository_dir)], capture_output=True, text=True,
                           timeout=60, check=True)
                    git = ["git", "-C", str(repository_dir)]
                    tree = "HEAD"
                    if separator:
                        sp.run(git + ["fetch", "--depth=1", "--filter=blob:none", "origin", revision],
                               capture_output=True, text=True, timeout=60, check=True)
                        tree = "FETCH_HEAD"
                    listing = sp.run(git + ["ls-tree", "--name-only", tree, "--",
                                            "scripts/install_env_check.py"],
                                     capture_output=True, text=True, timeout=60, check=True)
                    if not listing.stdout.strip():
                        return []
                    contents = sp.run(git + ["show", tree + ":scripts/install_env_check.py"],
                                      capture_output=True, text=True, timeout=60, check=True)
                    script.write_text(contents.stdout)
                else:
                    script_url = ("https://raw.githubusercontent.com/" + "/".join(parts) +
                                  "/" + quote(revision, safe="/") + "/scripts/install_env_check.py")
                    try:
                        with urlopen(script_url, timeout=30) as response:
                            script.write_bytes(response.read())
                    except HTTPError as error:
                        if error.code == 404:
                            return []
                        raise

            result = sp.run([sys.executable, "-B", str(script)], cwd=directory,
                            capture_output=True, text=True, timeout=60, check=True)
            response = json.loads(result.stdout)

            if (not isinstance(response, dict) or
                    not isinstance(response.get("good"), bool) or
                    not isinstance(response.get("message"), str) or
                    not isinstance(response.get("pip_args"), list) or
                    not all(isinstance(arg, str) for arg in response["pip_args"])):
                raise ValueError(
                    "The package environment checker returned an invalid result.")

            if not response["good"]:
                DialogEditListWithWindowList(
                    [[None, response["message"], 2]], modal=True, parent=parent,
                    title="Cannot install package", style=dialog_style)
                return None
            choice = DialogEditListWithWindowList(
                [[None, response["message"], 2]], modal=True, parent=parent,
                title="Install package?", style=dialog_style)
            return response["pip_args"] if choice[0] else None

    except (OSError, ValueError, sp.SubprocessError) as error:
        detail = str(error)
        if isinstance(error, sp.CalledProcessError) and error.stderr:
            detail += "\n" + error.stderr.strip()
        DialogEditListWithWindowList(
            [[None, "Package environment check failed:\n" + detail, 2]],
            modal=True, parent=parent, title="Cannot install package",
            style=dialog_style)

        return None
