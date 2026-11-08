"""
Minimal subprocess/download/CMake utilities adapted from PyMFEM and DS.
"""
import hashlib
from pathlib import Path
import subprocess
import tarfile
from urllib.request import urlopen

from build_globals import REPOS


def make_call(command, cwd=None):
    """
    Run a build step, preserving its output and propagating failures.
    """
    print("calling ... " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def download(name, source_root):
    """
    Fetch and verify a pinned source archive, then extract its single root.
    """
    release = REPOS[name]["releases"][-1]
    digest = release.sha256
    source_root = Path(source_root)
    source_root.mkdir(parents=True, exist_ok=True)
    destination = source_root / name
    marker = destination / ".petram-source-sha256"
    if marker.is_file() and marker.read_text().strip() == digest:
        return destination
    archive = source_root / (name + "-" + release.hash + ".tar.gz")
    if not archive.is_file():
        with urlopen(release.tarball, timeout=60) as response:
            archive.write_bytes(response.read())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Source archive checksum mismatch: " + str(archive))
    if destination.exists():
        raise RuntimeError("Unverified source directory; remove it before rebuilding: "
                           + str(destination))
    with tarfile.open(archive) as stream:
        # filter='data' is available from Python 3.12; validate paths explicitly
        # as well, since Petra-M also supports Python 3.10 and 3.11.
        members = stream.getmembers()
        roots = {Path(member.name).parts[0] for member in members}
        if len(roots) != 1:
            raise RuntimeError("Expected a single source archive root: " + str(archive))
        extracted = source_root / roots.pop()
        for member in members:
            path = source_root / member.name
            if (not path.resolve().is_relative_to(source_root.resolve()) or
                    member.issym() or member.islnk() or
                    not (member.isdir() or member.isfile())):
                raise RuntimeError("Unsafe source archive entry: " + member.name)
        options = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        stream.extractall(source_root, **options)
    extracted.rename(destination)
    marker.write_text(digest + "\n")
    return destination


def cmake(*args, **kwargs):
    """
    Run CMake with command arguments and option keys following PyMFEM and DS.
    """
    command = ["cmake"] + list(args)
    for key, value in kwargs.items():
        command.append("-" + key + "=" + str(value))
    make_call(command)
