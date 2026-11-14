"""
Minimal subprocess/download/CMake utilities adapted from PyMFEM and DS.
"""
import hashlib
import os
import shlex
from pathlib import Path
from shutil import which
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

def configure_mpi(bglb):

    """
    Locate the wrappers and query C++ flags without loading an MPI runtime.
    """
    def get_mpi_wrapper_info(compiler):
        """
        Query MPI wrapper flags using the PyMFEM build utility approach.

        It supposes OpenMPI, MPICH, Cray and Intel MPI wrappers. It also handles
        the compiler wrapper which accepts query with -show. (kind=show-compatible)
        The selected wrapper's active environment must be loaded by the caller.
        """
        command = shlex.split(compiler) if isinstance(compiler, str) else list(compiler)

        def query(option):
            try:
                result = subprocess.run(command + [option], capture_output=True,
                                        text=True, timeout=10)
            except (OSError, subprocess.TimeoutExpired) as error:
                raise RuntimeError(f'Cannot query MPI wrapper {command!r}: {error}') from error
            return result

        compile_result = query('--showme:compile')

        if compile_result.returncode == 0:
            # openmpi
            link_result = query('--showme:link')
            if link_result.returncode != 0:
                raise RuntimeError('Open MPI link query failed: ' + link_result.stderr.strip())
            kind = 'openmpi'
            output = compile_result.stdout + '\n' + link_result.stdout

        else:
            result = query('--cray-print-opts=all')
            if result.returncode == 0:
                kind = 'cray'
                output = result.stdout
            else:
                result = query('-show')
                if result.returncode != 0:
                    raise RuntimeError(f'Unsupported MPI wrapper {command!r}: ' +
                                    result.stderr.strip())
                output = result.stdout
                version = query('-v')
                banner = (version.stdout + '\n' + version.stderr).lower()
                executable = os.path.basename(command[0])

                if ('intel(r) mpi' in banner or 'intel mpi' in banner or
                        executable in ('mpiicc', 'mpiicpc', 'mpiicx', 'mpiicpx')):
                    kind = 'intel'
                elif 'mpich' in banner:
                    kind = 'mpich'
                else:
                    kind = 'show-compatible'

        flags = shlex.split(output)
        info = dict(type=kind, include_dirs=[], library_dirs=[], libraries=[], flags=flags)
        options = (('-isystem', 'include_dirs'), ('-I', 'include_dirs'),
                ('-L', 'library_dirs'), ('-l', 'libraries'))
        tokens = iter(flags)
        for token in tokens:
            for option, key in options:
                if token == option:
                    value = next(tokens, '')
                    if not value or value.startswith('-'):
                        raise ValueError(f'Missing value for {option} in MPI wrapper output')
                elif token.startswith(option):
                    value = token[len(option):]
                else:
                    continue
                if value not in info[key]:
                    info[key].append(value)
                break
        if not any(info[key] for key in ('include_dirs', 'library_dirs', 'libraries')):
            raise ValueError(f'No include or library flags reported by MPI wrapper {command!r}')
        return info


    for name in ('mpicc', 'mpicxx'):
        command = getattr(bglb, name)
        wrapper = which(command) if command else None
        if wrapper is None:
            raise RuntimeError(
                'Cannot find MPI compiler wrapper ' + name.upper() +
                ': ' + str(command) + '. Install MPI development tools or set ' +
                name.upper() + ' to an executable wrapper. To omit the native ' +
                'component, use -Cwith-superlu_dist=No.')
        
        setattr(bglb, name, wrapper)
        print('MPI compiler ' + name.upper() + ': ' + wrapper, flush=True)

    return get_mpi_wrapper_info(bglb.mpicxx)