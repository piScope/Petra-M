"""
Build configuration and pinned native sources (following Petra-M--DS).
"""
from pathlib import Path
from dataclasses import dataclass, field
from collections import namedtuple
import os


@dataclass(slots=True)
class BuildGlobal:
    """
    Build parameters with fixed member names and mutable values.
    """

    # Frontend configuration and resolved native-build selection.
    cfs: dict = field(default_factory=dict)
    with_superlu_dist: bool = True

    # Tool defaults; configure_build() applies frontend selections.
    cc: str | None = os.getenv("CC", "cc")
    cxx: str | None = os.getenv("CXX", "c++")
    swig: str | None = os.getenv("SWIG", "swig")
    mpicc: str | None = os.getenv("MPICC", "mpicc")
    mpicxx: str | None = os.getenv("MPICXX", "mpicxx")

    # Source root and output directories assigned by BuildExt.
    rootdir: Path = Path(__file__).resolve().parent.parent
    bdist_wheel_prefix: Path | None = None
    build_temp: Path | None = None


bglb = BuildGlobal()

# Follow DS's release records, retaining a checksum for each source archive.
rels = namedtuple('Release', ['version', 'defbranch', 'hash', 'tarball', 'sha256'])

REPOS = dict(
    superlu_dist=dict(
        url="https://github.com/xiaoyeli/superlu_dist",
        releases=[
            rels("9.3.2", "master", "6a70d6f6abcf1ad945748856c35d5612f20065ba",
                 "https://codeload.github.com/xiaoyeli/superlu_dist/tar.gz/6a70d6f6abcf1ad945748856c35d5612f20065ba",
                 "03637ef0c3429b46eab685befb9b89605ab1a43465888784f6f9e29e0d394606"),
        ],
    ),
    openblas=dict(
        url="https://github.com/OpenMathLib/OpenBLAS",
        releases=[
            rels("0.3.34", "develop", "e0166008be8e466242aa76b2ff75ce3f0fbf574a",
                 "https://codeload.github.com/OpenMathLib/OpenBLAS/tar.gz/e0166008be8e466242aa76b2ff75ce3f0fbf574a",
                 "b35b87e3580efc61a664248f143172fd998116c109fee790cd372480c64492a2"),
        ],
    ),
    metis=dict(
        url="https://github.com/scivision/METIS",
        releases=[
            rels("5.2.1.4", "main", "456244e99d7f0027706033d18bc1b3424d3507cc",
                 "https://codeload.github.com/scivision/METIS/tar.gz/456244e99d7f0027706033d18bc1b3424d3507cc",
                 "c4ec016a6161330b2ccf0b85be6d8dbdf45370be4b985ebcb2950e0d8cdbf722"),
        ],
    ),
    gklib=dict(
        url="https://github.com/scivision/GKlib",
        releases=[
            # Snapshot version reported by the pinned CMake project.
            rels("1.0.0", "master", "b4c45d104d421d3adb85fb900ca240eba781c95f",
                 "https://codeload.github.com/scivision/GKlib/tar.gz/b4c45d104d421d3adb85fb900ca240eba781c95f",
                 "053c5d49137345d16256046190916c7aeaff3e80510e7a998625c4af4ecabe6b"),
        ],
    ),
)
