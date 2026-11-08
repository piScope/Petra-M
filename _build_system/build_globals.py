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
    with_superlu_mt: bool = True

    # Tool defaults; configure_build() applies frontend selections.
    cc: str | None = os.getenv("CC", "cc")
    cxx: str | None = os.getenv("CXX", "c++")
    swig: str | None = os.getenv("SWIG", "swig")

    # Source root and output directories assigned by BuildExt.
    rootdir: Path = Path(__file__).resolve().parent.parent
    bdist_wheel_prefix: Path | None = None
    build_temp: Path | None = None


bglb = BuildGlobal()

# Follow DS's release records, retaining a checksum for each source archive.
rels = namedtuple('Release', ['version', 'defbranch', 'hash', 'tarball', 'sha256'])

REPOS = dict(
    superlu_mt=dict(
        url="https://github.com/xiaoyeli/superlu_mt",
        releases=[
            rels("4.1.0", "master", "f2e50bfcbb7b20ef304c84cf92c038c283019863",
                 "https://codeload.github.com/xiaoyeli/superlu_mt/tar.gz/f2e50bfcbb7b20ef304c84cf92c038c283019863",
                 "e9c65de47453f99a317d089b3b8c8bce34e7a79f45d328de6082c07d4d24eaf5"),
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
