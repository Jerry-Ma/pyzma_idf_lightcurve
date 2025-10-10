"""
Types and constants.

"""

import re
from pathlib import Path
from typing import Literal, TypedDict, get_args
from .utils.naming import NameTemplate, make_regex_stub_from_literal


GroupNameT = str
ChanT = Literal["ch1", "ch2"]
ImageKindT = Literal["sci", "std", "unc", "cov"]
ImageSuffixT = str
FileExtT = Literal[".fits", ".sexcat", ".ecsv"]


class IDFFilenameT(TypedDict):
    group_name: GroupNameT
    chan: ChanT
    kind: ImageKindT
    suffix: ImageSuffixT
    fileext: FileExtT


class IDFFilename(NameTemplate[IDFFilenameT]):
    """Template for IDF filenames."""
    template = "IDF_{group_name}_{chan}_{kind}{suffix}{fileext}"
    pattern = re.compile(
        rf"^IDF_(?P<group_name>gr\d+)"
        rf"_{make_regex_stub_from_literal("chan", ChanT)}"
        rf"_{make_regex_stub_from_literal("kind", ImageKindT)}"
        rf"(?P<suffix>_[^.]+|)"
        rf"{make_regex_stub_from_literal("fileext", FileExtT)}"
        )
    
    @classmethod
    def make_sci_filepath(cls, parent_path: Path, group_name: GroupNameT, chan: ChanT) -> Path:
        """Generate sci filepath."""
        return cls.make_filepath(parent_path, group_name=group_name, chan=chan, kind='sci', suffix='', fileext='.fits')
    
    @classmethod
    def make_unc_filepath(cls, parent_path: Path, group_name: GroupNameT, chan: ChanT) -> Path:
        """Generate unc filepath."""
        return cls.make_filepath(parent_path, group_name=group_name, chan=chan, kind='unc', suffix='', fileext='.fits')

