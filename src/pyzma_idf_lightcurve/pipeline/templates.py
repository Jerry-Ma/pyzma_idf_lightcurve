"""
Template classes for name parsing and generation in the IDF pipeline.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal, TypeVar, Generic
from typing_extensions import TypedDict


# Generic type variable for the key type
NameKeyT = TypeVar('NameKeyT', bound=Dict[str, str])

class NameTemplate(Generic[NameKeyT]):
    """
    A generic template-based name codec for parsing and generating names.
    
    This class handles bidirectional conversion between dictionaries of parameters
    and formatted names using template patterns with full type safety.
    
    Subclasses should define:
        template: str - Format string template
        pattern: re.Pattern[str] - Regex pattern for parsing
    
    All methods are class methods - no instantiation needed.
    """
    
    template: str
    pattern: re.Pattern[str]
    
    def __init_subclass__(cls, **kwargs):
        """Extract TypedDict type from generic base and validate template definition."""
        super().__init_subclass__(**kwargs)
        
        # Get the generic base classes
        for base in cls.__orig_bases__:
            if hasattr(base, '__origin__') and base.__origin__ is NameTemplate:
                # Extract the TypedDict type argument
                if hasattr(base, '__args__') and base.__args__:
                    cls._typed_dict_type = base.__args__[0]
                    break
        
        # Extract expected keys from TypedDict type
        if hasattr(cls, '_typed_dict_type') and hasattr(cls._typed_dict_type, '__annotations__'):
            cls._expected_keys = set(cls._typed_dict_type.__annotations__.keys())
        else:
            cls._expected_keys = set()
        
        # Validate template and pattern if they're defined
        if hasattr(cls, 'template') and hasattr(cls, 'pattern') and cls._expected_keys:
            cls._validate_class_definition()
    
    @classmethod
    def _validate_class_definition(cls):
        """
        Validate that the class-level pattern contains all expected keys as named groups.
        
        Raises:
            ValueError: If pattern is missing required named groups or has unexpected groups
        """
        if not hasattr(cls, 'pattern') or not hasattr(cls, '_expected_keys'):
            return
            
        pattern_groups = set(cls.pattern.groupindex.keys())
        missing_keys = cls._expected_keys - pattern_groups
        extra_keys = pattern_groups - cls._expected_keys
        
        errors = []
        if missing_keys:
            errors.append(f"Pattern missing required TypedDict keys as named groups: {sorted(missing_keys)}")
        if extra_keys:
            errors.append(f"Pattern has unexpected named groups not in TypedDict: {sorted(extra_keys)}")
        
        if errors:
            raise ValueError(
                f"Pattern validation failed for class {cls.__name__} with template '{cls.template}'. "
                f"Expected keys from TypedDict: {sorted(cls._expected_keys)}. "
                f"Pattern named groups: {sorted(pattern_groups)}. "
                f"Errors: {'; '.join(errors)}"
            )
    
    def __init__(self):
        """Prevent instantiation - use class methods directly."""
        raise TypeError(f"{self.__class__.__name__} should not be instantiated. Use class methods directly.")
    
    @classmethod
    def parse(cls, name: str) -> NameKeyT:
        """
        Parse name into components dictionary.
        
        Args:
            name: Name to parse
            
        Returns:
            Dictionary of parsed components
            
        Raises:
            ValueError: If name doesn't match pattern
        """
        match = cls.pattern.match(name)
        if not match:
            raise ValueError(f"Name '{name}' does not match pattern '{cls.pattern.pattern}'")
        
        # Convert None values to empty strings for optional groups
        result = {}
        for key, value in match.groupdict().items():
            result[key] = value if value is not None else ""
            
        return result  # type: ignore
    
    @classmethod
    def remake(cls, name: str, **kwargs) -> str:
        """
        Parse name, update with new values, and generate new name.
        
        Args:
            name: Original name to parse and modify
            **kwargs: Component values to update (overwrites parsed values)
            
        Returns:
            New name with updated components
            
        Raises:
            ValueError: If name doesn't match pattern
            KeyError: If required template parameters are missing after update
            
        Example:
            # Change channel from ch1 to ch2
            new_name = IDFFilename.remake("IDF_gr123_ch1_sci.fits", chan="ch2")
            # Result: "IDF_gr123_ch2_sci.fits"
        """
        # Parse the original name
        parsed = cls.parse(name)
        
        # Update with new values
        parsed.update(kwargs)
        
        # Generate new name
        return cls.make(**parsed)
    
    @classmethod
    def remake_filepath(cls, filepath: Path, parent_path: Optional[Path] = None, **kwargs) -> Path:
        """
        Parse filepath name, update with new values, and generate new Path.
        
        Args:
            filepath: Original Path object to parse and modify
            parent_path: Optional different parent directory for the new path.
                        If None, uses the same parent as the original filepath.
            **kwargs: Component values to update (overwrites parsed values)
            
        Returns:
            New Path object with updated filename in specified or same directory
            
        Raises:
            ValueError: If filename doesn't match pattern
            KeyError: If required template parameters are missing after update
            
        Example:
            # Change channel from ch1 to ch2, same directory
            old_path = Path("/data/IDF_gr123_ch1_sci.fits")
            new_path = IDFFilename.remake_filepath(old_path, chan="ch2")
            # Result: Path("/data/IDF_gr123_ch2_sci.fits")
            
            # Change channel and move to different directory
            new_path = IDFFilename.remake_filepath(old_path, Path("/workdir"), chan="ch2")
            # Result: Path("/workdir/IDF_gr123_ch2_sci.fits")
        """
        # Extract filename and remake it
        new_filename = cls.remake(filepath.name, **kwargs)
        
        # Use specified parent_path or original parent directory
        target_parent = parent_path if parent_path is not None else filepath.parent
        
        # Return new path with target parent directory
        return target_parent / new_filename
    
    @classmethod
    def make(cls, **kwargs: str) -> str:
        """
        Generate name from components.
        
        Args:
            **kwargs: Component values to substitute into template
            
        Returns:
            Formatted name
            
        Raises:
            KeyError: If required template parameters are missing
        """
        try:
            return cls.template.format(**kwargs)
        except KeyError as e:
            raise KeyError(f"Missing required parameter {e} for template '{cls.template}'")
    
    @classmethod
    def make_filepath(cls, parent_path: Path, **kwargs: str) -> Path:
        """
        Generate full Path object from components.
        
        Args:
            parent_path: Parent directory path
            **kwargs: Component values to substitute into template
            
        Returns:
            Full Path object
        """
        filename = cls.make(**kwargs)
        return parent_path / filename


# Template classes - use directly, no instantiation needed!

# Type aliases for commonly used literals
ChanT = Literal["ch1", "ch2"]
KindT = Literal["sci", "std", "unc", "cov"]

# IDF filename template with flexible suffix and extension
# Also used for partition keys since they're derived from IDF filenames
class PartitionKeyT(TypedDict):
    """Type for partition key components (subset of IDF filename)."""
    group_name: str
    chan: ChanT

class IDFFilenameT(PartitionKeyT):
    """Type for IDF filename components with all required fields."""
    kind: KindT
    suffix: str  # e.g., "" or "_clean" or "_div"
    fileext: str  # e.g., "fits" or "sexcat" or "ecsv"

class PartitionKey(NameTemplate[PartitionKeyT]):
    template = "{group_name}_{chan}"
    pattern = re.compile(r"^(?P<group_name>gr\d+)_(?P<chan>ch[1-2])$")
    
    @classmethod
    def get_components(cls, partition_key: str) -> tuple[str, ChanT]:
        """Extract group_name and chan from partition key."""
        parsed = cls.parse(partition_key)
        return parsed['group_name'], parsed['chan']  # type: ignore

class IDFFilename(NameTemplate[IDFFilenameT]):
    template = "IDF_{group_name}_{chan}_{kind}{suffix}{fileext}"
    pattern = re.compile(r"^IDF_(?P<group_name>gr\d+)_(?P<chan>ch[1-2])_(?P<kind>sci|std|unc|cov)(?P<suffix>_[^.]*)?(?P<fileext>\..+)$")
    
    @classmethod
    def make_sci_filepath(cls, parent_path: Path, group_name: str, chan: ChanT) -> Path:
        """Generate sci filepath."""
        return cls.make_filepath(parent_path, group_name=group_name, chan=chan, kind='sci', suffix='', fileext='.fits')
    
    @classmethod
    def make_unc_filepath(cls, parent_path: Path, group_name: str, chan: ChanT) -> Path:
        """Generate unc filepath."""
        return cls.make_filepath(parent_path, group_name=group_name, chan=chan, kind='unc', suffix='', fileext='.fits')