"""
Generic template-based naming utilities.

This module provides the general NameTemplate class for parsing and generating
structured names using template patterns with full type safety.
"""

import re
from pathlib import Path
from types import UnionType
from typing import Any, ClassVar, Generic, TypedDict, TypeVar, get_args

# Generic type variable for the key type - no bound since TypedDict isn't assignable to dict
NameKeyT = TypeVar('NameKeyT')


class NameTemplate(Generic[NameKeyT]):
    """
    A generic template-based name codec for parsing and generating names.
    
    This class handles bidirectional conversion between dictionaries of parameters
    and formatted names using template patterns with full type safety.
    
    Subclasses should define:
        template: str - Format string template
        pattern: re.Pattern[str] - Regex pattern for parsing
    
    All methods are class methods - no instantiation needed.
    
    Example:
        class MyFilename(NameTemplate[MyFilenameT]):
            template = "{prefix}_{name}_{version}.{ext}"
            pattern = re.compile(r"^(?P<prefix>\\w+)_(?P<name>\\w+)_(?P<version>\\d+)\\.(?P<ext>\\w+)$")
    """
    
    template: str
    pattern: re.Pattern[str]
    
    def __init_subclass__(cls, **kwargs):
        """Extract TypedDict type from generic base and validate template definition."""
        super().__init_subclass__(**kwargs)
        
        # Try to get the generic base classes (fallback for older Python versions)
        try:
            for base in getattr(cls, '__orig_bases__', []):
                if hasattr(base, '__origin__') and base.__origin__ is NameTemplate:
                    # Extract the TypedDict type argument
                    if hasattr(base, '__args__') and base.__args__:
                        cls._typed_dict_type = base.__args__[0]
                        break
        except (AttributeError, TypeError):
            # Fallback: assume the class defines _typed_dict_type explicitly if needed
            pass
        
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
        result: dict[str, str] = {}
        for key, value in match.groupdict().items():
            result[key] = value if value is not None else ""
            
        return result  # type: ignore
    
    @classmethod
    def remake(cls, name: str, **kwargs: Any) -> str:
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
            new_name = MyTemplate.remake("data_ch1_v1.fits", chan="ch2")
            # Result: "data_ch2_v1.fits"
        """
        # Parse the original name
        parsed: dict[str, str] = cls.parse(name)  # type: ignore
        
        # Update with new values
        parsed.update(kwargs)
        
        # Generate new name
        return cls.make(**parsed)
    
    @classmethod
    def remake_filepath(cls, filepath: Path, parent_path: Path | None = None, **kwargs: Any) -> Path:
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
            # Change version and move to different directory
            old_path = Path("/data/file_v1.txt")
            new_path = MyTemplate.remake_filepath(old_path, Path("/workdir"), version="2")
            # Result: Path("/workdir/file_v2.txt")
        """
        # Extract filename and remake it
        new_filename = cls.remake(filepath.name, **kwargs)
        
        # Use specified parent_path or original parent directory
        target_parent = parent_path if parent_path is not None else filepath.parent
        
        # Return new path with target parent directory
        return target_parent / new_filename
    
    @classmethod
    def make(cls, **kwargs: Any) -> str:
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
    def make_filepath(cls, parent_path: Path, **kwargs: Any) -> Path:
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


def make_regex_stub_from_literal(group_name: str, literal: UnionType) -> str:
    """Generate a regex stub from a Literal type."""
    # Extract the possible values from the Literal
    values = get_args(literal)
    # Escape each value for regex and join with |
    pattern = "|".join(re.escape(v) for v in values)
    return f"(?P<{group_name}>{pattern})"


StrSepT = TypeVar('StrSepT')


class StrSepSegment(TypedDict):

    prefix: str
    stem: str
    suffix: str


class StrSepNameTemplate(NameTemplate[StrSepSegment]):
    """Name template that parses names separated by a delimiter.
    
    Subclasses should set the 'sep' class attribute to define their separator.
    """

    match_prefix: bool = True
    match_suffix: bool = True
    sep: ClassVar[str] = NotImplemented

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Use the class's sep attribute to build pattern
        sep = cls.sep
        assert isinstance(sep, str), f"sep must be a string, got {type(sep)}"

        match_prefix = cls.match_prefix
        match_suffix = cls.match_suffix
        assert match_prefix + match_suffix >= 1, "at least one of match_{prefix|suffix} should be True"
        if match_prefix and match_suffix:
            re_str = rf'^(?:(?P<prefix>[^{sep}]+{sep})(?=[^{sep}]*{sep}))?(?P<stem>[^{sep}]+)(?P<suffix>{sep}.+)?$'
        elif match_prefix:
            # greed on prefix
            re_str = rf'^(?P<prefix>.+{sep})?(?P<stem>[^{sep}]+)$'
        elif match_suffix:
            # greed on suffix
            re_str = rf'^(?P<stem>[^{sep}]+)(?P<suffix>{sep}.+)?$'
        else:
            assert False
        cls.template = "{prefix}{stem}{suffix}"
        cls.pattern = re.compile(re_str)
    
    @classmethod
    def parse(cls, name: str) -> StrSepSegment:
        """Parse name and ensure all three keys (prefix, stem, suffix) are present."""
        result = super().parse(name)
        # Ensure all three keys exist, filling with empty string if not in regex groups
        if "prefix" not in result:
            result["prefix"] = ""
        if "suffix" not in result:
            result["suffix"] = ""
        return result  # type: ignore


class UnderscoreSeparated(StrSepNameTemplate):
    sep: ClassVar[str] = "_"


class DashSeparated(StrSepNameTemplate):
    sep: ClassVar[str] = "-"


class DotSeparated(StrSepNameTemplate):
    sep: ClassVar[str] = "."
