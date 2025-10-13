"""
Unit tests for the naming utilities module.

Tests the NameTemplate generic class and StrSepNameTemplate for:
- Template parsing and generation
- Path manipulation
- Pattern validation
- Custom template subclasses
- Edge cases and error handling
"""

import re
from pathlib import Path
from typing import TypedDict

import pytest

from pyzma_idf_lightcurve.utils.naming import (
    DashSeparated,
    NameTemplate,
    UnderscoreSeparated,
    make_regex_stub_from_literal,
)


class TestMakeRegexStubFromLiteral:
    """Test the regex stub generation from Literal types."""
    
    def test_simple_literal(self):
        """Test with simple literal values."""
        from typing import Literal
        lit = Literal["ch1", "ch2"]
        stub = make_regex_stub_from_literal("channel", lit)
        assert "(?P<channel>" in stub
        assert "ch1" in stub
        assert "ch2" in stub
        
    def test_literal_with_special_chars(self):
        """Test that special regex characters are escaped."""
        from typing import Literal
        lit = Literal["a.b", "c+d"]
        stub = make_regex_stub_from_literal("name", lit)
        # Should escape the special characters
        assert r"a\.b" in stub or "a\\.b" in stub
        assert r"c\+d" in stub or "c\\+d" in stub


class TestStrSepNameTemplate:
    """Test the StrSepNameTemplate base class."""
    
    def test_subclass_with_hyphen(self):
        """Test the DashSeparated concrete class."""
        assert DashSeparated.sep == "-"
        assert DashSeparated.template == "{prefix}{stem}{suffix}"
        
    def test_subclass_with_underscore(self):
        """Test the UnderscoreSeparated concrete class."""
        assert UnderscoreSeparated.sep == "_"
        
    def test_parse_stem_only(self):
        """Test parsing name with only stem."""
        parsed = DashSeparated.parse("simple")
        assert parsed["stem"] == "simple"
        assert parsed["prefix"] == ""
        assert parsed["suffix"] == ""
        
    def test_parse_with_prefix(self):
        """Test parsing name with prefix (requires 3 parts)."""
        # With 3 parts: prefix, stem, suffix all present
        parsed = DashSeparated.parse("pre-stem-suffix")
        assert parsed["prefix"] == "pre-"
        assert parsed["stem"] == "stem"
        assert parsed["suffix"] == "-suffix"
        
    def test_parse_with_suffix(self):
        """Test parsing name with suffix."""
        parsed = DashSeparated.parse("stem-suf")
        assert parsed["stem"] == "stem"
        assert parsed["suffix"] == "-suf"
        assert parsed["prefix"] == ""
        
    def test_parse_with_prefix_and_suffix(self):
        """Test parsing name with both prefix and suffix."""
        parsed = DashSeparated.parse("pre-stem-suf")
        assert parsed["prefix"] == "pre-"
        assert parsed["stem"] == "stem"
        assert parsed["suffix"] == "-suf"
        
    def test_parse_complex(self):
        """Test parsing complex name with 3 parts."""
        # With 2 parts: stem + suffix (no prefix)
        parsed = UnderscoreSeparated.parse("data_value")
        assert parsed["prefix"] == ""
        assert parsed["stem"] == "data"
        assert parsed["suffix"] == "_value"


class TestNameTemplateBasic:
    """Test basic NameTemplate functionality."""
    
    def test_cannot_instantiate(self):
        """Test that NameTemplate cannot be instantiated."""
        class MyName(NameTemplate):
            template = "{name}.txt"
            pattern = re.compile(r"^(?P<name>\w+)\.txt$")
        
        with pytest.raises(TypeError, match="should not be instantiated"):
            MyName()
    
    def test_simple_parse(self):
        """Test parsing a simple name."""
        class SimpleFilename(NameTemplate):
            template = "{name}.{ext}"
            pattern = re.compile(r"^(?P<name>\w+)\.(?P<ext>\w+)$")
        
        parsed = SimpleFilename.parse("file.txt")
        assert parsed["name"] == "file"
        assert parsed["ext"] == "txt"
        
    def test_parse_with_numbers(self):
        """Test parsing name with numbers."""
        class VersionedFile(NameTemplate):
            template = "{name}_v{version}.{ext}"
            pattern = re.compile(r"^(?P<name>\w+)_v(?P<version>\d+)\.(?P<ext>\w+)$")
        
        parsed = VersionedFile.parse("document_v3.pdf")
        assert parsed["name"] == "document"
        assert parsed["version"] == "3"
        assert parsed["ext"] == "pdf"
        
    def test_parse_invalid_name(self):
        """Test parsing fails for non-matching name."""
        class SimpleFilename(NameTemplate):
            template = "{name}.{ext}"
            pattern = re.compile(r"^(?P<name>\w+)\.(?P<ext>\w+)$")
        
        with pytest.raises(ValueError, match="does not match pattern"):
            SimpleFilename.parse("invalid-name")
    
    def test_parse_with_optional_groups(self):
        """Test parsing with optional regex groups."""
        class OptionalFilename(NameTemplate):
            template = "{prefix}{name}.{ext}"
            pattern = re.compile(r"^(?P<prefix>\w+_)?(?P<name>\w+)\.(?P<ext>\w+)$")
        
        # With prefix
        parsed = OptionalFilename.parse("pre_file.txt")
        assert parsed["prefix"] == "pre_"
        assert parsed["name"] == "file"
        
        # Without prefix - should convert None to ""
        parsed = OptionalFilename.parse("file.txt")
        assert parsed["prefix"] == ""
        assert parsed["name"] == "file"


class TestNameTemplateMake:
    """Test name generation with make() method."""
    
    def test_simple_make(self):
        """Test generating simple name."""
        class SimpleFilename(NameTemplate):
            template = "{name}.{ext}"
            pattern = re.compile(r"^(?P<name>\w+)\.(?P<ext>\w+)$")
        
        name = SimpleFilename.make(name="test", ext="txt")
        assert name == "test.txt"
        
    def test_make_complex(self):
        """Test generating complex name."""
        class ComplexFilename(NameTemplate):
            template = "{prefix}_{name}_v{version}.{ext}"
            pattern = re.compile(r"^(?P<prefix>\w+)_(?P<name>\w+)_v(?P<version>\d+)\.(?P<ext>\w+)$")
        
        name = ComplexFilename.make(prefix="data", name="file", version="2", ext="csv")
        assert name == "data_file_v2.csv"
        
    def test_make_missing_parameter(self):
        """Test make() raises error for missing parameter."""
        class SimpleFilename(NameTemplate):
            template = "{name}.{ext}"
            pattern = re.compile(r"^(?P<name>\w+)\.(?P<ext>\w+)$")
        
        with pytest.raises(KeyError, match="Missing required parameter"):
            SimpleFilename.make(name="test")  # Missing ext


class TestNameTemplateRemake:
    """Test remake() method for parsing and regenerating names."""
    
    def test_remake_change_one_field(self):
        """Test remaking name with one changed field."""
        class VersionedFile(NameTemplate):
            template = "{name}_v{version}.{ext}"
            pattern = re.compile(r"^(?P<name>\w+)_v(?P<version>\d+)\.(?P<ext>\w+)$")
        
        new_name = VersionedFile.remake("file_v1.txt", version="2")
        assert new_name == "file_v2.txt"
        
    def test_remake_change_multiple_fields(self):
        """Test remaking name with multiple changed fields."""
        class ComplexName(NameTemplate):
            template = "{prefix}_{name}.{ext}"
            pattern = re.compile(r"^(?P<prefix>\w+)_(?P<name>\w+)\.(?P<ext>\w+)$")
        
        new_name = ComplexName.remake("old_file.txt", prefix="new", ext="csv")
        assert new_name == "new_file.csv"
        
    def test_remake_no_changes(self):
        """Test remaking without changes returns same name."""
        class SimpleFilename(NameTemplate):
            template = "{name}.{ext}"
            pattern = re.compile(r"^(?P<name>\w+)\.(?P<ext>\w+)$")
        
        new_name = SimpleFilename.remake("file.txt")
        assert new_name == "file.txt"


class TestNameTemplateFilepath:
    """Test filepath-related methods."""
    
    def test_make_filepath(self):
        """Test creating full filepath."""
        class SimpleFilename(NameTemplate):
            template = "{name}.{ext}"
            pattern = re.compile(r"^(?P<name>\w+)\.(?P<ext>\w+)$")
        
        path = SimpleFilename.make_filepath(Path("/data"), name="test", ext="txt")
        assert path == Path("/data/test.txt")
        
    def test_remake_filepath_same_dir(self):
        """Test remaking filepath in same directory."""
        class VersionedFile(NameTemplate):
            template = "{name}_v{version}.{ext}"
            pattern = re.compile(r"^(?P<name>\w+)_v(?P<version>\d+)\.(?P<ext>\w+)$")
        
        old_path = Path("/data/file_v1.txt")
        new_path = VersionedFile.remake_filepath(old_path, version="2")
        
        assert new_path == Path("/data/file_v2.txt")
        
    def test_remake_filepath_different_dir(self):
        """Test remaking filepath to different directory."""
        class SimpleFilename(NameTemplate):
            template = "{name}.{ext}"
            pattern = re.compile(r"^(?P<name>\w+)\.(?P<ext>\w+)$")
        
        old_path = Path("/old/file.txt")
        new_path = SimpleFilename.remake_filepath(
            old_path, 
            parent_path=Path("/new"),
            ext="csv"
        )
        
        assert new_path == Path("/new/file.csv")


class TestNameTemplateValidation:
    """Test template and pattern validation."""
    
    def test_validation_with_typeddict(self):
        """Test that subclass with TypedDict validates correctly."""
        class MyKeys(TypedDict):
            name: str
            version: str
        
        # This should validate successfully
        class MyFilename(NameTemplate[MyKeys]):
            template = "{name}_v{version}"
            pattern = re.compile(r"^(?P<name>\w+)_v(?P<version>\d+)$")
        
        # Should work without error
        result = MyFilename.parse("file_v1")
        assert result["name"] == "file"
        assert result["version"] == "1"
    
    def test_validation_missing_group(self):
        """Test validation fails when pattern missing required group."""
        class MyKeys(TypedDict):
            name: str
            version: str
        
        # This should fail validation - pattern missing 'version' group
        with pytest.raises(ValueError, match="Pattern missing required TypedDict keys"):
            class BadFilename(NameTemplate[MyKeys]):
                template = "{name}_v{version}"
                pattern = re.compile(r"^(?P<name>\w+)_v\d+$")  # Missing version group
    
    def test_validation_extra_group(self):
        """Test validation warns about unexpected groups."""
        class MyKeys(TypedDict):
            name: str
        
        # This should fail validation - pattern has extra 'ext' group
        with pytest.raises(ValueError, match="Pattern has unexpected named groups"):
            class BadFilename(NameTemplate[MyKeys]):
                template = "{name}"
                pattern = re.compile(r"^(?P<name>\w+)\.(?P<ext>\w+)$")


class TestRealWorldExamples:
    """Test real-world use cases."""
    
    def test_astronomical_filename(self):
        """Test parsing astronomical image filename."""
        class IRAcFilename(NameTemplate):
            template = "SPITZER_I{chan}_{aorkey}_bcd.fits"
            pattern = re.compile(
                r"^SPITZER_I(?P<chan>[12])_(?P<aorkey>\d+)_bcd\.fits$"
            )
        
        parsed = IRAcFilename.parse("SPITZER_I1_12345678_bcd.fits")
        assert parsed["chan"] == "1"
        assert parsed["aorkey"] == "12345678"
        
        # Make a similar filename
        name = IRAcFilename.make(chan="2", aorkey="87654321")
        assert name == "SPITZER_I2_87654321_bcd.fits"
        
    def test_mosaic_filename(self):
        """Test mosaic filename handling."""
        class MosaicFilename(NameTemplate):
            template = "mosaic_{field}_{filter}.fits"
            pattern = re.compile(
                r"^mosaic_(?P<field>\w+)_(?P<filter>\w+)\.fits$"
            )
        
        # Change filter
        new_name = MosaicFilename.remake("mosaic_IDF_ch1.fits", filter="ch2")
        assert new_name == "mosaic_IDF_ch2.fits"
        
    def test_catalog_filename_with_path(self):
        """Test catalog filename with path manipulation."""
        class CatalogFilename(NameTemplate):
            template = "{field}_{epoch}_cat.ecsv"
            pattern = re.compile(
                r"^(?P<field>\w+)_(?P<epoch>r\d+)_cat\.ecsv$"
            )
        
        old_path = Path("/data/epoch1/IDF_r12345678_cat.ecsv")
        new_path = CatalogFilename.remake_filepath(
            old_path,
            parent_path=Path("/data/epoch2"),
            epoch="r87654321"
        )
        
        assert new_path == Path("/data/epoch2/IDF_r87654321_cat.ecsv")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_string_parse(self):
        """Test parsing empty string."""
        class SimpleFilename(NameTemplate):
            template = "{name}"
            pattern = re.compile(r"^(?P<name>\w+)$")
        
        with pytest.raises(ValueError):
            SimpleFilename.parse("")
            
    def test_none_to_empty_string(self):
        """Test that None in optional groups converts to empty string."""
        class OptionalPrefix(NameTemplate):
            template = "{prefix}{name}"
            pattern = re.compile(r"^(?P<prefix>\w+_)?(?P<name>\w+)$")
        
        parsed = OptionalPrefix.parse("file")
        assert parsed["prefix"] == ""  # None converted to ""
        assert parsed["name"] == "file"
