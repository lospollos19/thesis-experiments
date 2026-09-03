"""One test per binding form. Three read a value; two call a function."""

from demo import config


def test_reads_constant_as_attribute():
    assert config.SIZE == 10


def test_reads_import_time_derived_value():
    assert config.DERIVED == 50


def test_reads_mutable_structure():
    assert len(config.NAMES) == 2


def test_calls_function_with_default_argument():
    assert config.scale(2) == 6


def test_calls_function_reading_constant_at_runtime():
    assert config.limit_plus(1) == 6
