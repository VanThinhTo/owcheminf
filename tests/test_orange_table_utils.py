from __future__ import annotations

import math

import pytest

pytest.importorskip("Orange")

from chem_inf_widgets.chemcore.services.orange_table_utils import (
    as_float_or_nan,
    column_is_numeric,
    is_missing,
)


def test_is_missing_handles_common_blank_tokens():
    assert is_missing(None) is True
    assert is_missing("") is True
    assert is_missing(" ? ") is True
    assert is_missing("nan") is True
    assert is_missing("None") is True
    assert is_missing("CCO") is False


def test_as_float_or_nan_parses_decimal_comma_and_invalid_values():
    assert as_float_or_nan("3,5") == 3.5
    assert math.isnan(as_float_or_nan("not-a-number"))
    assert math.isnan(as_float_or_nan(None))


def test_column_is_numeric_accepts_missing_values_but_rejects_text():
    assert column_is_numeric([1, "2,5", None, ""]) is True
    assert column_is_numeric(["1.0", "abc"]) is False
    assert column_is_numeric([None, "", "?"]) is False
