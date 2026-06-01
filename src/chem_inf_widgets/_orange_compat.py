from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def _patched_table_to_frame(tab, include_metas: bool = False):
    """Orange ``table_to_frame`` variant compatible with newer pandas."""
    from Orange.data import pandas_compat as orange_pandas_compat

    def _column_to_series(col, vals):
        result = ()
        if col.is_discrete:
            # pandas 2.3 warns when object-dtype arrays are downcast via fillna.
            # Orange discrete columns already store numeric codes, so we coerce
            # explicitly before replacing missing values.
            codes = pd.to_numeric(pd.Series(vals), errors="coerce").fillna(-1).astype(int)
            result = (
                col.name,
                pd.Categorical.from_codes(
                    codes=codes,
                    categories=col.values,
                    ordered=True,
                ),
            )
        elif col.is_time:
            result = (col.name, pd.to_datetime(vals, unit="s").to_series().reset_index()[0])
        elif col.is_continuous:
            dt = float
            if col.number_of_decimals == 0 and not np.any(pd.isnull(vals)):
                dt = int
            result = (col.name, pd.Series(vals).astype(dt))
        elif col.is_string:
            result = (col.name, pd.Series(vals))
        return result

    def _columns_to_series(cols, vals):
        return [_column_to_series(col, vals[:, i]) for i, col in enumerate(cols)]

    x, y, metas = [], [], []
    domain = tab.domain
    if domain.attributes:
        x = _columns_to_series(domain.attributes, tab.X)
    if domain.class_vars:
        y_values = tab.Y.reshape(tab.Y.shape[0], len(domain.class_vars))
        y = _columns_to_series(domain.class_vars, y_values)
    if include_metas and domain.metas:
        metas = _columns_to_series(domain.metas, tab.metas)
    all_series = dict(x + y + metas)
    all_vars = tab.domain.variables
    if include_metas:
        all_vars += tab.domain.metas
    original_column_order = [var.name for var in all_vars]
    unsorted_columns_df = pd.DataFrame(all_series)
    return unsorted_columns_df[original_column_order]


def patch_orange_table_to_frame() -> None:
    try:
        import Orange.data as orange_data
        from Orange.data import pandas_compat as orange_pandas_compat
    except Exception:
        return

    current = getattr(orange_pandas_compat, "table_to_frame", None)
    if current is None or getattr(current, "_owcheminf_patched", False):
        return

    patched = _patched_table_to_frame
    patched._owcheminf_patched = True  # type: ignore[attr-defined]
    orange_pandas_compat.table_to_frame = patched
    if getattr(orange_data, "table_to_frame", None) is current:
        orange_data.table_to_frame = patched

    # Modules such as Orange.data.aggregate can bind the old function during
    # import. Update any already-loaded aliases that still point at it.
    for module in tuple(sys.modules.values()):
        if module is None:
            continue
        if getattr(module, "table_to_frame", None) is current:
            try:
                setattr(module, "table_to_frame", patched)
            except Exception:
                continue


__all__ = ["patch_orange_table_to_frame"]
