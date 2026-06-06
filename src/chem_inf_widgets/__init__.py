try:
    from chem_inf_widgets._orange_compat import patch_orange_table_to_frame
except Exception:
    # Resource-only contexts (for example wheel smoke checks installed with
    # ``--no-deps``) should still be able to import the top-level package even
    # when Orange/numpy/pandas are not available yet.
    patch_orange_table_to_frame = None
else:
    patch_orange_table_to_frame()

__version__ = "0.3.0"
