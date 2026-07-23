"""Restore NumPy<2 aliases that disimpy 0.3.0 still uses. Import BEFORE disimpy."""
import numpy as np

_aliases = {
    "trapz": "trapezoid",
    "product": "prod",
    "cumproduct": "cumprod",
    "alltrue": "all",
    "sometrue": "any",
    "round_": "round",
}
for old, new in _aliases.items():
    if not hasattr(np, old) and hasattr(np, new):
        setattr(np, old, getattr(np, new))
for old, new in {"float_": "float64", "int_": "int64", "bool_": "bool_"}.items():
    if not hasattr(np, old) and hasattr(np, new):
        setattr(np, old, getattr(np, new))
