"""Tiny shim for the small subset of `funcy` used by this repo.

This implements minimal versions of:
- notnone: predicate returning True if value is not None
- once: decorator that runs a function once and caches the result
- select_values: return a dict filtered by predicate applied to values

This is intentionally tiny and only meant to help run the project without
installing the real `funcy` package. Install `funcy` via pip if you need the
full library behavior.
"""
from functools import wraps
from typing import Callable, Dict, Any


def notnone(x: Any) -> bool:
    """Return True when x is not None."""
    return x is not None


def once(fn: Callable):
    """Decorator that calls a function at most once and caches the result.

    Subsequent calls return the cached result. This ignores arguments and
    matches the small usage pattern in this repository (single no-arg
    factory functions).
    """
    called = False
    result = None

    @wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal called, result
        if not called:
            result = fn(*args, **kwargs)
            called = True
        return result

    return wrapper


def select_values(pred: Callable[[Any], bool], d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow dict with items whose values satisfy pred(value).

    This mirrors funcy's select_values(pred, mapping).
    """
    return {k: v for k, v in d.items() if pred(v)}
