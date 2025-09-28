"""Normalize Google GenAI SDK or bridge responses to a common shape.

The function `normalize_genai_response` accepts either an SDK response object
or a plain dict (bridge JSON) and returns a dict with canonical keys.

This module is intentionally small and defensive: it uses getattr/getitem
checks and preserves the raw response for debugging.
"""
from __future__ import annotations
import json
from typing import Any, Dict, Optional


def _try_getattr(obj: Any, *attrs):
    """Try a sequence of attribute names on obj and return the first non-None value."""
    for a in attrs:
        try:
            v = getattr(obj, a)
        except Exception:
            v = None
        if v is not None:
            return v
    return None


def _try_getitem(d: Dict, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _as_text_from_candidate(candidate: Any) -> Optional[str]:
    # Candidate may be SDK object or dict
    if candidate is None:
        return None
    # Try attribute access
    txt = _try_getattr(candidate, "text", "content")
    if isinstance(txt, str):
        return txt
    # Candidate may contain output items
    try:
        if hasattr(candidate, "output") and candidate.output:
            o = candidate.output[0]
            t = _try_getattr(o, "text", "content")
            if isinstance(t, str):
                return t
    except Exception:
        pass
    # dict access
    try:
        if isinstance(candidate, dict):
            t = _try_getitem(candidate, "text", "content", "output")
            if isinstance(t, str):
                return t
            # if output is a list of content pieces
            out = candidate.get("output")
            if isinstance(out, list) and out:
                first = out[0]
                if isinstance(first, dict):
                    return _try_getitem(first, "text", "content")
    except Exception:
        pass
    return None


def normalize_genai_response(response: Any, request_meta: Optional[Dict] = None) -> Dict[str, Any]:
    """Normalize various GenAI response shapes into a canonical dict.

    Returns keys: text, finish_reason, function_call, model, usage, safety, raw
    """
    out: Dict[str, Any] = {
        "text": None,
        "finish_reason": None,
        "function_call": None,
        "model": None,
        "usage": None,
        "safety": None,
        "raw": response,
    }

    # If response is None or empty
    if response is None:
        return out

    # If response looks like a dict (bridge path)
    if isinstance(response, dict):
        # Common bridge keys
        out["model"] = _try_getitem(response, "model", "model_name") or (
            request_meta.get("model") if request_meta else None
        )
        # Try direct text
        out_text = _try_getitem(response, "text", "output", "contents")
        if isinstance(out_text, str):
            out["text"] = out_text
        else:
            # If output is list/dict, try to drill into first candidate
            cands = response.get("candidates") or response.get("outputs") or response.get("output")
            if isinstance(cands, list) and cands:
                t = _as_text_from_candidate(cands[0])
                if t:
                    out["text"] = t

        # function_call may live under candidate->function_call or top-level
        func = _try_getitem(response, "function_call", "functionCall")
        if func is None and isinstance(cands, list) and cands:
            func = _try_getitem(cands[0], "function_call", "functionCall")
        out["function_call"] = func

        out["finish_reason"] = _try_getitem(response, "finish_reason", "reason")
        out["usage"] = response.get("usage")
        out["safety"] = response.get("safety") or response.get("safetyAttributes")
        return out

    # Otherwise, try SDK-style objects with attributes
    try:
        # model
        model = _try_getattr(response, "model", "model_name")
        if not model and request_meta:
            model = request_meta.get("model")
        out["model"] = model

        # Some SDKs provide text attribute directly
        text = _try_getattr(response, "text", "content", "output")
        if isinstance(text, str):
            out["text"] = text
        else:
            # candidates (common in GenAI SDKs)
            candidates = _try_getattr(response, "candidates", "outputs", "output")
            if candidates:
                # candidates may be a list-like
                try:
                    first = candidates[0]
                except Exception:
                    first = None
                t = _as_text_from_candidate(first)
                if t:
                    out["text"] = t

                # function call inside candidate
                func = _try_getattr(first, "function_call", "functionCall")
                if func is not None:
                    out["function_call"] = func

        # finish reason
        finish = _try_getattr(response, "finish_reason", "reason")
        # Some SDK finish_reason may be an enum with a name
        try:
            if hasattr(finish, "name"):
                finish = finish.name
        except Exception:
            pass
        out["finish_reason"] = finish

        # usage & safety
        usage = _try_getattr(response, "usage") or _try_getattr(response, "usage_metadata")
        out["usage"] = usage
        safety = _try_getattr(response, "safety") or _try_getattr(response, "safety_attributes")
        out["safety"] = safety

    except Exception:
        # best-effort: return what we have plus raw
        return out

    return out
