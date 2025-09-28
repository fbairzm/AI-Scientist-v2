"""Simple test runner for normalize_genai_response.

These tests are lightweight and do not require pytest. Run via
`python -m ai_scientist.treesearch.backend.tests.test_response_utils`.
"""
from __future__ import annotations
import json
import sys

from ai_scientist.treesearch.backend.response_utils import normalize_genai_response


class SimpleObj:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_sdk_text():
    resp = SimpleObj(text="hello world", model="gemini-1")
    out = normalize_genai_response(resp)
    assert out["text"] == "hello world", out
    assert out["model"] == "gemini-1", out
    print("test_sdk_text: PASS")


def test_sdk_candidates_function():
    func = {"name": "foo", "arguments": '{"x":1}'}
    candidate = SimpleObj(text=None, function_call=func)
    resp = SimpleObj(candidates=[candidate], model="gemini-2")
    out = normalize_genai_response(resp)
    assert out["function_call"] == func, out
    # text may be None
    print("test_sdk_candidates_function: PASS")


def test_bridge_dict():
    bridge = {
        "model": "gemini-bridge-1",
        "candidates": [{"text": "bridge reply"}],
        "usage": {"prompt_tokens": 5},
    }
    out = normalize_genai_response(bridge)
    assert out["text"] == "bridge reply", out
    assert out["usage"] == {"prompt_tokens": 5}, out
    print("test_bridge_dict: PASS")


def run_all():
    test_sdk_text()
    test_sdk_candidates_function()
    test_bridge_dict()


if __name__ == "__main__":
    try:
        run_all()
        print("ALL TESTS PASS")
    except AssertionError as e:
        print("TEST FAILURE:", e)
        sys.exit(2)