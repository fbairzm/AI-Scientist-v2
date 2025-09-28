import time
import types
import json
import os

from google.genai.errors import APIError as GenaiAPIError

import ai_scientist.treesearch.backend.backend_googleapi as bg


class FakeResponse:
    def __init__(self):
        self.text = "fake gemini reply"
        self.usage_metadata = types.SimpleNamespace(prompt_token_count=1, candidates_token_count=2)
        self.candidates = [types.SimpleNamespace(finish_reason=types.SimpleNamespace(name="COMPLETE"))]
        self.model = "gemini-fake"


def test_gemini_client_path():
    # Stub the genai client to return a fake response quickly and avoid backoff
    class FakeClient:
        class models:
            @staticmethod
            def generate_content(*args, **kwargs):
                return FakeResponse()

    # Inject fake client and a simple backoff that directly calls the function
    bg._client = FakeClient()
    bg.backoff_create = lambda fn, excs, **kw: fn(**kw)

    out, rt, in_tok, out_tok, info = bg.query(system_message=None, user_message="Hi there", model="gemini-2.5-flash", max_tokens=50)

    assert out == "fake gemini reply"
    assert info.get("model") == "gemini-fake"
