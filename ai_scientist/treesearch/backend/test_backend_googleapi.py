import time
import types

from ai_scientist.treesearch.backend import backend_googleapi as bg


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = types.SimpleNamespace(content=content, tool_calls=None)


class FakeUsage:
    def __init__(self):
        self.prompt_tokens = 1
        self.completion_tokens = 2


class FakeCompletion:
    def __init__(self, content="fake reply"):
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsage()
        self.system_fingerprint = "fp"
        self.model = "gemini-fake"
        self.created = int(time.time())


class FakeCompletionsAPI:
    def create(self, *args, **kwargs):
        # Accepts messages/input etc and returns a FakeCompletion
        return FakeCompletion()


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletionsAPI()


class FakeClient:
    def __init__(self):
        self.chat = FakeChat()


def test_googleapi_query_mocked():
    # Monkeypatch the client factory to return a fake client that doesn't call network
    bg._create_gemini_client = lambda: FakeClient()

    output, req_time, in_tok, out_tok, info = bg.query(
        system_message="System here",
        user_message="User here",
        model="gemini-2.5-flash",
        max_tokens=10,
    )

    assert isinstance(output, str)
    assert output == "fake reply"
    assert isinstance(req_time, float)
    assert in_tok == 1
    assert out_tok == 2
    assert info["model"] == "gemini-fake"
