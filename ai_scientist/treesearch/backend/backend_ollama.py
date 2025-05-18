import time
import json
import logging
from typing import Any
from ollama import chat as ollama_chat
from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create

def prompt_to_str(prompt):
    if isinstance(prompt, dict):
        # Join all key-value pairs as "Key: Value" lines
        return "\n".join(f"{k}: {v}" for k, v in prompt.items())
    return str(prompt) if prompt is not None else ""

def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: Any = None,  # Ollama does not support function calling, so this is ignored
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query the Ollama local model using the chat API.

    Returns:
        output: The assistant's message content (str)
        req_time: Time taken for the request (float)
        in_tokens: Placeholder (int, always 0)
        out_tokens: Placeholder (int, always 0)
        info: Dict with model info
    """
    model = model_kwargs.pop("model", None)
    # Compose messages in OpenAI-style format, always as strings
    messages =  opt_messages_to_list(system_message, user_message)
    # Replace max_tokens only if it is None
    if model_kwargs.get("max_tokens") is None:
        model_kwargs["max_tokens"] = 16*1023 # 16k - 16 tokens
    if model_kwargs.get("temperature") is None:
        model_kwargs["temperature"] = 0.75 # 16k - 16 tokens

    t0 = time.time()
    response = ollama_chat(
        model=model,
        messages=messages,
        options=model_kwargs
    )
    req_time = time.time() - t0

    logging.debug(f"Response from Ollama: {response}") 

    # Ollama returns a dict with 'message' key
    output = response["message"]["content"]

    # If a function specification was provided, try to simulate a tool call.
    if func_spec is not None:
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict) and "tool" in parsed and parsed["tool"] == func_spec.name:
                output = parsed
        except json.JSONDecodeError as e: 
            logging.warning(
                f"Tool call simulation: Could not parse output as JSON for func_spec '{func_spec.name}': {e}. Returning raw output."
            )
            output = {"tool": func_spec.name, "arguments": output}

    info = {
        "model": model,
        "created": int(time.time()),
    }

    # Ollama does not return token usage, so set to 0
    in_tokens = 0
    out_tokens = 0

    return output, req_time, in_tokens, out_tokens, info