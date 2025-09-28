import os
import time
from typing import Any

from funcy import notnone, select_values
from rich import print

import google.generativeai as genai 
from google.api_core import exceptions as google_exceptions
from google.generativeai.types import Tool, FunctionDeclaration
 

# Project-Specific Imports
from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from .response_utils import normalize_genai_response
import json
import os

# --- Client Setup ---
# Best practice: Configure once at the start of your application.
genai.configure() 

# --- Exceptions for Retry Logic ---
GEMINI_RETRY_EXCEPTIONS = (
    google_exceptions.GoogleAPIError,
    google_exceptions.ResourceExhausted,  # For rate limiting
    google_exceptions.DeadlineExceeded,   # For timeouts
)

def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Queries the Gemini API with modernized SDK practices.
    """
    # 1. Prepare model and generation config
    filtered_kwargs = select_values(notnone, model_kwargs)
    model_name = filtered_kwargs.pop('model', 'gemini-1.5-flash-latest')
    
    generation_config = genai.types.GenerationConfig(**filtered_kwargs)

    # 2. Prepare tools if a function is specified
    tools = None
    if func_spec:
        function_declaration = FunctionDeclaration(
            name=func_spec.name,
            description=func_spec.description,
            parameters=func_spec.json_schema,
        )
        tools = [Tool(function_declarations=[function_declaration])]

    # 3. Instantiate the model with its configuration
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_message,
        tools=tools,
    )

    # 4. Prepare chat history from the system and user messages
    messages_list = opt_messages_to_list(system_message, user_message)
    chat_history = [
        {"role": "model" if msg["role"] == "assistant" else "user", "parts": [msg["content"]]}
        for msg in messages_list
    ]

    # Ensure contents is never empty — the SDK will raise if it is.
    if not chat_history:
        fallback_text = user_message or system_message or ""
        # Parts should be a list of dicts with a text field for the generativeai SDK
        chat_history = [{"role": "user", "parts": [{"text": str(fallback_text)}]}]

    t0 = time.time()

    # 5. Call the API with backoff/retry logic
    response = backoff_create(
        model.generate_content, # ✅ THIS IS THE CORRECTED LINE
        GEMINI_RETRY_EXCEPTIONS,
        contents=chat_history,
        generation_config=generation_config,
        tool_config={"function_calling_config": "ANY"} if tools else None,
    )
    
    req_time = time.time() - t0

    # 6. Parse the response
    # Normalize into canonical dict
    norm = normalize_genai_response(response, request_meta={"model": model_name})

    # Optional GEMINI_DEBUG: dump masked request/response for debugging
    try:
        if os.getenv("GEMINI_DEBUG") in ("1", "true", "True"):
            # Mask potential secrets: don't dump API keys, only model and truncated text
            dbg = {
                "model": model_name,
                "request_preview": chat_history[:2],
                "response_preview": {
                    "text": (norm.get("text") or "")[:1000],
                    "finish_reason": norm.get("finish_reason"),
                    "function_call": norm.get("function_call"),
                },
            }
            dbg_path = os.getenv("GEMINI_DEBUG_PATH", "/tmp/gemini_debug.jsonl")
            with open(dbg_path, "a") as fh:
                fh.write(json.dumps(dbg) + "\n")
    except Exception:
        # Don't let debugging break normal execution
        pass

    # Tokens / usage extraction (best-effort)
    usage = norm.get("usage") or {}
    in_tokens = None
    out_tokens = None
    try:
        in_tokens = usage.get("prompt_token_count") if isinstance(usage, dict) else getattr(usage, "prompt_token_count", None)
    except Exception:
        in_tokens = None
    try:
        out_tokens = usage.get("candidates_token_count") if isinstance(usage, dict) else getattr(usage, "candidates_token_count", None)
    except Exception:
        out_tokens = None

    info = {
        "model": norm.get("model") or model_name,
        "finish_reason": norm.get("finish_reason"),
        "safety": norm.get("safety"),
        "raw": norm.get("raw"),
    }

    # If function_call present, return its args as output
    func_call = norm.get("function_call")
    if func_call:
        try:
            # function_call may be an object or dict
            if isinstance(func_call, dict):
                output = dict(func_call.get("args") or func_call.get("arguments") or {})
            else:
                args = getattr(func_call, "args", None) or getattr(func_call, "arguments", None)
                # args could be a JSON string or a mapping
                if isinstance(args, str):
                    try:
                        output = json.loads(args)
                    except Exception:
                        output = {"_raw_args": args}
                else:
                    output = dict(args or {})
            print(f"[cyan]Function call triggered (gemini): {getattr(func_call, 'name', func_call.get('name') if isinstance(func_call, dict) else None)}[/cyan]")
        except Exception:
            output = norm.get("text")
    else:
        output = norm.get("text")

    return output, req_time, in_tokens, out_tokens, info