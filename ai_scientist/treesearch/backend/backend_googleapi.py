import os
import time
from typing import Any, Dict

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

def extract_gemini_function_call(norm: dict) -> Dict[str, Any]:
    """
    Return a dict with the arguments of the first Gemini function call.
    Works whether `norm["raw"]` is a dict or a GenerateContentResponse object.
    """
    raw = norm.get("raw")
    if raw is None:
        return {}

    # ---- turn the protobuf‑style object into a dict ----
    # most Gemini objects expose a `to_dict()` method; fall back to attribute access
    if hasattr(raw, "to_dict"):
        raw_dict = raw.to_dict()
    else:
        # build a shallow dict from public attributes
        raw_dict = {k: getattr(raw, k) for k in dir(raw)
                    if not k.startswith("_") and not callable(getattr(raw, k))}

    # ---- locate the first candidate ----
    candidates = raw_dict.get("candidates", [])
    if not candidates:
        return {}

    first = candidates[0]
    parts = first.get("content", {}).get("parts", [])

    # ---- find a part that contains a function call ----
    for part in parts:
        func = part.get("function_call") or part.get("functionCall")
        if not func:
            continue

        # ---- extract arguments (Gemini uses the key "args") ----
        args = func.get("args") or func.get("arguments")
        if isinstance(args, dict):
            arg_dict = args
        elif isinstance(args, str):
            try:
                arg_dict = json.loads(args)
            except json.JSONDecodeError:
                arg_dict = {"_raw_args_string": args}
        else:
            arg_dict = {}

        # add a little meta‑info (optional, helps debugging)
        arg_dict["_function_name"] = func.get("name")
        return arg_dict

    return {}

# NOTE: The function signature is maintained exactly as requested.
def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Queries the Gemini API with modernized SDK practices.
    
    Returns:
        (output, req_time, in_tokens, out_tokens, info)
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
        # Use the system_instruction argument for best practice
        system_instruction=system_message,
        tools=tools,
    )

    # 4. Prepare contents for a single-turn generate_content call
    # --- FIX: Ensure `contents` is correctly structured for the API call ---
    contents: list[dict[str, Any]] = []

    # The system message is handled by system_instruction in the model setup (Step 3).
    # The `contents` list only needs the user's input for a new turn.
    if user_message:
        # `parts` must be a list of dictionaries with a "text" key for the SDK
        contents = [
            {"role": "user", "parts": [{"text": str(user_message)}]}
        ]
    else:
        # Fallback if no user message is provided
        print("[yellow]Warning: Query received with no user message. Using fallback content.[/yellow]")
        contents = [{"role": "user", "parts": [{"text": "Hello."}]}]
    
    # NOTE: If you needed to support multi-turn history, this logic would need to
    # be expanded to map the full `messages_list` into `Content` objects.
    # We prioritize the single-turn query structure here.
    # ---------------------------------------------------------------------

    t0 = time.time()

    # 5. Call the API with backoff/retry logic
    response = backoff_create(
        model.generate_content,
        GEMINI_RETRY_EXCEPTIONS,
        # Pass the correctly structured contents list
        contents=contents,
        generation_config=generation_config,
        # Pass tool_config if tools are present
        tool_config={"function_calling_config": "ANY"} if tools else None,
    )

    print(f"Vispi Response In backend: \n{response}")

    req_time = time.time() - t0

    # 6. Parse the response
    # Normalize into canonical dict
    norm = normalize_genai_response(response, request_meta={"model": model_name})

    print(f"Vispi Norm In backend: \n{norm}")

    # Optional GEMINI_DEBUG: dump masked request/response for debugging
    try:
        if os.getenv("GEMINI_DEBUG") in ("1", "true", "True"):
            # Mask potential secrets: don't dump API keys, only model and truncated text
            dbg = {
                "model": model_name,
                # Use the contents variable for the request preview
                "request_preview": contents[:2],
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

    # 7. Tokens / usage extraction (BEST PRACTICE)
    # Token counts are reliably located in the response's usage_metadata,
    # which is assumed to be standardized by `normalize_genai_response` into the 'usage' key.
    usage = norm.get("usage") or {}

    # Safely extract prompt_token_count
    in_tokens = getattr(usage, "prompt_token_count", None)
    if in_tokens is None and isinstance(usage, dict):
        in_tokens = usage.get("prompt_token_count")

    # Safely extract candidates_token_count
    out_tokens = getattr(usage, "candidates_token_count", None)
    if out_tokens is None and isinstance(usage, dict):
        out_tokens = usage.get("candidates_token_count")

    info = {
        "model": norm.get("model") or model_name,
        "finish_reason": norm.get("finish_reason"),
        "safety": norm.get("safety"),
        "raw": norm.get("raw"),
    }

    # 8. Determine final output
    output = extract_gemini_function_call(norm)

    if not output:                     # fallback to plain text if no call was found
        output = norm.get("text")
    
    print(f"Vispi Output In backend: \n{output}")
    print(f"Vispi Info In backend: \n{info}")
    
    return output, req_time, in_tokens, out_tokens, info
