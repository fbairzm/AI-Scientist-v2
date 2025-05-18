import re
from . import backend_anthropic, backend_openai
from . import backend_ollama
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
    }

    # Handle models with beta limitations
    # ref: https://platform.openai.com/docs/guides/reasoning/beta-limitations
    if model.startswith("o1"):
        if system_message and user_message is None:
            user_message = system_message
        elif system_message is None and user_message:
            pass
        elif system_message and user_message:
            system_message["Main Instructions"] = {}
            system_message["Main Instructions"] |= user_message
            user_message = system_message
        system_message = None
        # model_kwargs["temperature"] = 0.5
        model_kwargs["reasoning_effort"] = "high"
        model_kwargs["max_completion_tokens"] = 100000  # max_tokens
        # remove 'temperature' from model_kwargs
        model_kwargs.pop("temperature", None)
    else:
        model_kwargs["max_tokens"] = max_tokens

 
    # Determine the backend to use based on the model name
    # Default use OpenAI or Anthropic, use the respective backend 
    query_func = backend_openai.query 
    if "claude-" in model :
        # If the model is Claude, use the Anthropic backend
        query_func = backend_anthropic.query
    elif  model.startswith("ollama/") or  re.match(r"^(gemma3|devstral|qwen|deepcoder|phi4|command-r7b|deepscaler|deepseek)", model):
        # If the model is Ollama, use the Ollama backend
        query_func = backend_ollama.query
        model_kwargs["model"] = model.split("/", 1)[1]  # Remove "ollama/" prefix

    
    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message=compile_prompt_to_md(system_message) if system_message else None,
        user_message=compile_prompt_to_md(user_message) if user_message else None,
        func_spec=func_spec,
        **model_kwargs,
    )

    return output
