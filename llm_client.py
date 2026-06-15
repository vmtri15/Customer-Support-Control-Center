import os
import time

from openai import OpenAI

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://ai-api.inetcloud.vn/openfang/v1")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("LLM_API_KEY", "EMPTY"))
LLM_MODEL = os.getenv("LLM_MODEL", "knowledge")

_client = None


def get_llm_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    return _client


def chat_completion(prompt: str, system_prompt: str | None = None, max_tokens: int | None = None, temperature: float = 0.8) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    response = get_llm_client().chat.completions.create(**kwargs)
    
    choice = response.choices[0]
    content = choice.message.content
    
    if not content or not content.strip():
        raise RuntimeError(f"LLM returned an empty response. Finish reason: {choice.finish_reason}")
        
    return content.strip()


def chat_completion_with_metrics(prompt: str, system_prompt: str | None = None, max_tokens: int | None = None, temperature: float = 0.8) -> dict:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    start_time = time.perf_counter()
    response = get_llm_client().chat.completions.create(**kwargs)
    latency = time.perf_counter() - start_time
    
    choice = response.choices[0]
    content = choice.message.content
    
    if not content or not content.strip():
        raise RuntimeError(f"LLM returned an empty response. Finish reason: {choice.finish_reason}")
        
    content = content.strip()
    input_tokens = len(prompt) // 4
    output_tokens = len(content) // 4
    
    return {
        "text": content,
        "latency_seconds": round(latency, 2),
        "tokens_per_second": round(output_tokens / latency, 1) if latency > 0 else 0
    }


def chat_completion_stream(prompt: str, system_prompt: str | None = None, temperature: float = 0.8):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = get_llm_client().chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
