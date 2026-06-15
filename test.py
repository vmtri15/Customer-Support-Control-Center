from openai import OpenAI
import requests
import time

# Đổi port theo vLLM server của bạn
VLLM_BASE_URL = "https://ai-api.inetcloud.vn/openfang/v1"

# Tên model phải trùng với --served-model-name
# Ví dụ bạn serve bằng: --served-model-name vision
MODEL_NAME = "knowledge"

client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key="EMPTY"  # Nếu vLLM không bật --api-key thì để gì cũng được
)

def check_server():
    try:
        res = requests.get(f"{VLLM_BASE_URL}/models", timeout=10)
        res.raise_for_status()
        print("✅ vLLM server is running")
        print(res.json())
    except Exception as e:
        print("❌ Cannot connect to vLLM server:", e)
        return False
    return True


def test_chat():
    start = time.time()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Bạn là trợ lý AI hữu ích, trả lời ngắn gọn bằng tiếng Việt."
            },
            {
                "role": "user",
                "content": "Hãy giải thích vLLM là gì trong 3 câu."
            }
        ],
        temperature=0.8
    )

    end = time.time()
    print(response)
    output = response.choices[0].message.content

    print("\n===== MODEL OUTPUT =====")
    print(output)

    print("\n===== STATS =====")
    print(f"Time: {end - start:.2f}s")

    if hasattr(response, "usage") and response.usage:
        print("Prompt tokens:", response.usage.prompt_tokens)
        print("Completion tokens:", response.usage.completion_tokens)
        print("Total tokens:", response.usage.total_tokens)


if __name__ == "__main__":
    if check_server():
        test_chat()