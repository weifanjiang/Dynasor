"""dyansor-vLLM client API example"""

import openai

openai_api_key = "dr32r34tnjnfkd"
openai_api_base = "http://localhost:8000/v1"
model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
client = openai.OpenAI(
    base_url=openai_api_base,
    api_key=openai_api_key,
)

probe_text = "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{"
probe_text_end = "} \\]"


def format_deepseek_prompt(user_message: str) -> str:
    """Format prompt with DeepSeek template"""
    return f"<｜begin▁of▁sentence｜><｜User｜>{user_message}<｜Assistant｜><think>\n"


def completions_example():
    user_message = "Solve x^4 + x^3 = x^2 + x + 4"
    prompt = format_deepseek_prompt(user_message)

    response = client.completions.create(
        model=model,
        prompt=prompt,
        stream=True,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        extra_body=dict(
            adaptive_compute=dict(
                mode="prompting",
                # TODO: Properly form the names
                probe_text=probe_text,
                probe_text_end=probe_text_end,
                certainty_window=2,
                token_interval=32,
            )
        ),
    )

    for chunk in response:
        token = chunk.choices[0].text
        print(token, end="", flush=True)


def chat_example():
    raise NotImplementedError("Chat example is not implemented yet")
    user_message = "Solve x^4 + x^3 = x^2 + x + 4"
    prompt = format_deepseek_prompt(user_message)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        extra_body=dict(
            adaptive_compute=dict(
                mode="prompting",
                # TODO: Properly form the names
                probe_text=probe_text,
                probe_text_end=probe_text_end,
                certainty_window=2,
                token_interval=32,
            )
        ),
    )

    for chunk in response:
        token = chunk.choices[0].text
        print(token, end="", flush=True)


def main():
    completions_example()


if __name__ == "__main__":
    main()
