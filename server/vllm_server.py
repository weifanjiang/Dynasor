from openai import OpenAI
from transformers import AutoTokenizer

from entropy import eqaul_group, obtaint_answer

openai_api_key = "dr32r34tnjnfkd"
openai_api_base = "http://localhost:8000/v1"
model = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
tokenizer = AutoTokenizer.from_pretrained(model)


def format_deepseek_prompt(user_message: str) -> str:
    """Format prompt with DeepSeek template"""
    return f"<｜begin▁of▁sentence｜><｜User｜>{user_message}<｜Assistant｜><think>\n"


uncertain_words = ['wait', 'hold', 'but', 'okay', 'no', 'hmm']
DEFAULT_PROBE_SUFFIX = '... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{'
DEFAULT_PROBE_SUFFIX_2 = '}\n\\]' + "\n"

import logging


def init_logging():
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    return logger


logger = init_logging()


def should_early_exit(
        answers: list[str],
        probe_response_text: str,
        uncertain_words: list[str],
        continue_certain_bar: int,
) -> bool:
    """
    Check if the answer is consistent or certain.
    1. Number of answers should be greater than the threshold
    2. The probe response text should not contain any uncertain words
    3. The answers should be consistent
    """

    # Number of answers should be greater than the threshold
    if len(answers) < continue_certain_bar:
        return False

    # The probe response text should not contain any uncertain words
    probe_response_text_lower = probe_response_text.lower()
    if any(word in probe_response_text_lower for word in uncertain_words):
        return False

    # The answers should be consistent - may need to use some small model to verify this.
    if not eqaul_group(answers):
        return False

    return True


def get_completion(user_message: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
    """Get completion from the model using OpenAI API"""

    prompt = format_deepseek_prompt(user_message)
    answers = []
    history: list[str] = [prompt]

    # Maximum length of the prompt
    max_len = 2048
    # Window of the certainty answer to check against
    continue_certain_bar = 2
    # Number of tokens to detect
    detect_tokens = 32
    probe_suffix_text = DEFAULT_PROBE_SUFFIX
    probe_suffix_text_2 = DEFAULT_PROBE_SUFFIX_2

    while True:
        # Prompt the model to get 32 tokens
        prompt = "".join(history)
        logger.info(f"Prompt: {prompt}")
        response = client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=detect_tokens, top_p=0.95,
        )

        text = response.choices[0].text
        history.append(text)

        # Probe the model to see if it can get the answer 
        message_sending = "".join(history)
        logger.info(f"Message sending: {message_sending}")
        probe_response = client.completions.create(
            model=model,
            temperature=0.6,
            prompt=message_sending + probe_suffix_text,
            stream=True, max_tokens=20, top_p=0.95,
        )

        probe_response_text = probe_response.choices[0].text
        logger.info(f"Probe response: {probe_response_text}")

        # Get the answer from the probe response
        answer = obtaint_answer(probe_response_text)
        answers.append(answer)
        logger.info(f"Answer: {answer}")

        if should_early_exit(answers, probe_response_text, uncertain_words, continue_certain_bar):
            break

    # if chunk.choices[0].finish_reason is not None and chunk.choices[0].finish_reason != 'length': break
    return text


# Example usage
user_message = "2 + 2 = ?"
response = get_completion(user_message)
logger.info(f"Response: {response}")
