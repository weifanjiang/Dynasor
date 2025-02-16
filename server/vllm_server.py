import asyncio
from dataclasses import dataclass
from typing import AsyncGenerator

from openai import OpenAI, AsyncOpenAI
from transformers import AutoTokenizer

from entropy import eqaul_group, obtaint_answer, count_not_empty


@dataclass
class TokenResult:
    token: str


@dataclass
class CompletionResult:
    final_text: str
    answers: list[str]


openai_api_key = "dr32r34tnjnfkd"
openai_api_base = "http://localhost:8000/v1"
model = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
async_client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_api_base)
tokenizer = AutoTokenizer.from_pretrained(model)


def format_deepseek_prompt(user_message: str) -> str:
    """Format prompt with DeepSeek template"""
    return f"<｜begin▁of▁sentence｜><｜User｜>{user_message}<｜Assistant｜><think>\n"


uncertain_words = ['wait', 'hold', 'but', 'okay', 'no', 'hmm']
DEFAULT_PROBE_SUFFIX = '... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{'
DEFAULT_PROBE_SUFFIX_2 = '}\n\\]' + "\n"

import logging


def init_logging():
    # Define color formatting for different log levels
    logging.basicConfig(
        # level=logging.INFO,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler()
        ]
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    # Add colors to different logging levels
    logging.addLevelName(
        logging.INFO,
        "\033[32m%s\033[0m" % logging.getLevelName(logging.INFO)
    )  # Green
    logging.addLevelName(
        logging.DEBUG,
        "\033[34m%s\033[0m" % logging.getLevelName(logging.DEBUG)
    )  # Blue
    logging.addLevelName(
        logging.ERROR,
        "\033[31m%s\033[0m" % logging.getLevelName(logging.ERROR)
    )  # Red
    logging.addLevelName(
        logging.WARNING,
        "\033[33m%s\033[0m" % logging.getLevelName(logging.WARNING)
    )  # Yellow

    logger = logging.getLogger(__name__)
    return logger


logger = init_logging()


def has_value(x) -> bool:
    if x is None:
        return False
    if isinstance(x, str):
        return len(x.strip()) > 0
    if isinstance(x, list):
        return len(x) > 0
    return True


def should_early_exit(
    answers: list[str],
    probe_response_text: str,
    uncertain_words: list[str],
    continue_certain_bar: int,
    is_certains: list[bool],
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

    # The last answer window should be consistent
    answer_candidates = answers[-continue_certain_bar:]
    is_certains = is_certains[-continue_certain_bar:]
    if eqaul_group(answer_candidates):
        if count_not_empty(answer_candidates) == continue_certain_bar:
            if sum(is_certains) == continue_certain_bar:
                logger.debug(f"Early exit on: {answer_candidates = } ({is_certains = })")
                return True

    return True


class PromptLengthExceeded(Exception):
    """Prompt length exceeded"""

    def __init__(self, prompt_len: int, max_len: int):
        self.prompt_len = prompt_len
        self.max_len = max_len
        super().__init__(f"Prompt length exceeded: {prompt_len} > {max_len}")


def is_certain_answer(probe_response_text: str, uncertain_words: list[str]) -> bool:
    """Check if the answer is certain"""
    return not any(word in probe_response_text.lower() for word in uncertain_words)


def guard_prompt_len(prompt: str, max_len: int) -> str:
    """Guard the prompt length"""
    prompt_len = len(tokenizer.encode(prompt))
    if prompt_len > max_len:
        raise PromptLengthExceeded(prompt_len, max_len)
    return prompt


def get_completion(
    user_message: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    continue_certain_bar: int = 2,
    detect_tokens: int = 32,
    probe_suffix_text: str = DEFAULT_PROBE_SUFFIX,
) -> tuple[str, list[str]]:
    """
    Get completion from the model using OpenAI API

    Args:
        user_message: The user message to complete
        temperature: The temperature of the model
        max_tokens: The maximum number of tokens to generate
        max_len: The maximum length of the prompt
        continue_certain_bar: The number of answers to continue the prompt
        detect_tokens: The number of tokens to detect
        probe_suffix_text: The suffix text to probe the model

    Returns:
        final_text: The final text of the completion
        answers: The answers from the model
    """

    max_prompt_len: int = tokenizer.model_max_length
    logger.debug(f"Max prompt length: {max_prompt_len}")

    prompt = format_deepseek_prompt(user_message)
    answers = []
    is_certains = []
    history: list[str] = [prompt]

    try:
        while True:
            # (1) Prompt the model to get 32 tokens
            response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=detect_tokens,
                top_p=0.95,
            )

            output_text = response.choices[0].text
            history.append(output_text)

            # (2) Probe the model to see if it can get the answer
            prompt += output_text
            message_sending = prompt + probe_suffix_text
            logger.debug(f"Message sending: {repr(message_sending)}")
            probe_response = client.completions.create(
                model=model,
                temperature=0.6,
                prompt=message_sending,
                max_tokens=20, top_p=0.95,
            )

            probe_response_text = probe_response.choices[0].text
            logger.debug(f"Probe response: {repr(probe_response_text)}")

            # (3) Get the answer from the probe response
            answer = obtaint_answer(probe_response_text)
            answers.append(answer)
            is_certain = is_certain_answer(probe_response_text, uncertain_words)
            is_certains.append(is_certain)
            logger.debug(f"Answer: {repr(answer)}")

            if should_early_exit(answers, probe_response_text, uncertain_words, continue_certain_bar, is_certains):
                logger.debug("Early exit")
                final_text = message_sending + probe_response_text
                return final_text, answers

    except PromptLengthExceeded as e:
        pass

    logger.debug("Max output token length exceeded...")
    text = "".join(history)
    return text, answers


async def get_completion_async(
    user_message: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    continue_certain_bar: int = 2,
    detect_tokens: int = 32,
    probe_suffix_text: str = DEFAULT_PROBE_SUFFIX,
    probe_suffix_text_end: str = DEFAULT_PROBE_SUFFIX_2,
    format_final_answer: bool = True,
) -> AsyncGenerator[CompletionResult, None]:
    """
    Get completion from the model using OpenAI API

    Args:
        user_message: The user message to complete
        temperature: The temperature of the model
        max_tokens: The maximum number of tokens to generate
        max_len: The maximum length of the prompt
        continue_certain_bar: The number of answers to continue the prompt
        detect_tokens: The number of tokens to detect
        probe_suffix_text: The suffix text to probe the model

    Returns:
        final_text: The final text of the completion
        answers: The answers from the model
    """

    max_prompt_len: int = tokenizer.model_max_length
    logger.debug(f"Max prompt length: {max_prompt_len}")

    prompt = format_deepseek_prompt(user_message)
    answers = []
    is_certains = []
    history: list[str] = [prompt]

    try:
        while True:
            # (1) Prompt the model to get 32 tokens
            response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=detect_tokens,
                top_p=0.95,
                stream=True,
            )

            output_text = ""
            for chunk in response:
                token = chunk.choices[0].text
                output_text += token
                yield TokenResult(token)

            history.append(output_text)

            # (2) Probe the model to see if it can get the answer
            prompt += output_text
            message_sending = prompt + probe_suffix_text
            logger.debug(f"Message sending: {repr(message_sending)}")
            probe_response = client.completions.create(
                model=model,
                temperature=0.6,
                prompt=message_sending,
                max_tokens=20, top_p=0.95,
            )

            probe_response_text = probe_response.choices[0].text
            logger.debug(f"Probe response: {repr(probe_response_text)}")

            # (3) Get the answer from the probe response
            answer = obtaint_answer(probe_response_text)
            answers.append(answer)
            is_certain = is_certain_answer(probe_response_text, uncertain_words)
            is_certains.append(is_certain)
            logger.debug(f"Answer: {repr(answer)}")

            if should_early_exit(answers, probe_response_text, uncertain_words, continue_certain_bar, is_certains):
                logger.debug("Early exit")
                if not format_final_answer:
                    yield TokenResult(probe_suffix_text)
                    yield TokenResult(probe_response_text)
                else:
                    yield TokenResult(probe_suffix_text)
                    yield TokenResult(answer)
                    yield TokenResult(probe_suffix_text_end)

                    # Get response text after </think> if present
                    # Remove anything before </think> if present
                    import re
                    if "</think>" in probe_response_text:
                        after_think_text = re.sub(r'.*?</think>', '', probe_response_text)
                        yield TokenResult(after_think_text)

                final_text = message_sending + probe_response_text
                yield CompletionResult(final_text, answers)

    except PromptLengthExceeded as e:
        pass

    logger.debug("Max output token length exceeded...")
    text = "".join(history)
    yield CompletionResult(text, answers)


def main():
    # Example usage
    user_message = "What is the ultimate answer to the universe?"
    response, answers = get_completion(user_message)
    logger.info(f"Response: {repr(response)}")
    logger.info(f"Answers: {repr(answers)}")

    final_answer = answers[-1]
    logger.info(f"Final answer: {repr(final_answer)}")


async def main_async():
    user_message = "What is the ultimate answer to the universe?"
    logger.info(f"User message: {repr(user_message)}")
    logger.info("Start generating...")
    async for item in get_completion_async(
        user_message,
        format_final_answer=False,
    ):
        if isinstance(item, TokenResult):
            print(item.token, end="", flush=True)
        elif isinstance(item, CompletionResult):
            print()
            answers = item.answers
            final_text = item.final_text
            logger.info(f"Final text: {repr(final_text)}")
            logger.info(f"Answers: {repr(answers)}")
            break


if __name__ == "__main__":
    asyncio.run(main_async())
