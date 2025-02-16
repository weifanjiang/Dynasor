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
    # Define color formatting for different log levels
    logging.basicConfig(
        level=logging.INFO,
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


def guard_prompt_len(prompt: str, max_len: int) -> str:
    """Guard the prompt length"""
    prompt_len = len(tokenizer.encode(prompt))
    if prompt_len > max_len:
        raise ValueError(f"Prompt length exceeded: {prompt_len} > {max_len}")
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
    logger.info(f"Max prompt length: {max_prompt_len}")

    prompt = format_deepseek_prompt(user_message)
    answers = []
    history: list[str] = [prompt]

    # TODO: (1) Ensure prompt max token is not exceeded; 

    while True:
        # Prompt the model to get 32 tokens
        prompt = "".join(history)
        prompt = guard_prompt_len(prompt, max_prompt_len)
        response = client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=detect_tokens, 
            top_p=0.95,
        )

        text = response.choices[0].text
        history.append(text)

        # Probe the model to see if it can get the answer 
        message_sending = "".join(history) + probe_suffix_text
        message_sending = guard_prompt_len(message_sending, max_prompt_len)
        logger.info(f"Message sending: {repr(message_sending)}")
        probe_response = client.completions.create(
            model=model,
            temperature=0.6,
            prompt=message_sending,
            max_tokens=20, top_p=0.95,
        )

        probe_response_text = probe_response.choices[0].text
        logger.info(f"Probe response: {repr(probe_response_text)}")

        # Get the answer from the probe response
        answer = obtaint_answer(probe_response_text)
        answers.append(answer)
        logger.info(f"Answer: {repr(answer)}")

        if should_early_exit(answers, probe_response_text, uncertain_words, continue_certain_bar):
            logger.info("Early exit")
            final_text = message_sending + probe_response_text
            return final_text, answers

    text = "".join(history)
    return text, answers


# Example usage
user_message = "2 + 2 = ?"
response, answers = get_completion(user_message)
logger.info(f"Response: {repr(response)}")
logger.info(f"Answers: {repr(answers)}")

final_answer = answers[-1]
logger.info(f"Final answer: {repr(final_answer)}")
