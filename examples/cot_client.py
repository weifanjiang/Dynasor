import asyncio
from dataclasses import dataclass
from typing import AsyncGenerator, Union

from openai import OpenAI, AsyncOpenAI
from transformers import AutoTokenizer

from dynasor.server.entropy import (
    eqaul_group, obtain_answer, count_not_empty, should_early_exit,
    uncertain_words, is_certain_answer
)

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
        level=logging.INFO,
        # level=logging.DEBUG,
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




class PromptLengthExceeded(Exception):
    """Prompt length exceeded"""

    def __init__(self, prompt_len: int, max_len: int):
        self.prompt_len = prompt_len
        self.max_len = max_len
        super().__init__(f"Prompt length exceeded: {prompt_len} > {max_len}")




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
            answer = obtain_answer(probe_response_text)
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
            answer = obtain_answer(probe_response_text)
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


async def get_completion_async_2(
    user_message: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    continue_certain_bar: int = 2,
    detect_tokens: int = 32,
    probe_suffix_text: str = DEFAULT_PROBE_SUFFIX,
    probe_suffix_text_end: str = DEFAULT_PROBE_SUFFIX_2,
    format_final_answer: bool = False,
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

    task_queue = asyncio.Queue()
    exit_event = asyncio.Event()


    async def loop_continual_request():
        remaining_tokens = max_tokens
        running_prompt = prompt

        while remaining_tokens > 0:
            await asyncio.sleep(0.1)
            if remaining_tokens <= 0:
                break
            
            if exit_event.is_set():
                break

            response = await async_client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=detect_tokens,
                top_p=0.95,
                stream=True,
            )

            output_text = ""
            async for chunk in response:
                token = chunk.choices[0].text
                output_text += token
                remaining_tokens -= 1
                yield TokenResult(token)
            
            history.append(output_text)
            
            running_prompt += output_text
            
            task_queue.put_nowait(dict(
                prompt=running_prompt,
            ))
        pass


    async def loop_probe_request():
        # TODO: Ensure always get the last task.
        while True:
            if exit_event.is_set():
                break

            if task_queue.empty():
                await asyncio.sleep(0.1)
                continue
            
            task = task_queue.get_nowait()
            prompt = task["prompt"]
            message_sending = prompt + probe_suffix_text
            response = await async_client.completions.create(
                model=model,
                temperature=0.6,
                prompt=message_sending,
                max_tokens=20, 
                top_p=0.95,
            )
            probe_response_text = response.choices[0].text
            
            # (3) Get the answer from the probe response
            answer = obtain_answer(probe_response_text)
            answers.append(answer)
            is_certain = is_certain_answer(probe_response_text, uncertain_words)
            is_certains.append(is_certain)

            if should_early_exit(answers, probe_response_text, uncertain_words, continue_certain_bar, is_certains):
                yield TokenResult(probe_suffix_text)
                yield TokenResult(answer)
                yield TokenResult(probe_suffix_text_end)
                # if format_final_answer:
                #     # Get response text after </think> if present
                #     # Remove anything before </think> if present
                    
                #     if "</think>" in probe_response_text:
                #         after_think_text = re.sub(r'.*?</think>', '', probe_response_text)
                #         yield TokenResult(after_think_text)
                exit_event.set()

                final_text = message_sending + probe_response_text
                yield CompletionResult(final_text, answers)
                break
        pass


    
    output_queue = asyncio.Queue()
    exit_queue = asyncio.Queue()
    
    # Helper function to forward the yielded tokens from a generator into our output_queue.
    async def forward(
        gen: AsyncGenerator[Union[TokenResult, CompletionResult], None],
        queue: asyncio.Queue,
    ):
        async for item in gen:
            await queue.put(item)

    # Run both loops concurrently.
    task_continual = asyncio.create_task(forward(loop_continual_request(), output_queue))
    task_probe = asyncio.create_task(forward(loop_probe_request(), exit_queue))
    while not (task_continual.done() and task_probe.done() and output_queue.empty() and exit_queue.empty()):
        if not exit_queue.empty():
            yield await exit_queue.get()
            yield await exit_queue.get()
            yield await exit_queue.get()
            yield await exit_queue.get()
            break
        
        try:
            token = await asyncio.wait_for(output_queue.get(), timeout=0.01)
            yield token
        except asyncio.TimeoutError:
            continue

    # Wait for both tasks to finish (to propagate any exceptions)
    # await asyncio.gather(task_continual, task_probe)
    # end both tasks
    task_continual.cancel()
    task_probe.cancel()


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



async def main_async_2():
    user_message = "2+2="
    user_message = format_deepseek_prompt(user_message)
    logger.info(f"User message: {repr(user_message)}")
    logger.info("Start generating...")
    async for item in get_completion_async_2(
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
    # asyncio.run(main_async())
    asyncio.run(main_async_2())
