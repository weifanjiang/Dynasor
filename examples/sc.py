from collections import Counter
from openai import OpenAI
from gsm8k_utils import extract_answer, math_equal, majority_voting, load_jsonl
from concurrent.futures import ThreadPoolExecutor

# Initialize OpenAI client with vLLM's API server
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:30000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def get_self_consistent_response(model: str, messages: list, n_samples: int = 3, sampling_params: dict = None):
    # TODO(GindaChen) Get rid of the model name paramter - put it as a part of other parameters
    """
    Generate multiple completions for a prompt and determine the most consistent result.

    Parameters:
    - model (str): The model name.
    - messages (list): The input messages.
    - n_samples (int): The number of completions to generate.

    Returns:
    - list[str]: The list of completions.
    """
    responses = []
    
    sampling_params = sampling_params or dict(
        max_tokens=512,
        temperature=0.7,
    )

    # Generate multiple completions
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        n=n_samples,
        **sampling_params,
    )
    for choice in completion.choices:
        text = choice.message.content
        responses.append(text)

    return responses



def load_gsm8k_questions():
    return load_jsonl("data/GSM8K/test.jsonl")

def prepare_prompt_gsm8k(question: str) -> str:
    """
    Prepare the prompt for the model with system and user messages.
    
    Parameters:
    - question (str): The input question to be answered
    
    Returns:
    - str: The formatted prompt with system and user messages
    """
    system_msg = "You are a helpful assistant."
    user_msg = f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    
    return messages

# Example usage
def test_one_sc_gsm8k():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    question = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    messages = prepare_prompt_gsm8k(question)
    sampling_params = dict(
        temperature=0.7,  # Enable some variability in responses
        max_tokens=512,
    )
    responses = get_self_consistent_response(
        model=model_name,
        messages=messages,
        n_samples=3,
        sampling_params=sampling_params,
    )
    print(f"Generated responses: {responses}")
    # Extract answers and get majority vote
    answers = [extract_answer(r) for r in responses]
    print(f"Answers: {answers}")
    final_answer = majority_voting(answers)
    print(f"Final answer: {final_answer}")
    return final_answer

def sc_gsm8k_process_item(item, model_name, n_samples, sampling_params):
    question = item["question"]
    _answer = item["answer"]
    answer = extract_answer(_answer)
    messages = prepare_prompt_gsm8k(question)
    
    responses = get_self_consistent_response(
        model=model_name,
        messages=messages,
        n_samples=3,
        sampling_params=sampling_params,
    )
    answers = [extract_answer(r) for r in responses]
    final_answer = majority_voting(answers)
    # print(f"Final answer: {final_answer}")
    # print(f"Answer: {answer}")
    return dict(
        is_equal=math_equal(final_answer, answer),
        ground_truth_answer=answer,
        final_answer=final_answer,
    )

def test_sc_gsm8k(
    is_parallel: bool = False,
    n_samples: int = 3,
    n_dataset_rows: int = 5,
):
    # TODO(GindaChen) Make this multithreading
    gsm8k_test_dataset = load_gsm8k_questions()
    gsm8k_test_dataset = gsm8k_test_dataset[:n_dataset_rows]

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    sampling_params = dict(
        temperature=0.7,  # Enable some variability in responses
        max_tokens=512,
    )
    
    if is_parallel:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            # TODO(GindaChen) Make this using `map`?
            for item in gsm8k_test_dataset:
                future = executor.submit(
                    sc_gsm8k_process_item,
                    item,
                    model_name,
                    n_samples,
                    sampling_params,
                )
                futures.append(future)
            results = [f.result() for f in futures]
    else:
        results = [sc_gsm8k_process_item(
            item,
            model_name,
            n_samples,
            sampling_params,
        ) for item in gsm8k_test_dataset]

    correct_count = 0
    for result in results:
        if result["is_equal"]:
            correct_count += 1
    
    accuracy = correct_count / len(gsm8k_test_dataset)
    print(f"Accuracy: {correct_count} / {len(gsm8k_test_dataset)} = {accuracy * 100:.2f}%")
    return dict(
        accuracy=accuracy,
        results=results,
    )


if __name__ == "__main__":
    test_sc_gsm8k(is_parallel=True)
