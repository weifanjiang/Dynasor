from abc import ABC, abstractmethod
import os
from openai import OpenAI
from collections import Counter
from openai import OpenAI
from gsm8k_utils import extract_answer, math_equal, majority_voting, load_jsonl
from concurrent.futures import ThreadPoolExecutor


class Client(ABC):
    @abstractmethod
    def get_responses(self, messages, n_samples=3, **kwargs) -> list[str]:
        raise NotImplementedError

class OpenAIClient(Client):
    def __init__(
            self, 
            model="meta-llama/Meta-Llama-3-8B-Instruct", 
        ):
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
        )

    def get_responses(self, messages, **kwargs) -> list[str]:
        responses = []
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        
        for choice in response.choices:
            responses.append(choice.message.content)
        return responses

from collections import Counter
def majority_vote(results: list[str]) -> str:
    return Counter(results).most_common(1)[0][0]

import json
def load_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


class DatasetLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
    
    def load(self, n_rows: int = None):
        dataset = load_jsonl(self.dataset_path)
        if n_rows is not None:
            dataset = dataset[:n_rows]
        self._dataset = dataset
        
        messages = [self.prepare_prompt(item) for item in dataset]
        self.messages = messages

    def prepare_prompt(self, item: dict) -> str:
        raise NotImplementedError

    def __getitem__(self, index: int):
        return self.messages[index]
    
    def __len__(self):
        return len(self.messages)
    
    def __iter__(self):
        for item in self.messages:
            yield item

class GSM8KDatasetLoader(DatasetLoader):
    def prepare_prompt(self, item: dict) -> str:
        question = item["question"]
        system_msg = "You are a helpful assistant."
        user_msg = f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        return messages


class ReasoningProgram:
    def __init__(self):
        self.knobs = {}
        self.state = dict(
            terminated=False,
        )
        self.certitude = {}
        
        self.input_message = None
        self.results = []
        self.result = None
        pass

    @property
    def is_terminated(self):
        return self.state['terminated']

    @abstractmethod
    def init_resource(self, app: Client):
        raise NotImplementedError

    @abstractmethod
    def update_certitude(self):
        raise NotImplementedError
    
    @abstractmethod
    def expand(self, app: Client):
        raise NotImplementedError
    
    @abstractmethod
    def aggregate(self, results: list[str]):
        raise NotImplementedError
    
    def run(self, app: Client):
        # TODO(GindaChen&Yichao) This is a default implementation. Different reasoning programs may have different run function, but let's assume it is simple and consistent in base class.
        self.init_resource(app)
        
        while not self.is_terminated:
            self.expand(app)
            self.update_certitude()
        
        return self.aggregate()
    
    @classmethod
    def run_batch(
        cls, app: Client, dataset: DatasetLoader, is_parallel=False
    ):
        programs = [cls(dataset[i]) for i in range(len(dataset))]

        if is_parallel:
            with ThreadPoolExecutor() as executor:
                futures = []
                for program in programs:
                    futures.append(executor.submit(program.run, app))
                results = [future.result() for future in futures]
        else:
            results = []
            for program in programs:
                results.append(program.run(app))

        return results



class SelfConsistentReasoningProgram(ReasoningProgram):
    def __init__(self, input_message: dict):
        super().__init__()
        self.sampling_params = dict(
            max_tokens=512,
            temperature=0.7,
        )
        self.input_message = input_message

    def init_resource(self, app: Client):
        self.knobs['n_branches'] = 5
        pass

    def expand(self, app: Client):
        knobs = self.knobs
        n_branches = knobs['n_branches']

        messages = self.input_message
        responses = app.get_responses(
            messages, n=n_branches,
            **self.sampling_params,
        )
        answers = [extract_answer(r) for r in responses]
        self.results.extend(answers)
        return

    def aggregate(self):
        results = self.results
        return majority_vote(results)
    
    def update_certitude(self):
        self.state['terminated'] = True
        return

    # def run(self, app: Client):
    #     # TODO(GindaChen&Yichao) This is a default implementation. Different reasoning programs may have different run function, but let's assume it is simple and consistent in base class.
    #     self.init_resource(app)
        
    #     while not self.is_terminated:
    #         self.expand(app)
    #         self.update_certitude()
        
    #     return self.aggregate()
    

def test_sc_gsm8k(
    is_parallel: bool = False,
    n_samples: int = 3,
    n_dataset_rows: int = 5,
):
    dataset = GSM8KDatasetLoader(dataset_path="data/GSM8K/test.jsonl")
    dataset.load(n_dataset_rows)

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:30000/v1"
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["OPENAI_API_BASE"] = openai_api_base
    
    client = OpenAIClient()
    results = SelfConsistentReasoningProgram.run_batch(
        client, dataset, is_parallel=is_parallel,
    )
    return results

if __name__ == "__main__":
    results = test_sc_gsm8k(is_parallel=True, n_dataset_rows=5)
    print(results)
