
# Dynasor

Simple extension on vLLM to help you speed up LLM reasoning (e.g., Long Chain of Thought).

## Installation

```bash
pip install dynasor
```


### How it works

Dynasor uses a combination of techniques to speed up LLM reasoning:

1. **Prompt Engineering**: We use a combination of techniques to improve the prompt.
2. **Dynamic Execution**: We dynamically execute the prompt, and stop when the LLM has enough information to make a decision.


# Launch

## `dynasor-chat`: CLI Chat Interface

> [!WARNING]
> We recommend enabling prefix caching, otherwise probing will be very slow.

1. Setup a vLLM server
```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -tp 1 \
    --enable-prefix-caching
```


2. Open Dynasor Chat in command line

```bash
dynasor-chat
```

Then you should be entering the Dynasor Chat interface.
```bash
Welcome to Dynasor Chat!
• Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
• Endpoint URL: http://localhost:8000/v1
• Dynasor Saving Effort: mid
Type 'help' to show help, and 'exit' to end the conversation.
Type anything to start chatting.

> : The point $(a, b)$ lies on the line with the equation $3x + 2y = 12.$ When $a = 4$, what is the value of $b$?

# ... wait for the answer ...
**Final Answer:**
[
 \boxed{0}
\]
```


> [!NOTE]  
> You can also disable Dynasor by running
> ```bash
> # Disable Dynasor
> dynasor-chat --dynasor-saving-effort none
> ```


## `dynasor-openai` OpenAI Compatible Server

> [!NOTE]  
> We only support the `/chat/completions` endpoint for now. We will add the `/completions` endpoint in the future.

1. Setup a vLLM server
```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -tp 1 \
    --enable-prefix-caching
```

2. Setup OpenAI compatible proxy server to server Dynasor
```bash
dynasor-openai
```

3. Use our simiple client script to query:
```bash
# Dynasor client
python examples/client.py --prompt "2+2=?"
python examples/client.py --prompt "Solve x^2 + 4x = 4"
python examples/client.py --prompt "How many nonzero points are there on x^3y + y^3z + z^3x = 0 over the finite field  𝔽_{{5}^{18}}  up to scaling?"
```


## `dynasor-vllm`: vLLM-native Server

We integrate Dynasor into vLLM as a part of the vLLM OpenAI compatible server endpoint.

> [!NOTE]  
> We only support the `/completion` endpoint for now. We will add the `/chat/completion` endpoint in the future.


1. Setup a dynasor-vllm server
```bash
dynasor-vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -tp 1 \
    --enable-prefix-caching
```

2. Use our simple client script to query:
```bash
python examples/client-vllm.py
```


# Benchmark

## Token Deprivation Experiment

Run the following command to perform token deprivation experiment on math500 dataset.
```bash
mkdir -p benchmark-output

python benchmark/TokenDeprivation/run.py \
--model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
--dataset math500 \
--step 32 --max-tokens 16384 --start 0 --end 10 \
--output benchmark-output/math500_step32_max16384_trials10  --probe-tokens 32 --probe "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{" 
```

