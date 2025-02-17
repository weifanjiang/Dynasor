
# Dynasor

Simple extension on vLLM to help you speed up reasoning model without training.

<!-- https://hao-ai-lab.github.io/blogs/dynasor-cot/ -->
<!-- <div align="center" style="line-height: 1;">
    <a href="https://viol2000.github.io/SubDomain/gradio-2.html" target="_blank" style="margin: 2px;">
        <img alt="Demo" src="https://img.shields.io/badge/🤖Chat-Deepseek-blue?" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="https://x.com/haoailab" target="_blank" style="margin: 2px;">
        <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-haoailab-white?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="https://hao-ai-lab.github.io/blogs/dynasor-cot/" target="_blank" style="margin: 2px;">
        <img alt="Blog" src="https://img.shields.io/badge/Blog-Dynasor-4CAF50?&color=4CAF50" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="https://arxiv.org/abs/2412.20993" style="margin: 2px;">
        <img alt="License" src="https://img.shields.io/badge/Paper-Dynasor-4CAF50?&color=4CAF50" style="display: inline-block; vertical-align: middle;"/>
    </a>
</div> -->

<div align="center" style="line-height: 1;">
    | <a href="https://hao-ai-lab.github.io/blogs/dynasor-cot/">📝 Blog</a> 
    | <a href="https://arxiv.org/abs/2412.20993">📄 Paper</a> 
    | <a href="https://viol2000.github.io/SubDomain/gradio-2.html">🤖 Demo</a> 
    | <a href="https://x.com/haoailab">🐦 Twitter/X</a> 
    |
</div>




## Quick Start 

```bash
# Install vllm and dynasor
pip install vllm dynasor

# Setup a vLLM server
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -tp 1 --enable-prefix-caching

# Start Dynasor Chat
dynasor-chat
```


### How it works

Dynasor uses a combination of techniques to speed up LLM reasoning:

1. **Prompt Engineering**: We use a combination of techniques to improve the prompt.
2. **Dynamic Execution**: We dynamically execute the prompt, and stop when the LLM has enough information to make a decision.


## How to use Dynasor

We provide 3 tools to launch Dynasor:

1. [`dynasor-chat`](#dynasor-chat-cli-chat-interface): CLI chat interface to interact with Dynasor
2. [`dynasor-openai`](#dynasor-openai-openai-compatible-server): OpenAI compatible server.
3. [`dynasor-vllm`](#dynasor-vllm-vllm-native-server): vLLM-native server


### `dynasor-chat`: CLI Chat Interface

> [!WARNING]
> We recommend enabling prefix caching, otherwise probing will be very slow.

1. Setup a vLLM server
```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -tp 1 --enable-prefix-caching
```


2. Open Dynasor Chat in command line

```bash
dynasor-chat
```

Then you should be entering the Dynasor Chat interface.
```bash
> : The point $(a, b)$ lies on the line with the equation $3x + 2y = 12.$ When $a = 4$, what is the value of $b$?
# ... wait for the answer ...
**Final Answer:**
[ \boxed{0} \]
```


### `dynasor-openai` OpenAI Compatible Server


1. Setup a vLLM server
```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -tp 1 --enable-prefix-caching
```

2. Setup OpenAI compatible proxy server to server Dynasor
```bash
dynasor-openai
```

3. Use our simiple client script to query:
```bash
# Sample Dynasor client script to ask some questions
python examples/client.py --prompt "2+2=?"
python examples/client.py --prompt "Solve x^2 + 4x = 4"
python examples/client.py --prompt "How many nonzero points are there on x^3y + y^3z + z^3x = 0 over the finite field  𝔽_{{5}^{18}}  up to scaling?"
```


### `dynasor-vllm`: vLLM-native Server

We build Dynasor on top of vLLM as a part of the vLLM OpenAI compatible server endpoint.

1. Setup a dynasor-vllm server
```bash
dynasor-vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -tp 1 --enable-prefix-caching
```

2. Use our simple client script to query:
```bash
python examples/client-vllm.py
```


## Benchmark

### Token Deprivation Experiment

Run the following command to perform token deprivation experiment on math500 dataset.
```bash
bash benchmark/TokenDeprivation/run.sh
```

## Citation

If you use Dynasor for your research, please cite our [paper](https://arxiv.org/abs/2412.20993):

```bibtex
@article{fu2024efficiently,
  title={Efficiently Serving LLM Reasoning Programs with Certaindex},
  author={Fu, Yichao and Chen, Junda and Zhu, Siqi and Fu, Zheyu and Dai, Zhongdongming and Qiao, Aurick and Zhang, Hao},
  journal={arXiv preprint arXiv:2412.20993},
  year={2024}
}
```