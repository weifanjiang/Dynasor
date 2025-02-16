# Dynasor


## Quick Start

Install vLLM
```bash
pip install vllm
```

Serve Dynasor via vLLM
```bash
python -m dynasor.server.vllm_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 
```

Run examples for self-consistency and CoT
```bash
python examples/cot_client.py
```
