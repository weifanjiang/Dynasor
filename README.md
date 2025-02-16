# Dynasor


## Quick Start

Install vLLM
```bash
pip install vllm
```

Install Dynasor
```bash
git clone https://github.com/hao-ai-lab/Dynasor.git
cd Dynasor && pip install Dynasor && cd -
```

Serve Dynasor via vLLM
```bash
python -m dynasor.server.vllm_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 
```

Run examples for self-consistency and CoT
```bash
python examples/cot_client.py
```
