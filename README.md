# Dynasor


## Quick Start

Install vLLM
```bash
pip install vllm
```

Serve vLLM
```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 30000
```

Run examples for self-consistency and CoT
```bash
# Simple implementation
python examples/sc.py
python examples/cot.py

# Modular implementation
python examples/sc_mod.py
```
