# Local Deployment

This tutorial shows how to deploy Dynasor locally with ollama and deepseek-r1.

# Dynasor with ollama + deepseek-r1

## Install ollama

Install [ollama](https://ollama.com/download) and pull a [deepseek-r1](https://ollama.com/library/deepseek-r1) model.
```bash
ollama run deepseek-r1
```

## Install Dynasor

```bash
git clone https://github.com/hao-ai-lab/Dynasor.git
cd Dynasor
pip install .
```

## Run `dynasor-chat`

We provide a command line tool `dynasor-chat` to interact with ollama via Dynasor.

```bash
dynasor-chat --base-url http://localhost:11434/v1 --model deepseek-r1
```

## Run `dynasor-openai`

We also provide a proxy server `dynasor-openai` for Dynasor that is compatible with any OpenAI API.

```bash
dynasor-openai --base-url http://localhost:11434/v1 --model deepseek-r1 --port 28080
```

Then run `dynasor-chat` with the base URL `http://localhost:28080/v1`.

```bash
dynasor-chat --base-url http://localhost:28080/v1
```

or use our simple example:
