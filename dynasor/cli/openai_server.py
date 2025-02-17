import json
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

from dynasor.core.cot import effort_level
from dynasor.core.cot import openai_chat_completion_stream


# import vllm.entrypoints.openai.api_server


class DynasorOpenAIClient:
    # The Dynasor OpenAI Client is a wrapper that applys the chat template
    # and the reasoning stuff to the OpenAI API of vLLM.
    def __init__(
        self,
        model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        probe: str = "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{",
        dynasor_saving_effort: tuple = (2, 32),
    ):
        """
        Initialize the OpenAI Chat Client.

        Args:
            api_key: OpenAI API key.
            model: The model name to use (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B).
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.probe = probe
        self.dynasor_saving_effort = dynasor_saving_effort


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models for request validation
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


@app.get("/v1/models")
async def models():
    client = DynasorOpenAIClient()
    models = client.client.models.list()
    return models


# @app.post("/v1/chat/completions")
# async def chat_completions(request: ChatCompletionRequest):
#     """Handle chat completion requests."""
#     client = DynasorOpenAIClient()
#     print(f"request: {request}")
#     response = client.client.chat.completions.create(
#         model=request.model,
#         messages=request.messages,
#         temperature=request.temperature,
#         max_tokens=request.max_tokens,
#         stream=request.stream
#     )
#     return response

import re


def format_history(messages: List[ChatMessage]) -> str:
    """
    Convert chat conversation history into a prompt string.
    """
    formatted = ""
    for message in messages:
        role = message.role
        content = message.content
        if role == "system":
            formatted += "" + content + "\n"
        elif role == "user":
            formatted += "<｜User｜>" + content + "\n"
        else:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            formatted += "<｜Assistant｜>" + content + "\n"
    result = formatted + "<｜Assistant｜>"
    return result


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests."""
    client = DynasorOpenAIClient()
    openai_client = client.client
    messages = request.messages
    prompt = format_history(messages)
    stream = request.stream

    # Check for extra body parameters
    dynasor_saving_effort = client.dynasor_saving_effort
    if hasattr(request, "extra_body") and request.extra_body:
        if "dynasor" in request.extra_body:
            dynasor_config = request.extra_body["dynasor"]
            if "saving_effort" in dynasor_config:
                dynasor_saving_effort = effort_level(dynasor_config["saving_effort"])

    generator = openai_chat_completion_stream(
        client=openai_client,
        model=request.model,
        prompt=prompt,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        dynasor_saving_effort=dynasor_saving_effort,
        probeing_suffix=client.probe,
    )
    if not stream:
        result = ""
        for i in generator:
            result += i
            print(result)
        return JSONResponse(content={"choices": [{"message": {"content": result}}]})
    else:
        import time

        async def stream_response():
            request_id = f"chatcmpl-{int(time.time())}"
            created_time = int(time.time())

            # Send the role first
            first_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            print("yielding first chunk")
            yield f"data: {json.dumps(first_chunk)}\n\n"

            # Stream the content
            for content in generator:
                print("yielding content", content)
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Send the final [DONE] message
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")


def main():
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
