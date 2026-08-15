# observability.py
from langfuse import get_client
from langfuse.langchain import CallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()

langfuse = get_client()

def get_langfuse_handler():
    return CallbackHandler()

def track_query(
    question: str,
    answer: str,
    tools_used: list,
    tokens: int,
    latency_ms: int,
    api_key: str
):
    with langfuse.start_as_current_observation(
        as_type="span",
        name="agent-query"
    ) as span:
        span.update(
            input={"question": question},
            output={"answer": answer},
            metadata={
                "tools_used": tools_used,
                "tokens": tokens,
                "latency_ms": latency_ms,
                "api_key_prefix": api_key[:8] + "..."
            }
        )

def track_tool_call(tool_name: str, input: str, output: str, latency_ms: int):
    with langfuse.start_as_current_observation(
        as_type="span",
        name=f"tool-{tool_name}"
    ) as span:
        span.update(
            input={"query": input},
            output={"result": output},
            metadata={"latency_ms": latency_ms}
        )

def shutdown():
    langfuse.flush()