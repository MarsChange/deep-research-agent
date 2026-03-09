import asyncio
import json
import re
from typing import AsyncIterator, Optional

from ag_ui.core import RunAgentInput
from agent_loop import agent_loop
from agui import stream_agui_events, to_openai_messages, to_sse_data
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

# Try to import tools, fallback to empty list if not available
try:
    from tools import MAIN_AGENT_TOOLS
except ImportError:
    MAIN_AGENT_TOOLS = []

app = FastAPI()

# Sub-agent progress message prefixes (should not appear in final answer)
_PROGRESS_PREFIXES = ("🔍", "⚙️", "✅", "📊", "💬", "⏳", "⚠️")


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {"question": "What is the weather in Beijing today?"}
        },
    )

    question: str
    chat_history: Optional[list] = None

    def to_messages(self) -> list:
        if self.chat_history:
            return self.chat_history + [{"role": "user", "content": self.question}]
        else:
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": self.question},
            ]


class QueryResponse(BaseModel):
    answer: str


# ---------------------------------------------------------------------------
# SSE heartbeat interleaving (from LangStudio llm-basic template)
# ---------------------------------------------------------------------------

async def stream_with_heartbeat(
    messages: AsyncIterator[str], ping_interval: int = 5
) -> AsyncIterator[str]:
    """Interleave an SSE message stream with periodic Ping heartbeats."""

    async def ping(interval: int) -> AsyncIterator[str]:
        while True:
            yield "event: Ping\n\n"
            await asyncio.sleep(interval)

    ping_gen = ping(ping_interval)
    message_task = asyncio.create_task(anext(messages), name="message")
    ping_task = asyncio.create_task(anext(ping_gen), name="ping")

    try:
        while True:
            done, _ = await asyncio.wait(
                {message_task, ping_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if ping_task in done:
                yield ping_task.result()
                ping_task = asyncio.create_task(anext(ping_gen), name="ping")

            if message_task in done:
                try:
                    yield message_task.result()
                except StopAsyncIteration:
                    break
                message_task = asyncio.create_task(anext(messages), name="message")
    finally:
        for task in (message_task, ping_task):
            task.cancel()
        await asyncio.gather(message_task, ping_task, return_exceptions=True)

        for gen in (messages, ping_gen):
            aclose = getattr(gen, "aclose", None)
            if callable(aclose):
                await aclose()


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _extract_answer(raw: str) -> str:
    """Extract the answer value from LLM output.

    The LLM is instructed to output {"answer": "..."} JSON.
    This function extracts the value; falls back to the raw text.
    """
    raw = raw.strip()
    # Try parsing the whole text as JSON
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "answer" in parsed:
            return str(parsed["answer"])
    except (json.JSONDecodeError, ValueError):
        pass
    # Try regex extraction for {"answer": "..."}
    match = re.search(r'\{\s*"answer"\s*:\s*"(.*?)"\s*\}', raw, re.DOTALL)
    if match:
        return match.group(1)
    return raw


# ---------------------------------------------------------------------------
# Message generator — buffers text, emits final answer as Message event
# ---------------------------------------------------------------------------

async def _query_message_chunks(req: QueryRequest) -> AsyncIterator[str]:
    """Consume agent_loop, yield SSE Message event(s) for the final answer only.

    During the research phase this generator blocks (yielding nothing), so
    ``stream_with_heartbeat`` keeps sending Ping events. Once the answer is
    ready it is yielded as ``event: Message``.
    """
    answer_buffer = ""

    async for chunk in agent_loop(req.to_messages(), MAIN_AGENT_TOOLS):
        if chunk.type in ("tool_call", "tool_call_result"):
            # Reset — text before tool calls is intermediate reasoning
            answer_buffer = ""
        elif chunk.type == "text" and chunk.content:
            # Skip sub-agent progress messages
            if not chunk.content.lstrip().startswith(_PROGRESS_PREFIXES):
                answer_buffer += chunk.content

    # Emit the final answer
    answer = _extract_answer(answer_buffer)
    data = json.dumps({"answer": answer}, ensure_ascii=False)
    yield f"event: Message\ndata: {data}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request):
    """
    Supports both JSON and streaming SSE responses.

    When the request includes ``Accept: text/event-stream``, the response is
    streamed as Server-Sent Events with periodic Ping heartbeats.
    Otherwise a plain JSON ``{"answer": "..."}`` is returned.
    """

    accept = request.headers.get("accept", "")
    if "text/event-stream" in accept:
        return StreamingResponse(
            stream_with_heartbeat(_query_message_chunks(req)),
            media_type="text/event-stream",
        )
    else:
        result = ""

        async for chunk in agent_loop(req.to_messages(), MAIN_AGENT_TOOLS):
            if chunk.type == "tool_call" or chunk.type == "tool_call_result":
                result = ""
            elif chunk.type == "text" and chunk.content:
                if not chunk.content.lstrip().startswith(_PROGRESS_PREFIXES):
                    result += chunk.content

        return QueryResponse(answer=_extract_answer(result))


@app.post("/stream")
async def stream(req: QueryRequest) -> StreamingResponse:
    """
    Streaming SSE endpoint (always returns SSE regardless of Accept header).
    """
    return StreamingResponse(
        stream_with_heartbeat(_query_message_chunks(req)),
        media_type="text/event-stream",
    )


@app.post("/ag-ui")
async def ag_ui(run_agent_input: RunAgentInput) -> StreamingResponse:
    """
    AG-UI Protocol endpoint for streaming LLM interactions.

    AG-UI Protocol: https://docs.ag-ui.com/introduction
    """

    messages = to_openai_messages(run_agent_input.messages)

    async def stream_response():
        async for event in stream_agui_events(
            chunks=agent_loop(messages, MAIN_AGENT_TOOLS), run_agent_input=run_agent_input
        ):
            if isinstance(event, str):
                yield event
            else:
                yield to_sse_data(event)

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
    )
