import json
import os
from typing import Optional
from ag_ui.core import RunAgentInput
from agent_loop import agent_loop
from agui import stream_agui_events, to_openai_messages, to_sse_data
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv
from pathlib import Path
from utils.log_utils import get_latest_failure_experience

# Try to import tools, fallback to empty list if not available
try:
    from tools import MAIN_AGENT_TOOLS
except ImportError:
    MAIN_AGENT_TOOLS = []

app = FastAPI()
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
load_dotenv()


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {"question": "What is the weather in Beijing today?"}
        },
    )

    question: str
    chat_history: Optional[list] = None
    file_path: Optional[str] = None

    def to_messages(self) -> list:
        if self.chat_history:
            return self.chat_history + [{"role": "user", "content": self.question}]
        else:
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": self.question},
            ]


def get_latest_failure_summary(log_dir="logs") -> Optional[str]:
    """
    从最近的日志文件中提取失败经验总结
    用于在下一次重试时为 Agent 提供‘避坑指南’
    """
    try:
        log_path = Path(log_dir)
        if not log_path.exists():
            return None

        # 获取最新的 JSON 日志文件
        logs = sorted(log_path.glob("*.json"), key=os.path.getmtime, reverse=True)
        for log_file in logs:
            with open(log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 提取 AnswerGenerator 存入的失败总结
                if data.get("error") and "经验总结" in data["error"]:
                    return data["error"]
    except Exception:
        return None
    return None


class QueryResponse(BaseModel):
    answer: str


@app.post("/")
async def query(req: QueryRequest) -> QueryResponse:
    """
    Basic LLM API example.

    Invoke example:

    ```
    curl -X POST "http://localhost:8000/" \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the weather in Beijing today?"}'

    ```

    Response example:

    ```json
    {
        "answer": "Beijing has sunny weather today, with temperatures between 10°C and 20°C."
    }
    ```


    """

    result = ""

    # Return messages after the last tool call message as the final answer
    async for chunk in agent_loop(req.to_messages(), MAIN_AGENT_TOOLS):
        if chunk.type == "tool_call" or chunk.type == "tool_call_result":
            result = ""
        elif chunk.type == "text" and chunk.content:
            result += chunk.content
    try:
        potential_json = json.loads(result)
        if isinstance(potential_json, dict) and "answer" in potential_json:
            result = str(potential_json["answer"])
    except:
        pass

    clean_answer = result.strip().strip('"')
    return QueryResponse(answer=clean_answer)


@app.post("/stream")
async def stream(req: QueryRequest) -> StreamingResponse:
    """
    Streaming query example.
    Invoke example:

    ```shell
    curl -N -X POST "http://localhost:8000/stream" \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the weather in Beijing today?"}'

    ```

    Response example:

    ```text

    data: {"answer": "Beijing has "}

    data: {"answer": "sunny weather"}

    data: {"answer": " today, with"}

    data: {"answer": " temperatures"}

    data: {"answer": " between 10°C and 20°C."}


    ```

    """

    async def stream_response():
        async for chunk in agent_loop(req.to_messages(), MAIN_AGENT_TOOLS):
            if chunk.type == "text" and chunk.content:
                data = {
                    "answer": chunk.content,
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            elif chunk.type == "text" and chunk.content == "":
                yield ": keepalive\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
    )


@app.post("/ag-ui")
async def ag_ui(run_agent_input: RunAgentInput) -> StreamingResponse:
    """
    AG-UI Protocol endpoint for streaming LLM interactions.

    AG-UI Protocol: https://docs.ag-ui.com/introduction
    """

    messages = to_openai_messages(run_agent_input.messages)

    # 2. 获取历史失败经验总结
    history_exp = get_latest_failure_experience()

    if history_exp:
        # 统一变量名：将总结封装为引导 Prompt
        experience_prompt = (
            "\n\n### 历史失败复盘指引 (PREVIOUS ATTEMPT SUMMARY)\n"
            "你在之前的尝试中未能完成任务，以下是失败原因的复盘总结，请在本次尝试中务必避开这些错误路径：\n"
            f"```text\n{history_exp}\n```\n"
            "请根据上述反馈调整你的搜索和推理策略。"
        )

        # 3. 注入经验到 Context
        # 修正：将引用的变量名改为 experience_prompt
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += experience_prompt
        else:
            messages.insert(0, {"role": "system", "content": experience_prompt})

    # 4. 执行多模态输入处理 (若 metadata 中包含 file_path)
    file_path = run_agent_input.metadata.get("file_path") if hasattr(run_agent_input, "metadata") else None
    if file_path and messages and messages[-1]["role"] == "user":
        from io.input_handler import process_input
        enhanced_prompt, _ = process_input(messages[-1]["content"], file_path)
        messages[-1]["content"] = enhanced_prompt

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


if __name__ == "__main__":
    import uvicorn

    # 启动服务器，监听 8000 端口
    uvicorn.run(app, host="127.0.0.1", port=8000)
