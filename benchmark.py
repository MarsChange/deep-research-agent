import json
import os
import asyncio
import httpx
import time
from datetime import datetime

# 配置参数
API_URL = "http://127.0.0.1:8000/"
INPUT_FILE = "question2.jsonl"
OUTPUT_FILE = "answers2_1.jsonl"
CONCURRENCY_LIMIT = 5  # 限制同时进行的测试任务数，避免压垮后端 LLM 额度


async def process_question(client, semaphore, question_data):
    """处理单个问题的异步函数"""
    qid = question_data.get("id")
    question_text = question_data.get("Question")

    async with semaphore:
        print(f"[*] [{datetime.now().strftime('%H:%M:%S')}] 正在启动任务 ID: {qid}")
        start_time = time.time()

        try:
            # 调用你的 agent.py 接口
            response = await client.post(
                API_URL,
                json={"question": question_text},
                timeout=600.0
            )
            response.raise_for_status()
            result_json = response.json()
            answer = result_json.get("answer", "")
        except Exception as e:
            print(f"[!] ID {qid} 出错: {e}")
            answer = f"Error: {str(e)}"

        duration = time.time() - start_time
        print(f"[+] ID {qid} 完成 | 耗时: {duration:.2f}s | 结果: {answer}")

        return {
            "id": qid,
            "answer": answer,
            "time_taken": round(duration, 2)
        }


async def main():
    # 限制并发，防止压垮后端或触发 API 频率限制
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    # 加载问题
    questions = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))

    print(f"开始测试，总计 {len(questions)} 个问题，最大并发: {CONCURRENCY_LIMIT}")


    mounts = {
        "http://127.0.0.1": httpx.AsyncHTTPTransport(),
        "http://localhost": httpx.AsyncHTTPTransport(),
    }

    # 使用 mounts 初始化异步客户端
    async with httpx.AsyncClient(mounts=mounts) as client:
        tasks = [process_question(client, semaphore, q) for q in questions]
        results = await asyncio.gather(*tasks)

    # 写入结果文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"\n[!] 所有任务已完成。结果保存在: {OUTPUT_FILE}")

if __name__ == "__main__":
    # 建议在脚本启动前强制禁用本地流量的代理环境变量
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    asyncio.run(main())