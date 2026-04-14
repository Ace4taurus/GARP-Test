import asyncio
import aiohttp
import os
import json
import time
from dotenv import load_dotenv
from typing import List, Dict, Any

# --- 实验核心参数配置区 ---
MODEL_NAME = "gemini-3.1-pro-preview"  
EXPERIMENT_TEMP = 0.7  
NUM_BUDGETS = 11
NUM_RUNS_PER_BUDGET = 50
PROMPT_TYPE = "baseline"   # "baseline" 或 "swap"
TYPE = "abstract" # "agentic" or "abstract"
MAX_CONCURRENT_REQUESTS = 100  # 并发请求数量限制

# 1. 定义实验矩阵 (Andreoni & Miller 预算集)
budget_sets = [
    {"m": 40, "rate_self": 3, "rate_other": 1},
    {"m": 40, "rate_self": 1, "rate_other": 3},
    {"m": 60, "rate_self": 2, "rate_other": 1},
    {"m": 60, "rate_self": 1, "rate_other": 2},
    {"m": 75, "rate_self": 1, "rate_other": 1},
    {"m": 40, "rate_self": 4, "rate_other": 1},
    {"m": 40, "rate_self": 1, "rate_other": 4},
    {"m": 60, "rate_self": 3, "rate_other": 1},
    {"m": 60, "rate_self": 1, "rate_other": 3},
    {"m": 100, "rate_self": 1, "rate_other": 1},
    {"m": 80, "rate_self": 2, "rate_other": 2}
]

# 动态生成输出文件名
timestamp = time.strftime("%Y%m%d_%H%M")
safe_model_name = MODEL_NAME.replace("-", "_")
output_filename = f"{safe_model_name}_{PROMPT_TYPE}_temp{EXPERIMENT_TEMP}_{NUM_BUDGETS}budgets_{NUM_RUNS_PER_BUDGET}runs_{timestamp}.json"

# 根据TYPE变量动态设置目录
data_dir = os.path.join("data", TYPE)
analysis_dir = os.path.join("analysis_results", TYPE)
prompts_dir = os.path.join("prompts", TYPE)

os.makedirs(data_dir, exist_ok=True)
os.makedirs(analysis_dir, exist_ok=True)
os.makedirs(prompts_dir, exist_ok=True)

output_file = os.path.join(data_dir, output_filename)

# 加载环境变量
load_dotenv()

# ================= 鉴权与 API 路由 =================
is_deepseek = "deepseek" in MODEL_NAME.lower()
is_gemini = "gemini" in MODEL_NAME.lower()

if is_deepseek:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_url = "https://api.deepseek.com/v1/chat/completions"
elif is_gemini:
    api_key = os.getenv("GEMINI_API_KEY")
    api_url = "https://poloapi.top/v1/chat/completions"
else:
    raise ValueError(f"不支持的模型名称: {MODEL_NAME}")

if not api_key:
    raise ValueError(f"未找到 {MODEL_NAME} 对应的 API KEY，请检查 .env 文件。")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# 读取 Prompt 模板
system_prompt_path = os.path.join(prompts_dir, "system_prompt.txt")
with open(system_prompt_path, "r", encoding="utf-8") as f:
    system_prompt = f.read().strip()
prompt_template_path = os.path.join(prompts_dir, f"prompt_{PROMPT_TYPE}.txt")
with open(prompt_template_path, "r", encoding="utf-8") as f:
    user_input_template = f.read().strip()

# 结果锁和共享列表
results_lock = asyncio.Lock()
all_results = []


async def make_request(session: aiohttp.ClientSession, 
                      budget: Dict[str, int], 
                      budget_idx: int, 
                      run_id: int,
                      semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """发送单个请求"""
    async with semaphore:
        current_user_input = user_input_template.format(
            m=budget["m"],
            rate_self=budget["rate_self"],
            rate_other=budget["rate_other"]
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": current_user_input}
        ]
        
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": EXPERIMENT_TEMP,
            "response_format": {"type": "json_object"}
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with session.post(
                    url=api_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=40)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        assistant_reply = data['choices'][0]['message']['content']
                        
                        result = {
                            "run_id": run_id,
                            "budget_params": budget,
                            "model_used": MODEL_NAME,
                            "response": assistant_reply,
                            "status": "success"
                        }
                        print(f"✓ 预算集 {budget_idx+1}/11 - 第 {run_id} 次采样完成")
                        return result
                    
                    elif response.status == 429:
                        print(f"⚠ 预算集 {budget_idx+1}/11 - 第 {run_id} 次采样: 触发限流，重试中...")
                        await asyncio.sleep(5)
                    else:
                        error_text = await response.text()
                        print(f"✗ 预算集 {budget_idx+1}/11 - 第 {run_id} 次采样: 请求失败 (状态码: {response.status})")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                        
            except asyncio.TimeoutError:
                print(f"✗ 预算集 {budget_idx+1}/11 - 第 {run_id} 次采样: 请求超时")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
            except Exception as e:
                print(f"✗ 预算集 {budget_idx+1}/11 - 第 {run_id} 次采样: 异常 {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)

        return {
            "run_id": run_id,
            "budget_params": budget,
            "model_used": MODEL_NAME,
            "response": None,
            "status": "failed"
        }


async def collect_data():
    """异步数据收集主函数"""
    # 创建信号量来限制并发数
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # 生成所有请求任务
    tasks = []
    for budget_idx, budget in enumerate(budget_sets):
        for run_id in range(1, NUM_RUNS_PER_BUDGET + 1):
            tasks.append((budget_idx, budget, run_id))
    
    print(f"开始批量请求: 总共 {len(tasks)} 个请求，并发限制: {MAX_CONCURRENT_REQUESTS}")
    print(f"预计加速倍率: 约 {len(tasks) / (len(budget_sets) * NUM_RUNS_PER_BUDGET / MAX_CONCURRENT_REQUESTS):.1f}x\n")
    
    start_time = time.time()
    
    # 使用 ClientSession 进行请求
    async with aiohttp.ClientSession() as session:
        coroutines = [
            make_request(session, budget, budget_idx, run_id, semaphore)
            for budget_idx, budget, run_id in tasks
        ]
        
        # 使用 gather 并发执行所有任务
        results = await asyncio.gather(*coroutines, return_exceptions=True)
    
    # 过滤有效结果
    valid_results = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ 数据收集完成！")
    print(f"  - 成功: {len(valid_results)} 个请求")
    print(f"  - 失败: {len(tasks) - len(valid_results)} 个请求")
    print(f"  - 总耗时: {elapsed:.2f} 秒")
    print(f"  - 平均速度: {len(tasks) / elapsed:.2f} req/s")
    
    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(valid_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n已保存至: {output_file}")
    return valid_results


if __name__ == "__main__":
    asyncio.run(collect_data())
