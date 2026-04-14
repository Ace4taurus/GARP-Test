import requests
import os
import json
import time
from dotenv import load_dotenv

# --- 实验核心参数配置区 ---
# 这里可以直接写商家支持的 Gemini 模型名，比如 "gemini-1.5-pro" 或 "gemini-2.5-flash"
MODEL_NAME = "gemini-3-flash-preview"  
EXPERIMENT_TEMP = 0.7  
NUM_BUDGETS = 11
NUM_RUNS_PER_BUDGET = 50
PROMPT_TYPE = "swap"   # "baseline" 或 "swap"

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
os.makedirs("data", exist_ok=True)
output_file = os.path.join("data", output_filename)

all_data = []
if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
    with open(output_file, "r", encoding="utf-8") as f:
        all_data = json.load(f)

# 加载环境变量
load_dotenv()

# ================= 鉴权与 API 路由 =================
is_deepseek = "deepseek" in MODEL_NAME.lower()
is_gemini = "gemini" in MODEL_NAME.lower()

if is_deepseek:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_url = "https://api.deepseek.com/v1/chat/completions"
elif is_gemini:
    api_key = os.getenv("GEMINI_API_KEY") # 这里填你的 sk- 密钥
    # 填入商家给你的中转站地址（通常要在后面加上 /v1/chat/completions 才能对齐 OpenAI 格式）
    api_url = "https://poloapi.top/v1/chat/completions"
else:
    raise ValueError(f"不支持的模型名称: {MODEL_NAME}")

if not api_key:
    raise ValueError(f"未找到 {MODEL_NAME} 对应的 API KEY，请检查 .env 文件。")

# 统一的请求头 (OpenAI 格式)
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# 读取 Prompt 模板
with open("./prompts/system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read().strip()
prompt_template_path = os.path.join("./prompts", f"prompt_{PROMPT_TYPE}.txt")
with open(prompt_template_path, "r", encoding="utf-8") as f:
    user_input_template = f.read().strip()

# 2. 遍历实验矩阵，发送请求
for idx, budget in enumerate(budget_sets):
    print(f"\n正在进行第 {idx+1} 组实验 ({MODEL_NAME}): 预算 m={budget['m']}, rate_self={budget['rate_self']}, rate_other={budget['rate_other']}")
    
    current_user_input = user_input_template.format(
        m=budget["m"],
        rate_self=budget["rate_self"],
        rate_other=budget["rate_other"]
    )

    # 统一的 Payload (OpenAI 格式)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": current_user_input}
    ]
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": EXPERIMENT_TEMP,
        "response_format": {"type": "json_object"} # 绝大多数中转站都支持透传这个参数
    }

    for run_id in range(1, NUM_RUNS_PER_BUDGET + 1):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 统一的 requests 调用
                response = requests.post(api_url, headers=headers, json=payload, timeout=40)
                
                if response.status_code == 200:
                    data = response.json()
                    assistant_reply = data['choices'][0]['message']['content']
                    
                    output = {
                        "run_id": run_id, 
                        "budget_params": budget,
                        "model_used": MODEL_NAME, 
                        "response": assistant_reply
                    }
                    all_data.append(output)
                    print(f"  - 第 {run_id} 次采样完成")
                    break 
                    
                elif response.status_code == 429:
                    print(f"  ! 触发限流，等待 5 秒后重试...")
                    time.sleep(5)
                else:
                    print(f"  ! 请求失败，状态码: {response.status_code}")
                    print(f"  ! 错误信息: {response.text}")
                    break
                    
            except Exception as e:
                print(f"  ! 发生异常: {e}，准备重试 ({attempt+1}/{max_retries})...")
                time.sleep(5) 
        
        # 休眠保护
        time.sleep(0.5)

# 写入文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"\n 实验数据收集完成！已保存至: {output_file}")