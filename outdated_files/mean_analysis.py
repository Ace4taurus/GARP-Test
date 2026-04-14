import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 1. 基础配置与准备 =================
# 请确保这是你包含1100条数据的JSON文件路径
INPUT_DATA_FILE = "data/gemini_3.1_pro_preview_swap_temp0.7_11budgets_50runs_20260315_1416.json"

# ================= 2. 理论模型与算法 =================
def theoretical_demand_pi_s(X, alpha, rho):
    m, p_s, p_o = X
    alpha = np.clip(alpha, 1e-5, 1 - 1e-5)
    term1 = (1 - alpha) / alpha
    term2 = p_s / p_o
    exponent = 1 / (1 - rho)
    denominator = p_s + p_o * (term1 * term2) ** exponent
    return m / denominator

def check_garp_with_efficiency(prices, bundles, e):
    """带 Afriat 效率指数 e 的 GARP 检验"""
    n = len(prices)
    if n == 0: return False
    direct_pref = np.zeros((n, n), dtype=bool)
    strict_direct_pref = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        expenditure_i = np.dot(prices[i], bundles[i])
        for j in range(n):
            cost_of_j_at_p_i = np.dot(prices[i], bundles[j])
            if cost_of_j_at_p_i <= e * expenditure_i + 1e-6:
                direct_pref[i, j] = True
            if cost_of_j_at_p_i < e * expenditure_i - 1e-6:
                strict_direct_pref[i, j] = True
                
    indirect_pref = direct_pref.copy()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                indirect_pref[i, j] = indirect_pref[i, j] or (indirect_pref[i, k] and indirect_pref[k, j])
                
    for i in range(n):
        for j in range(n):
            if indirect_pref[i, j] and strict_direct_pref[j, i]:
                return False
    return True

def calculate_ccei(prices, bundles, tol=1e-3):
    """二分查找计算最大可能的效率指数 CCEI"""
    if check_garp_with_efficiency(prices, bundles, 1.0):
        return 1.0
        
    low, high = 0.0, 1.0
    best_e = 0.0
    while high - low > tol:
        mid = (low + high) / 2.0
        if check_garp_with_efficiency(prices, bundles, mid):
            best_e = mid
            low = mid
        else:
            high = mid
    return best_e

# ================= 3. 数据处理与分析 =================
def generate_analysis():
    print(f"开始分析数据文件: {INPUT_DATA_FILE}")
    with open(INPUT_DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    all_trials = []
    
    for item in data:
        budget = item["budget_params"]
        try:
            decision = json.loads(item["response"])
            kept = decision.get("tokens_kept", 0)
            transferred = decision.get("tokens_transferred", 0)
        except: continue
        if kept + transferred != budget["m"]: continue
            
        all_trials.append({
            "m": budget["m"], "p_s": 1.0/budget["rate_self"], "p_o": 1.0/budget["rate_other"], 
            "pi_s": kept*budget["rate_self"], "pi_o": transferred*budget["rate_other"]
        })

    print(f"收集到的有效trials数: {len(all_trials)}")

    if len(all_trials) < 9:
        print("数据点不足，无法进行分析。")
        return

    # 准备整体数据
    X_data = np.array([[t["m"] for t in all_trials], [t["p_s"] for t in all_trials], [t["p_o"] for t in all_trials]])
    y_data = np.array([t["pi_s"] for t in all_trials])
    
    # 拟合参数
    initial_guess, param_bounds = [0.5, 0.5], ([0.001, -10.0], [0.999, 0.99])
    
    alpha, rho, r_squared = np.nan, np.nan, np.nan
    try:
        popt, _ = curve_fit(theoretical_demand_pi_s, X_data, y_data, p0=initial_guess, bounds=param_bounds, maxfev=10000)
        y_pred = theoretical_demand_pi_s(X_data, popt[0], popt[1])
        sst = np.sum((y_data - np.mean(y_data)) ** 2)
        ssr = np.sum((y_data - y_pred) ** 2)
        
        alpha = popt[0]
        rho = popt[1]
        r_squared = 1 - (ssr/sst) if sst!=0 else 1.0
    except:
        pass
    
    # 输出到终端
    print("====== 大语言模型行为偏好实验综合报告 ======")
    print(f"分析数据源: {INPUT_DATA_FILE}")
    print(f"总数据点数: {len(all_trials)}")
    print()
    print("--- CES 效用函数参数估计 ---")
    if not np.isnan(alpha):
        print(f"Alpha (自利权重): {alpha:.4f}")
        print(f"Rho (替代弹性): {rho:.4f}")
        print(f"R^2 (拟合优度): {r_squared:.4f}")
    else:
        print("拟合失败。")
    print("========================================")

if __name__ == "__main__":
    generate_analysis()
