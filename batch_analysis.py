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
INPUT_DATA_FILE = "data/abstract/gemini_3.1_pro_preview_baseline_temp0.7_11budgets_50runs_20260407_1707.json" 

# 自动识别数据类型（agentic 或 abstract）
def detect_type_from_path(file_path):
    """从文件路径中识别数据类型"""
    if "agentic" in file_path.lower():
        return "agentic"
    elif "abstract" in file_path.lower():
        return "abstract"
    else:
        raise ValueError(f"无法识别文件路径中的类型 (agentic/abstract): {file_path}")

data_type = detect_type_from_path(INPUT_DATA_FILE)
OUTPUT_DIR = f"analysis_results/{data_type}"

# 创建分析结果文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 从输入文件名提取基本信息
input_filename = os.path.basename(INPUT_DATA_FILE).replace(".json", "")

# 定义导出路径
CSV_PATH = os.path.join(OUTPUT_DIR, f"synthetic_subjects_results_{input_filename}.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, f"analysis_summary_report_{input_filename}.txt")
PLOT_PATH_ALPHARHO = os.path.join(OUTPUT_DIR, f"alpha_rho_distribution_{input_filename}.png")
PLOT_PATH_CCEI = os.path.join(OUTPUT_DIR, f"ccei_distribution_{input_filename}.png") # 新增 CCEI 图表路径

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
def generate_analysis_and_plots():
    print(f"开始分析数据文件: {INPUT_DATA_FILE}")
    print(f"检测到数据类型: {data_type.upper()}")
    with open(INPUT_DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    subjects_data = defaultdict(list)
    
    for item in data:
        run_id = item.get("run_id")
        if run_id is None: continue
        budget = item["budget_params"]
        try:
            decision = json.loads(item["response"])
            kept = decision.get("tokens_kept", 0)
            transferred = decision.get("tokens_transferred", 0)
        except: continue
        if kept + transferred != budget["m"]: continue
            
        subjects_data[run_id].append({
            "m": budget["m"], "p_s": 1.0/budget["rate_self"], "p_o": 1.0/budget["rate_other"], 
            "pi_s": kept*budget["rate_self"], "pi_o": transferred*budget["rate_other"]
        })

    results_list = []
    initial_guess, param_bounds = [0.5, 0.5], ([0.001, -10.0], [0.999, 0.99])

    print("正在进行批量 CCEI 计算与 NLS 拟合...")
    for run_id, trials in subjects_data.items():
        if len(trials) < 9: continue
        prices = np.array([[t["p_s"], t["p_o"]] for t in trials])
        bundles = np.array([[t["pi_s"], t["pi_o"]] for t in trials])
        
        # 首先计算该被试的 CCEI
        ccei_val = calculate_ccei(prices, bundles)
        passed_garp = (ccei_val == 1.0)
        
        # 准备数据结构
        res = {
            "run_id": run_id, 
            "CCEI": ccei_val, 
            "Passed_GARP": passed_garp,
            "Alpha": np.nan, 
            "Rho": np.nan, 
            "R_squared": np.nan
        }
        
        # 如果完全理性，执行 NLS 拟合
        if passed_garp:
            X_data = np.array([[t["m"] for t in trials], [t["p_s"] for t in trials], [t["p_o"] for t in trials]])
            y_data = np.array([t["pi_s"] for t in trials])
            try:
                popt, _ = curve_fit(theoretical_demand_pi_s, X_data, y_data, p0=initial_guess, bounds=param_bounds, maxfev=10000)
                y_pred = theoretical_demand_pi_s(X_data, popt[0], popt[1])
                sst = np.sum((y_data - np.mean(y_data)) ** 2)
                ssr = np.sum((y_data - y_pred) ** 2)
                
                res["Alpha"] = popt[0]
                res["Rho"] = popt[1]
                res["R_squared"] = 1 - (ssr/sst) if sst!=0 else 1.0
            except: pass
            
        results_list.append(res)

    # ================= 4. 输出文本报告 =================
    df = pd.DataFrame(results_list)
    total_valid = len(df)
    
    if total_valid == 0:
        print("未发现完整（凑齐11组决策）的有效被试数据，分析终止。")
        return
        
    passed_garp_count = df["Passed_GARP"].sum()
    pass_rate = (passed_garp_count / total_valid) * 100
    
    # CCEI 统计
    mean_ccei = df["CCEI"].mean()
    median_ccei = df["CCEI"].median()
    rational_enough = (df["CCEI"] >= 0.95).sum()
    rational_rate = (rational_enough / total_valid) * 100
    
    # 提取通过 GARP 的个体进行 Alpha/Rho 统计
    df_passed = df[df["Passed_GARP"] == True]
    
    report_content = (
        "====== 大语言模型行为偏好实验综合报告 ======\n"
        f"分析数据源: {INPUT_DATA_FILE}\n"
        f"有效被试总数 (11次决策完整): {total_valid}\n\n"
        "--- GARP 与 CCEI 理性检验 ---\n"
        f"完全理性 (通过 GARP): {passed_garp_count} 个 (占 {pass_rate:.2f}%)\n"
        f"CCEI 平均效率指数: {mean_ccei:.4f} (中位数: {median_ccei:.4f})\n"
        f"接近理性 (CCEI >= 0.95): {rational_enough} 个 (占 {rational_rate:.2f}%)\n\n"
        "--- CES 效用函数参数估计 (仅针对完全理性样本) ---\n"
    )
    
    if not df_passed.empty:
        report_content += (
            f"Alpha (自利权重) 平均值: {df_passed['Alpha'].mean():.4f} (标准差: {df_passed['Alpha'].std():.4f})\n"
            f"Rho   (替代弹性) 平均值: {df_passed['Rho'].mean():.4f} (标准差: {df_passed['Rho'].std():.4f})\n"
            f"R^2   (拟合优度) 平均值: {df_passed['R_squared'].mean():.4f}\n"
        )
    else:
        report_content += "没有通过 GARP 的数据可供拟合。\n"
        
    report_content += "========================================\n"
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
    print("\n" + report_content)
    
    # 保存大一统 CSV
    df.to_csv(CSV_PATH, index=False)

    # ================= 5. 学术级数据可视化 =================
    sns.set_theme(style="whitegrid")
    
    # 图表 1: Alpha-Rho 偏好分布图 (仅针对通过的个体)
    if not df_passed.empty and len(df_passed) > 1:
        plt.figure(figsize=(8, 6))
        g = sns.jointplot(
            data=df_passed, x="Alpha", y="Rho", 
            kind="scatter", alpha=0.7, color="#1f77b4", s=80,
            marginal_kws=dict(bins=15, fill=True, color="#1f77b4")
        )
        g.fig.suptitle("Distribution of Economic Preferences (Strict GARP Passed)", y=1.02, fontsize=14, fontweight='bold')
        g.set_axis_labels("Selfishness Weight (Alpha)", "Substitution Elasticity (Rho)", fontsize=12)
        g.ax_joint.set_xlim(0, 1)
        g.ax_joint.set_ylim(0, 1.0)
        plt.savefig(PLOT_PATH_ALPHARHO, dpi=300, bbox_inches='tight')
        plt.close('all') # 释放内存
        
    # 图表 2: CCEI 分布直方图 (针对所有有效个体)
    plt.figure(figsize=(8, 5))
    ax = sns.histplot(df["CCEI"], bins=20, kde=True, color="#2ca02c", edgecolor="black")
    plt.axvline(0.95, color='red', linestyle='--', linewidth=2, label='Rationality Threshold (0.95)')
    plt.title("Distribution of Afriat Critical Cost Efficiency Index (CCEI)", fontsize=14, fontweight='bold')
    plt.xlabel("CCEI (1.0 = Perfectly Rational)", fontsize=12)
    plt.ylabel("Number of Synthetic Subjects", fontsize=12)
    plt.legend()
    plt.savefig(PLOT_PATH_CCEI, dpi=300, bbox_inches='tight')
    plt.close('all')

    print(f"分析流水线执行完毕！所有成果（CSV、TXT报表、两张高清图）已保存至 '{OUTPUT_DIR}' 文件夹中。")

if __name__ == "__main__":
    generate_analysis_and_plots()