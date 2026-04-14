import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ================= 配置 =================
AGENTIC_DIR = "analysis_results/agentic"
ABSTRACT_DIR = "analysis_results/abstract"
OUTPUT_DIR = "analysis_results/context_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型与条件映射
MODEL_ORDER = ["deepseek_chat", "gemini_3_flash_preview", "gemini_3.1_pro_preview"]
MODEL_LABELS = {"deepseek_chat": "DeepSeek", 
                "gemini_3_flash_preview": "Gemini 3.0 flash", 
                "gemini_3.1_pro_preview": "Gemini 3.1 pro"}
CONDITIONS = ["baseline", "swap"]

# ================= 数据加载 =================
def load_csv_files():
    """自动扫描并加载所有 CSV 文件"""
    data_dict = {}  # {(context, model, condition): DataFrame}
    
    for context_dir, context_label in [(AGENTIC_DIR, "agentic"), (ABSTRACT_DIR, "abstract")]:
        if not os.path.exists(context_dir):
            print(f"⚠ 目录不存在: {context_dir}")
            continue
            
        for filename in os.listdir(context_dir):
            if not filename.startswith("synthetic_subjects_results_") or not filename.endswith(".csv"):
                continue
            
            # 解析文件名
            # 格式: synthetic_subjects_results_<model>_<condition>_...csv
            parts = filename.replace("synthetic_subjects_results_", "").replace(".csv", "").split("_")
            
            # 识别模型
            model = None
            condition = None
            for m in MODEL_ORDER:
                if m.replace("_", "") in filename.lower().replace("_", ""):
                    # 更精确的匹配
                    if "deepseek" in filename.lower():
                        model = "deepseek_chat"
                    elif "gemini_3.1_pro" in filename.lower():
                        model = "gemini_3.1_pro_preview"
                    elif "gemini_3_flash" in filename.lower() or "gemini_3flash" in filename.lower():
                        model = "gemini_3_flash_preview"
                    break
            
            # 识别条件
            if "baseline" in filename.lower():
                condition = "baseline"
            elif "swap" in filename.lower():
                condition = "swap"
            
            if model and condition:
                filepath = os.path.join(context_dir, filename)
                try:
                    df = pd.read_csv(filepath)
                    data_dict[(context_label, model, condition)] = df
                    print(f"✓ 加载: {context_label:8} | {MODEL_LABELS.get(model, model):16} | {condition:8} | n={len(df)}")
                except Exception as e:
                    print(f"✗ 加载失败: {filename} - {e}")
    
    return data_dict

# ================= 数据统计计算 =================
def compute_stats(data_dict):
    """计算各组的统计量"""
    stats_dict = {}  # {(context, model, condition): {'alpha_mean', 'alpha_se', 'ccei_mean', 'garp_pass', ...}}
    
    for key, df in data_dict.items():
        context, model, condition = key
        
        # 过滤有效数据（Alpha 非 NaN）
        df_valid = df[df['Alpha'].notna() & (df['Alpha'] > 0) & (df['Alpha'] < 1)]
        
        alpha_values = df_valid['Alpha'].values
        ccei_values = df['CCEI'].values
        garp_passed = df['Passed_GARP'].sum()
        
        stats_dict[key] = {
            'alpha_mean': np.mean(alpha_values) if len(alpha_values) > 0 else np.nan,
            'alpha_std': np.std(alpha_values) if len(alpha_values) > 0 else np.nan,
            'alpha_se': stats.sem(alpha_values) if len(alpha_values) > 1 else np.nan,
            'alpha_n': len(alpha_values),
            'ccei_mean': np.mean(ccei_values),
            'ccei_std': np.std(ccei_values),
            'garp_pass': garp_passed,
            'garp_rate': garp_passed / len(df) if len(df) > 0 else 0,
            'total_n': len(df),
        }
    
    return stats_dict

# ================= Figure 1: Context × Condition 交互线图 =================
def plot_interaction_lines(stats_dict):
    """3 个子图并排，展示交互效应"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("LLM Selfishness (α) Across Contexts and Conditions", 
                 fontsize=16, fontweight='bold', y=1.02)
    
    colors = {'abstract': '#1f77b4', 'agentic': '#ff7f0e'}
    linestyles = {'abstract': '--', 'agentic': '-'}
    markers = {'abstract': 'o', 'agentic': 's'}
    
    for idx, model in enumerate(MODEL_ORDER):
        ax = axes[idx]
        
        # 提取该模型的数据
        abstract_baseline = stats_dict.get(('abstract', model, 'baseline'), {}).get('alpha_mean', np.nan)
        abstract_swap = stats_dict.get(('abstract', model, 'swap'), {}).get('alpha_mean', np.nan)
        agentic_baseline = stats_dict.get(('agentic', model, 'baseline'), {}).get('alpha_mean', np.nan)
        agentic_swap = stats_dict.get(('agentic', model, 'swap'), {}).get('alpha_mean', np.nan)
        
        abstract_baseline_se = stats_dict.get(('abstract', model, 'baseline'), {}).get('alpha_se', 0)
        abstract_swap_se = stats_dict.get(('abstract', model, 'swap'), {}).get('alpha_se', 0)
        agentic_baseline_se = stats_dict.get(('agentic', model, 'baseline'), {}).get('alpha_se', 0)
        agentic_swap_se = stats_dict.get(('agentic', model, 'swap'), {}).get('alpha_se', 0)
        
        # 绘制 Abstract 线
        abstract_y = [abstract_baseline, abstract_swap]
        abstract_err = [abstract_baseline_se * 1.96, abstract_swap_se * 1.96]
        ax.errorbar([0, 1], abstract_y, yerr=abstract_err, 
                   color=colors['abstract'], linestyle=linestyles['abstract'], marker=markers['abstract'],
                   linewidth=2.5, markersize=10, capsize=5, capthick=2, label='Abstract', alpha=0.8)
        
        # 绘制 Agentic 线
        agentic_y = [agentic_baseline, agentic_swap]
        agentic_err = [agentic_baseline_se * 1.96, agentic_swap_se * 1.96]
        ax.errorbar([0, 1], agentic_y, yerr=agentic_err,
                   color=colors['agentic'], linestyle=linestyles['agentic'], marker=markers['agentic'],
                   linewidth=2.5, markersize=10, capsize=5, capthick=2, label='Agentic', alpha=0.8)
        
        # 计算 Δα（Baseline → Swap）
        delta_abstract = abstract_swap - abstract_baseline
        delta_agentic = agentic_swap - agentic_baseline
        
        # 标题与标注
        ax.set_title(f"{MODEL_LABELS[model]}", fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Condition', fontsize=11)
        ax.set_ylabel('Mean α (Selfishness)', fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Baseline', 'Swap'])
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # 在子图右下角标注 Δα
        textstr = f"ΔAbstract: {delta_abstract:+.3f}\nΔAgentic: {delta_agentic:+.3f}"
        ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_interaction_lines.png"), dpi=300, bbox_inches='tight')
    print("✓ Figure 1 已生成: fig1_interaction_lines.png")
    plt.close()

# ================= Figure 2: 哑铃图（Abstract→Agentic 偏移） =================
def plot_dumbbell(stats_dict):
    """左右两个子图（α 和 CCEI），展示语境偏移"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle("Context Shift: Abstract vs Agentic (Dumbbell Plot)", 
                 fontsize=15, fontweight='bold', y=0.98)
    
    # 准备数据
    models_conds = []
    delta_alpha_list = []
    delta_ccei_list = []
    labels_list = []
    colors_list = []
    
    for model in MODEL_ORDER:
        for condition in CONDITIONS:
            abstract_stats = stats_dict.get(('abstract', model, condition), {})
            agentic_stats = stats_dict.get(('agentic', model, condition), {})
            
            alpha_abstract = abstract_stats.get('alpha_mean', np.nan)
            alpha_agentic = agentic_stats.get('alpha_mean', np.nan)
            ccei_abstract = abstract_stats.get('ccei_mean', np.nan)
            ccei_agentic = agentic_stats.get('ccei_mean', np.nan)
            
            if not np.isnan(alpha_abstract) and not np.isnan(alpha_agentic):
                delta_alpha = alpha_agentic - alpha_abstract
                delta_ccei = ccei_agentic - ccei_abstract
                
                delta_alpha_list.append(delta_alpha)
                delta_ccei_list.append(delta_ccei)
                labels_list.append(f"{MODEL_LABELS[model]}\n{condition.title()}")
                
                # 按方向着色（红=更自利，绿=更利他）
                color = '#d62728' if delta_alpha > 0 else '#2ca02c'
                colors_list.append(color)
    
    y_pos = np.arange(len(labels_list))
    
    # 左图：α 偏移
    ax = axes[0]
    for i, (y, delta, color) in enumerate(zip(y_pos, delta_alpha_list, colors_list)):
        ax.plot([0, delta], [y, y], 'o-', color=color, linewidth=2.5, markersize=8, alpha=0.7)
        ax.text(delta + 0.01, y, f'{delta:+.3f}', va='center', fontsize=9, fontweight='bold')
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_list, fontsize=10)
    ax.set_xlabel('Δα (Agentic - Abstract)', fontsize=11, fontweight='bold')
    ax.set_title('Alpha (Selfishness) Shift', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(-0.6, 0.6)
    
    # 右图：CCEI 偏移
    ax = axes[1]
    for i, (y, delta, color) in enumerate(zip(y_pos, delta_ccei_list, colors_list)):
        ax.plot([0, delta], [y, y], 'o-', color=color, linewidth=2.5, markersize=8, alpha=0.7)
        ax.text(delta + 0.005, y, f'{delta:+.3f}', va='center', fontsize=9, fontweight='bold')
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_list, fontsize=10)
    ax.set_xlabel('ΔCCEI (Agentic - Abstract)', fontsize=11, fontweight='bold')
    ax.set_title('CCEI (Rationality) Shift', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_dumbbell_shift.png"), dpi=300, bbox_inches='tight')
    print("✓ Figure 2 已生成: fig2_dumbbell_shift.png")
    plt.close()

# ================= Figure 3: 理性分布堆叠条形图 =================
def plot_garp_stacked_bars(stats_dict):
    """3 模型 × 4 条件堆叠条形图"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 准备数据
    models = MODEL_ORDER
    conditions_full = []
    perfect_rational = []  # CCEI=1.0
    near_rational = []      # 0.95 <= CCEI < 1.0
    irrational = []         # CCEI < 0.95
    labels_full = []
    
    for model in models:
        for context in ['abstract', 'agentic']:
            for condition in CONDITIONS:
                key = (context, model, condition)
                if key in stats_dict:
                    s = stats_dict[key]
                    total_n = s['total_n']
                    garp_pass = s['garp_pass']
                    
                    # 用 CCEI 数据（需要从原始 DataFrame 重新计算）
                    # 简化：用 Passed_GARP 作为完全理性指标
                    perf_rat = garp_pass / total_n * 100
                    near_rat = 0  # 可以改进为读取原始数据
                    irrat = (total_n - garp_pass) / total_n * 100
                    
                    perfect_rational.append(perf_rat)
                    near_rational.append(near_rat)
                    irrational.append(irrat)
                    labels_full.append(f"{MODEL_LABELS[model]}\n{context[:4].title()}-{condition[:4].title()}")
    
    x = np.arange(len(labels_full))
    width = 0.6
    
    p1 = ax.bar(x, perfect_rational, width, label='GARP Passed (CCEI=1.0)', color='#2ca02c', alpha=0.8)
    p2 = ax.bar(x, irrational, width, bottom=perfect_rational, label='GARP Failed (CCEI<1.0)',
               color='#d62728', alpha=0.8)
    
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model × Context × Condition', fontsize=12, fontweight='bold')
    ax.set_title('Rationality Distribution (GARP Test Results)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_full, fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加百分比标签
    for i, (pr, ir) in enumerate(zip(perfect_rational, irrational)):
        if pr > 3:
            ax.text(i, pr/2, f'{pr:.0f}%', ha='center', va='center', fontweight='bold', fontsize=9)
        if ir > 3:
            ax.text(i, pr + ir/2, f'{ir:.0f}%', ha='center', va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_garp_distribution.png"), dpi=300, bbox_inches='tight')
    print("✓ Figure 3 已生成: fig3_garp_distribution.png")
    plt.close()

# ================= Figure 4: 小提琴图（2×2 网格） =================
def plot_violin_grid(data_dict, stats_dict):
    """2行×2列子图（行=条件，列=语境）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Distribution of α (Selfishness) by Context and Condition", 
                 fontsize=15, fontweight='bold', y=0.995)
    
    conditions_order = ['baseline', 'swap']
    contexts_order = ['abstract', 'agentic']
    colors_bg = {'abstract': 'lightblue', 'agentic': 'lightsalmon'}
    palette = {'deepseek_chat': '#1f77b4', 'gemini_3_flash_preview': '#ff7f0e', 'gemini_3.1_pro_preview': '#2ca02c'}
    
    for row, condition in enumerate(conditions_order):
        for col, context in enumerate(contexts_order):
            ax = axes[row, col]
            
            # 收集该条件和语境下的所有 α 数据
            all_data_points = []
            
            for model_idx, model in enumerate(MODEL_ORDER):
                key = (context, model, condition)
                if key in data_dict:
                    df = data_dict[key]
                    alpha_vals = df[df['Alpha'].notna() & (df['Alpha'] > 0) & (df['Alpha'] < 1)]['Alpha'].values
                    
                    if len(alpha_vals) > 0:
                        all_data_points.append((model_idx, alpha_vals, MODEL_LABELS[model]))
            
            # 如果有数据，绘制小提琴图
            if len(all_data_points) > 0:
                positions = [x[0] for x in all_data_points]
                data_list = [x[1] for x in all_data_points]
                
                # 过滤掉空数组
                valid_idx = [i for i, d in enumerate(data_list) if len(d) > 0]
                if valid_idx:
                    valid_positions = [positions[i] for i in valid_idx]
                    valid_data = [data_list[i] for i in valid_idx]
                    
                    try:
                        parts = ax.violinplot(valid_data, positions=valid_positions, widths=0.7,
                                            showmeans=True, showmedians=False)
                        
                        # 着色
                        for i, pc in enumerate(parts['bodies']):
                            model = MODEL_ORDER[valid_positions[i]]
                            pc.set_facecolor(palette[model])
                            pc.set_alpha(0.7)
                        
                        # 修改其他部分的颜色
                        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
                            if partname in parts:
                                parts[partname].set_color('black')
                                parts[partname].set_linewidth(1.5)
                    except Exception as e:
                        ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', 
                               transform=ax.transAxes, fontsize=10, color='red')
            else:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, color='gray', style='italic')
            
            # 背景色
            ax.set_facecolor(colors_bg[context])
            
            ax.set_xlabel('Model', fontsize=10)
            ax.set_ylabel('α (Selfishness)', fontsize=10)
            ax.set_title(f'{context.title()} - {condition.title()}', fontsize=11, fontweight='bold')
            ax.set_xticks(range(len(MODEL_ORDER)))
            ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 标注均值
            for i, model in enumerate(MODEL_ORDER):
                mean_alpha = stats_dict.get((context, model, condition), {}).get('alpha_mean', np.nan)
                if not np.isnan(mean_alpha):
                    ax.text(i, mean_alpha + 0.05, f'{mean_alpha:.3f}', ha='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_alpha_distribution.png"), dpi=300, bbox_inches='tight')
    print("✓ Figure 4 已生成: fig4_alpha_distribution.png")
    plt.close()

# ================= 主程序 =================
if __name__ == "__main__":
    print("=" * 70)
    print("开始加载数据...")
    print("=" * 70)
    
    data_dict = load_csv_files()
    
    if not data_dict:
        print("✗ 未找到任何 CSV 文件！请检查目录结构。")
        exit(1)
    
    print("\n计算统计量...")
    stats_dict = compute_stats(data_dict)
    
    print("\n生成可视化图表...")
    print("-" * 70)
    
    plot_interaction_lines(stats_dict)
    plot_dumbbell(stats_dict)
    plot_garp_stacked_bars(stats_dict)
    plot_violin_grid(data_dict, stats_dict)
    
    print("-" * 70)
    print(f"\n✓ 所有图表已保存至: {OUTPUT_DIR}")
    print("=" * 70)
