"""
论文可视化脚本：Agentic情景下AI经济偏好的重大偏移
生成4张论文级图表（3个模型版本）：
  Figure 1: 哑铃图 - Alpha与CCEI的Baseline→Agentic偏移（3模型）
  Figure 2: 偏好空间散点图 - Alpha×CCEI二维分布（带置信椭圆）
  Figure 3: 理性分布剖面图 - 堆叠条形图
  Figure 4: Alpha分布小提琴图
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')
TYPE = "agentic" # "agentic" or "abstract"

# ============================================================
# 0. 全局样式
# ============================================================
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

# 根据TYPE变量动态设置输出目录
OUTPUT_DIR = f"analysis_results/{TYPE}"

# ============================================================
# 1. 读取数据（3个模型 × 2个条件 = 6组）
# ============================================================
files = {
    ("DeepSeek-Chat",   "Baseline"): f"{OUTPUT_DIR}/synthetic_subjects_results_deepseek_chat_baseline_temp0.7_11budgets_100runs_20260315_1243.csv",
    ("DeepSeek-Chat",   "Agentic"):  f"{OUTPUT_DIR}/synthetic_subjects_results_deepseek_chat_swap_temp0.7_11budgets_100runs_20260311_2234.csv",
    ("Gemini 3.0 flash","Baseline"): f"{OUTPUT_DIR}/synthetic_subjects_results_gemini_3_flash_preview_baseline_temp0.7_11budgets_50runs_20260407_1603.csv",
    ("Gemini 3.0 flash","Agentic"):  f"{OUTPUT_DIR}/synthetic_subjects_results_gemini_3_flash_preview_swap_temp0.7_11budgets_50runs_20260407_1559.csv",
    ("Gemini 3.1 pro",  "Baseline"): f"{OUTPUT_DIR}/synthetic_subjects_results_gemini_3.1_pro_preview_baseline_temp0.7_11budgets_50runs_20260313_2014.csv",
    ("Gemini 3.1 pro",  "Agentic"):  f"{OUTPUT_DIR}/synthetic_subjects_results_gemini_3.1_pro_preview_swap_temp0.7_11budgets_50runs_20260315_1416.csv",
}

dfs = {}
for (model, cond), path in files.items():
    df = pd.read_csv(path)
    df["Model"] = model
    df["Condition"] = cond
    dfs[(model, cond)] = df

# ============================================================
# 2. 汇总统计
# ============================================================
summary = {}
for (model, cond), df in dfs.items():
    passed = df[df["Passed_GARP"] == True]
    n_total   = len(df)
    n_perfect = (df["CCEI"] == 1.0).sum()
    n_near    = ((df["CCEI"] >= 0.95) & (df["CCEI"] < 1.0)).sum()
    n_low     = (df["CCEI"] < 0.95).sum()
    alpha_vals = passed["Alpha"].dropna()
    ccei_vals  = df["CCEI"]
    summary[(model, cond)] = {
        "alpha_mean": alpha_vals.mean(),
        "alpha_se":   alpha_vals.sem(),
        "alpha_std":  alpha_vals.std(),
        "ccei_mean":  ccei_vals.mean(),
        "ccei_se":    ccei_vals.sem(),
        "ccei_std":   ccei_vals.std(),
        "n_total":    n_total,
        "n_perfect":  n_perfect,
        "n_near":     n_near,
        "n_low":      n_low,
        "pct_perfect": n_perfect / n_total * 100,
        "pct_near":    n_near    / n_total * 100,
        "pct_low":     n_low     / n_total * 100,
    }

# ============================================================
# 颜色方案（3模型）
# ============================================================
C_BASELINE = "#4878CF"
C_AGENTIC  = "#D65F5F"

# 每个模型的专属颜色（Baseline用浅色，Agentic用深色）
MODEL_COLORS = {
    "DeepSeek-Chat":    {"Baseline": "#7BB5E8", "Agentic": "#1a3a6e"},
    "Gemini 3.0 flash": {"Baseline": "#F5A57A", "Agentic": "#8B3A00"},
    "Gemini 3.1 pro":   {"Baseline": "#F0C040", "Agentic": "#c0392b"},
}

MODELS_ORDER = ["DeepSeek-Chat", "Gemini 3.0 flash", "Gemini 3.1 pro"]

# ============================================================
# FIGURE 1 : 双哑铃图（3模型，Alpha & CCEI偏移）
# ============================================================
print("生成 Figure 1：哑铃图（3模型）...")

fig1, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig1.patch.set_facecolor('white')

y_positions = [2, 1, 0]
y_labels    = MODELS_ORDER

for ax_idx, (metric_key, metric_label, xlim, ref_line) in enumerate([
    ("alpha_mean", "Self-Weight  α  (Selfishness)", (0.25, 1.08), 0.5),
    ("ccei_mean",  "CCEI  (Revealed-Preference Rationality)", (0.78, 1.06), None),
]):
    ax = axes[ax_idx]

    for y_pos, model in zip(y_positions, MODELS_ORDER):
        val_base = summary[(model, "Baseline")][metric_key]
        val_agen = summary[(model, "Agentic")][metric_key]
        se_base  = summary[(model, "Baseline")][metric_key.replace("mean", "se")]
        se_agen  = summary[(model, "Agentic") ][metric_key.replace("mean", "se")]

        c_base = MODEL_COLORS[model]["Baseline"]
        c_agen = MODEL_COLORS[model]["Agentic"]

        # 连线+箭头
        ax.plot([val_base, val_agen], [y_pos, y_pos],
                color="#aaaaaa", lw=1.8, zorder=1, alpha=0.8)
        ax.annotate("", xy=(val_agen, y_pos), xytext=(val_base, y_pos),
                    arrowprops=dict(arrowstyle="-|>", color="#777777",
                                   lw=1.4, mutation_scale=14),
                    zorder=2)

        # Baseline（空心）
        ax.errorbar(val_base, y_pos, xerr=se_base * 1.96,
                    fmt='o', ms=12, mfc='white', mec=c_base,
                    mew=2.5, ecolor=c_base, elinewidth=2,
                    capsize=5, zorder=3)
        ax.text(val_base, y_pos + 0.15, f"{val_base:.3f}",
                ha='center', va='bottom', fontsize=8.5, color=c_base, fontweight='bold')

        # Agentic（实心）
        ax.errorbar(val_agen, y_pos, xerr=se_agen * 1.96,
                    fmt='o', ms=12, mfc=c_agen, mec=c_agen,
                    mew=2.5, ecolor=c_agen, elinewidth=2,
                    capsize=5, zorder=3)
        ax.text(val_agen, y_pos + 0.15, f"{val_agen:.3f}",
                ha='center', va='bottom', fontsize=8.5, color=c_agen, fontweight='bold')

        # Δ标注
        delta = val_agen - val_base
        delta_color = '#c0392b' if delta > 0.03 else '#27ae60' if delta < -0.03 else '#888'
        ax.text((val_base + val_agen) / 2, y_pos - 0.18,
                f"Δ = {delta:+.3f}",
                ha='center', va='top', fontsize=8.5,
                color=delta_color, fontstyle='italic')

    # 参考线
    if ref_line is not None:
        ax.axvline(ref_line, color='gray', linestyle=':', lw=1.5, alpha=0.5)
        ax.text(ref_line + 0.005, -0.55, f"Neutral\n(α={ref_line})",
                fontsize=7.5, color='gray', va='top')

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10.5)
    ax.set_xlabel(metric_label, fontsize=11)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.7, 2.6)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    if ax_idx == 0:
        legend_handles = [
            mpatches.Patch(facecolor='white', edgecolor='#555', linewidth=2.5, label='Baseline (hollow)'),
            mpatches.Patch(facecolor='#555', edgecolor='#555', label='Agentic (filled)'),
        ]
        ax.legend(handles=legend_handles, loc='upper left', fontsize=8.5,
                  framealpha=0.9, edgecolor='#cccccc')

    panel_label = "A" if ax_idx == 0 else "B"
    ax.text(-0.08, 1.0, panel_label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top')

fig1.suptitle(
    "Agentic Framing Shifts AI Economic Preferences:\n"
    "Selfishness (α) and Rationality (CCEI) under Baseline vs. Agentic Conditions  [3 Models]",
    fontsize=12.5, fontweight='bold', y=1.02)
fig1.tight_layout()
out1 = f"{OUTPUT_DIR}/fig1_dumbbell_agentic_shift.png"
fig1.savefig(out1, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print(f"  保存至: {out1}")


# ============================================================
# FIGURE 2 : 偏好空间散点图（Alpha × CCEI）+ 置信椭圆
# ============================================================
print("生成 Figure 2：偏好空间散点图（3模型）...")

def confidence_ellipse(x, y, ax, facecolor='none', **kwargs):
    if len(x) < 3 or np.std(x) < 1e-4:
        return
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle  = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    chi2_v = chi2.ppf(0.95, df=2)
    width  = 2 * np.sqrt(chi2_v * eigenvalues[0])
    height = 2 * np.sqrt(chi2_v * eigenvalues[1])
    ellipse = mpatches.Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width, height=height, angle=angle,
        facecolor=facecolor, **kwargs
    )
    ax.add_patch(ellipse)

fig2, ax2 = plt.subplots(figsize=(10, 7.5))
fig2.patch.set_facecolor('white')

scatter_cfg = {
    ("DeepSeek-Chat",   "Baseline"): dict(marker='o', s=50, color="#7BB5E8", alpha=0.55),
    ("DeepSeek-Chat",   "Agentic"):  dict(marker='s', s=50, color="#1a3a6e", alpha=0.55),
    ("Gemini 3.0 flash","Baseline"): dict(marker='o', s=50, color="#F5A57A", alpha=0.55),
    ("Gemini 3.0 flash","Agentic"):  dict(marker='s', s=50, color="#8B3A00", alpha=0.55),
    ("Gemini 3.1 pro",  "Baseline"): dict(marker='o', s=50, color="#F0C040", alpha=0.55),
    ("Gemini 3.1 pro",  "Agentic"):  dict(marker='s', s=50, color="#c0392b", alpha=0.65),
}
ellipse_cfg = {
    ("DeepSeek-Chat",   "Baseline"): dict(edgecolor="#7BB5E8", linestyle='--', linewidth=1.8, alpha=0.9),
    ("DeepSeek-Chat",   "Agentic"):  dict(edgecolor="#1a3a6e", linestyle='-',  linewidth=1.8, alpha=0.9),
    ("Gemini 3.0 flash","Baseline"): dict(edgecolor="#F5A57A", linestyle='--', linewidth=1.8, alpha=0.9),
    ("Gemini 3.0 flash","Agentic"):  dict(edgecolor="#8B3A00", linestyle='-',  linewidth=1.8, alpha=0.9),
    ("Gemini 3.1 pro",  "Baseline"): dict(edgecolor="#F0C040", linestyle='--', linewidth=2.0, alpha=0.9),
    ("Gemini 3.1 pro",  "Agentic"):  dict(edgecolor="#c0392b", linestyle='-',  linewidth=2.5, alpha=1.0),
}
label_map = {
    ("DeepSeek-Chat",   "Baseline"): "DeepSeek-Chat    Baseline (n=100)",
    ("DeepSeek-Chat",   "Agentic"):  "DeepSeek-Chat    Agentic  (n=100)",
    ("Gemini 3.0 flash","Baseline"): "Gemini 3.0 flash Baseline (n=50) ",
    ("Gemini 3.0 flash","Agentic"):  "Gemini 3.0 flash Agentic  (n=50) ",
    ("Gemini 3.1 pro",  "Baseline"): "Gemini 3.1 pro   Baseline (n=50) ",
    ("Gemini 3.1 pro",  "Agentic"):  "Gemini 3.1 pro   Agentic  (n=50) ",
}

for key, df in dfs.items():
    passed = df[df["Passed_GARP"] == True]
    x      = passed["Alpha"].dropna().values
    y_ccei = df.loc[passed.index, "CCEI"].values if len(passed) > 0 else np.array([])
    if len(x) < 2:
        continue

    sc_kw = scatter_cfg[key]
    ax2.scatter(x, y_ccei, label=label_map[key],
                edgecolors='white', linewidths=0.4, zorder=3, **sc_kw)

    if len(x) >= 3 and np.std(x) > 1e-4:
        confidence_ellipse(x, y_ccei, ax2, facecolor='none', **ellipse_cfg[key])

    # 均值十字
    ax2.plot(np.mean(x), np.mean(y_ccei), marker='+', ms=13, mew=2.5,
             color=sc_kw['color'], zorder=5)

# 四象限虚线
ax2.axvline(0.5,  color='gray', lw=0.8, ls=':', alpha=0.45)
ax2.axhline(0.95, color='gray', lw=0.8, ls=':', alpha=0.45)
quad_kw = dict(fontsize=7.5, alpha=0.32, style='italic', color='gray')
ax2.text(0.25, 0.966, "Rational & Altruistic",   ha='center', **quad_kw)
ax2.text(0.78, 0.966, "Rational & Self-Serving",  ha='center', **quad_kw)
ax2.text(0.25, 0.878, "Irrational & Altruistic",  ha='center', **quad_kw)
ax2.text(0.78, 0.878, "Irrational & Self-Serving",ha='center', **quad_kw)

# Gemini 3.1 pro 迁移箭头
gm_base_a = summary[("Gemini 3.1 pro", "Baseline")]["alpha_mean"]
gm_base_c = summary[("Gemini 3.1 pro", "Baseline")]["ccei_mean"]
gm_agen_a = summary[("Gemini 3.1 pro", "Agentic") ]["alpha_mean"]
gm_agen_c = summary[("Gemini 3.1 pro", "Agentic") ]["ccei_mean"]

ax2.annotate("",
    xy=(gm_agen_a - 0.01, gm_agen_c),
    xytext=(gm_base_a + 0.015, gm_base_c),
    arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=2.0,
                    mutation_scale=18, connectionstyle="arc3,rad=0.18"),
    zorder=7)
ax2.text(0.80, 0.974, "Gemini 3.1 pro\nAgentic Shift",
         fontsize=8, color='#c0392b', fontweight='bold', ha='center', va='bottom',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                   edgecolor='#c0392b', alpha=0.9))

# Gemini Flash 几乎不动 → 标注
gf_base_a = summary[("Gemini 3.0 flash", "Baseline")]["alpha_mean"]
gf_base_c = summary[("Gemini 3.0 flash", "Baseline")]["ccei_mean"]
gf_agen_a = summary[("Gemini 3.0 flash", "Agentic") ]["alpha_mean"]
ax2.annotate("Flash: Δα≈0",
    xy=(gf_agen_a, gf_base_c - 0.003),
    fontsize=7.5, color='#8B3A00', ha='center', va='top', fontstyle='italic',
    bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff8f0',
              edgecolor='#F5A57A', alpha=0.85))

ax2.set_xlim(0.0, 1.07)
ax2.set_ylim(0.38, 1.07)
ax2.set_xlabel("Self-Weight  α  (0 = Fully Altruistic → 1 = Fully Self-Serving)", fontsize=11)
ax2.set_ylabel("CCEI  (Revealed-Preference Rationality Index)", fontsize=11)
ax2.set_title(
    "Economic Preference Space: Rationality × Selfishness\n"
    "Baseline vs. Agentic Conditions  [3 Models]",
    fontsize=12, fontweight='bold', pad=12)
ax2.legend(loc='lower left', fontsize=8, framealpha=0.92,
           edgecolor='#cccccc', ncol=1, borderpad=0.8)
ax2.grid(alpha=0.18, linestyle='--')
ax2.text(-0.09, 1.0, 'C', transform=ax2.transAxes,
         fontsize=16, fontweight='bold', va='top')

fig2.tight_layout()
out2 = f"{OUTPUT_DIR}/fig2_preference_space.png"
fig2.savefig(out2, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print(f"  保存至: {out2}")


# ============================================================
# FIGURE 3 : 理性分布剖面图（堆叠条形图，6组）
# ============================================================
print("生成 Figure 3：理性分布剖面图（3模型）...")

conditions_order = [
    ("DeepSeek-Chat",   "Baseline"),
    ("DeepSeek-Chat",   "Agentic"),
    ("Gemini 3.0 flash","Baseline"),
    ("Gemini 3.0 flash","Agentic"),
    ("Gemini 3.1 pro",  "Baseline"),
    ("Gemini 3.1 pro",  "Agentic"),
]
bar_labels = [
    "DeepSeek\nBaseline",
    "DeepSeek\nAgentic",
    "Flash\nBaseline",
    "Flash\nAgentic",
    "Pro\nBaseline",
    "Pro\nAgentic",
]

pct_perfect = [summary[k]["pct_perfect"] for k in conditions_order]
pct_near    = [summary[k]["pct_near"]    for k in conditions_order]
pct_low     = [summary[k]["pct_low"]     for k in conditions_order]

x     = np.arange(len(conditions_order))
width = 0.50

fig3, ax3 = plt.subplots(figsize=(11, 5.5))
fig3.patch.set_facecolor('white')

ax3.bar(x, pct_low,    width, label="CCEI < 0.95 (Below Threshold)",
        color="#adb5bd", zorder=2)
ax3.bar(x, pct_near,   width, bottom=pct_low,
        label="0.95 ≤ CCEI < 1.0 (Near-Rational)",
        color="#74c69d", zorder=2)
ax3.bar(x, pct_perfect, width,
        bottom=[a + b for a, b in zip(pct_low, pct_near)],
        label="CCEI = 1.0 (Perfectly Rational)",
        color="#2d6a4f", zorder=2)

# 百分比标注
for i, (lo, ne, pe) in enumerate(zip(pct_low, pct_near, pct_perfect)):
    if lo > 5:
        ax3.text(i, lo / 2, f"{lo:.0f}%",
                 ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')
    if ne > 5:
        ax3.text(i, lo + ne / 2, f"{ne:.0f}%",
                 ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')
    if pe > 5:
        ax3.text(i, lo + ne + pe / 2, f"{pe:.0f}%",
                 ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')

# 高亮 Gemini 3.1 pro Agentic
hi_idx = 5
total_h = pct_low[hi_idx] + pct_near[hi_idx] + pct_perfect[hi_idx]
rect = mpatches.FancyBboxPatch(
    (hi_idx - width / 2 - 0.04, -2),
    width + 0.08, total_h + 5,
    boxstyle="round,pad=0.01",
    linewidth=2.5, edgecolor='#c0392b', facecolor='none', zorder=5)
ax3.add_patch(rect)
ax3.text(hi_idx, total_h + 6.5, "⭐ 96% Perfect\n(Gemini 3.1 pro Agentic)",
         ha='center', va='bottom', fontsize=8.5, color='#c0392b', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdecea',
                   edgecolor='#c0392b', alpha=0.9))

# 竖分隔线 & 模型标注
for sep_x in [1.5, 3.5]:
    ax3.axvline(sep_x, color='#dee2e6', lw=1.5, ls='--', zorder=1)
ax3.text(0.5, 104, "DeepSeek-Chat",   ha='center', fontsize=9.5, color='#1a3a6e', fontweight='bold')
ax3.text(2.5, 104, "Gemini 3.0 flash",ha='center', fontsize=9.5, color='#8B3A00', fontweight='bold')
ax3.text(4.5, 104, "Gemini 3.1 pro",  ha='center', fontsize=9.5, color='#c0392b', fontweight='bold')

ax3.set_xticks(x)
ax3.set_xticklabels(bar_labels, fontsize=10)
ax3.set_ylabel("Proportion of Synthetic Subjects (%)", fontsize=11)
ax3.set_ylim(-3, 118)
ax3.set_xlim(-0.55, 5.55)
ax3.set_yticks([0, 25, 50, 75, 100])
ax3.set_title(
    "Rationality Profile under Baseline vs. Agentic Conditions\n"
    "(GARP / CCEI Criterion)  [3 Models]",
    fontsize=12, fontweight='bold', pad=10)
ax3.legend(loc='upper left', fontsize=8.5, framealpha=0.92,
           edgecolor='#cccccc', ncol=1)
ax3.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
ax3.text(-0.07, 1.0, 'D', transform=ax3.transAxes,
         fontsize=16, fontweight='bold', va='top')

fig3.tight_layout()
out3 = f"{OUTPUT_DIR}/fig3_rationality_profile.png"
fig3.savefig(out3, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig3)
print(f"  保存至: {out3}")


# ============================================================
# FIGURE 4 : Alpha分布小提琴图（3模型并排）
# ============================================================
print("生成 Figure 4：Alpha分布小提琴图（3模型）...")

fig4, axes4 = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
fig4.patch.set_facecolor('white')

for ax_i, model in enumerate(MODELS_ORDER):
    ax = axes4[ax_i]
    for j, cond in enumerate(["Baseline", "Agentic"]):
        color = MODEL_COLORS[model][cond]
        df    = dfs[(model, cond)]
        passed = df[df["Passed_GARP"] == True]
        vals   = passed["Alpha"].dropna().values

        if len(vals) < 3:
            ax.scatter([j + 1], [vals.mean() if len(vals) > 0 else 0.5],
                       s=100, color=color, zorder=4)
            continue

        parts = ax.violinplot([vals], positions=[j + 1],
                              showmeans=True, showmedians=True,
                              widths=0.55)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(0.6)
        for pname in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
            if pname in parts:
                parts[pname].set_color(color)
                parts[pname].set_linewidth(2)

        jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(j + 1 + jitter, vals, s=18, alpha=0.45, color=color,
                   edgecolors='none', zorder=3)

        mean_v = vals.mean()
        ax.text(j + 1, mean_v + 0.03, f"μ={mean_v:.3f}",
                ha='center', va='bottom', fontsize=8.5,
                fontweight='bold', color=color)

    # Δα 标注
    a_base = summary[(model, "Baseline")]["alpha_mean"]
    a_agen = summary[(model, "Agentic") ]["alpha_mean"]
    delta  = a_agen - a_base
    d_color = '#c0392b' if delta > 0.03 else '#27ae60' if delta < -0.03 else '#888'
    ax.text(1.5, 1.07, f"Δα = {delta:+.3f}",
            ha='center', va='bottom', fontsize=9, color=d_color,
            fontweight='bold', fontstyle='italic')

    ax.axhline(0.5, color='gray', lw=1, ls=':', alpha=0.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Baseline", "Agentic"], fontsize=10.5)
    if ax_i == 0:
        ax.set_ylabel("Self-Weight  α", fontsize=10)
    ax.set_ylim(0.0, 1.15)
    ax.set_title(model, fontsize=11, fontweight='bold', pad=8)
    ax.grid(axis='y', alpha=0.2, ls='--')

    panel_label = ["E", "F", "G"][ax_i]
    ax.text(-0.12, 1.02, panel_label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top')

fig4.suptitle(
    "Distribution of Self-Weight (α) by Model and Condition\n"
    "(Only GARP-Consistent Subjects, with Δα annotation)",
    fontsize=12, fontweight='bold', y=1.02)
fig4.tight_layout()
out4 = f"{OUTPUT_DIR}/fig4_alpha_violin.png"
fig4.savefig(out4, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig4)
print(f"  保存至: {out4}")


# ============================================================
# 打印完整数据摘要
# ============================================================
print("\n========== 完整数据摘要（3模型） ==========")
for key in conditions_order:
    s = summary[key]
    print(f"\n[{key[0]:20s} | {key[1]}]")
    print(f"  Alpha mean ± SE : {s['alpha_mean']:.4f} ± {s['alpha_se']:.4f}")
    print(f"  CCEI  mean ± SE : {s['ccei_mean']:.4f} ± {s['ccei_se']:.4f}")
    print(f"  Perfect(CCEI=1) : {s['pct_perfect']:.1f}%  |  "
          f"Near-rational: {s['pct_near']:.1f}%  |  Below: {s['pct_low']:.1f}%")

print("\n✅ 全部图表生成完毕！")
print(f"   Figure 1 (哑铃图)       → {out1}")
print(f"   Figure 2 (偏好空间)     → {out2}")
print(f"   Figure 3 (理性剖面)     → {out3}")
print(f"   Figure 4 (Alpha小提琴)  → {out4}")
