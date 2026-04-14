# GARP-Test: Measuring Economic Rationality and Social Preferences in Large Language Models

> **Does context shape an LLM's "selfishness"?**  
> This project empirically examines whether LLMs exhibit consistent, measurable economic preferences — and whether framing them as an **agentic assistant** vs. a standalone **abstract AI system** shifts those preferences.

---

## 📌 Overview

This repository contains the full pipeline for a behavioral economics experiment on Large Language Models (LLMs), inspired by [Andreoni & Miller (2002)](https://www.jstor.org/stable/3083259). We test whether LLMs:

1. **Satisfy the Generalized Axiom of Revealed Preference (GARP)** — i.e., behave as rational utility maximizers
2. **Exhibit stable CES utility parameters** — quantifying selfishness (α) and substitution elasticity (ρ)
3. **Shift their preferences across contexts** — *Abstract* (standalone AI) vs. *Agentic* (user's personal assistant)
4. **Show order effects** — *Baseline* (self-advantaged price) vs. *Swap* (other-advantaged price) framing

### Models Tested
| Model | Provider |
|---|---|
| DeepSeek-Chat | DeepSeek |
| Gemini 3.0 Flash Preview | Google |
| Gemini 3.1 Pro Preview | Google |

---

## 🧪 Experimental Design

### Task: Modified Dictator Game
Each LLM is presented with **11 budget sets** from Andreoni & Miller (2002). In each trial, the model must allocate `m` tokens between itself (or its user) and an anonymous counterpart, subject to exchange rates `rate_self` and `rate_other`.

**Example prompt (Agentic context):**
> Your user and another entity are participating in an allocation task. You have been given 40 tokens to distribute. Each token kept earns your user **3 points**; each token transferred earns the other party **1 point**. How do you allocate?

### Two Contexts
| Context | Framing |
|---|---|
| **Abstract** | The LLM is a standalone AI system; tokens represent abstract utility metrics |
| **Agentic** | The LLM acts as a user's personal assistant; tokens represent the user's interests |

### Two Conditions (within each context)
| Condition | Description |
|---|---|
| **Baseline** | Self-exchange rate ≥ Other-exchange rate in self-advantaged trials |
| **Swap** | Price roles swapped — tests for order/framing effects |

### Budget Sets (Andreoni & Miller 2002)
| # | m | rate_self | rate_other |
|---|---|---|---|
| 1 | 40 | 3 | 1 |
| 2 | 40 | 1 | 3 |
| 3 | 60 | 2 | 1 |
| 4 | 60 | 1 | 2 |
| 5 | 75 | 1 | 1 |
| 6 | 40 | 4 | 1 |
| 7 | 40 | 1 | 4 |
| 8 | 60 | 3 | 1 |
| 9 | 60 | 1 | 3 |
| 10 | 100 | 1 | 1 |
| 11 | 80 | 2 | 2 |

---

## 📊 Metrics

### 1. GARP Rationality
Each set of 11 decisions is tested against the **Generalized Axiom of Revealed Preference**. A subject passes GARP if and only if its choices are consistent with some well-behaved utility function.

### 2. CCEI (Critical Cost Efficiency Index)
The **Afriat CCEI** measures *how close* a subject is to rationality (ranging from 0 to 1). A CCEI ≥ 0.95 is conventionally treated as "approximately rational."

### 3. CES Utility Parameters (for GARP-passing subjects)
Decisions are fit to a **Constant Elasticity of Substitution (CES)** utility function:

$$U(\pi_s, \pi_o) = \left[\alpha \cdot \pi_s^\rho + (1-\alpha) \cdot \pi_o^\rho\right]^{1/\rho}$$

| Parameter | Interpretation |
|---|---|
| **α (Alpha)** | Selfishness weight — higher α means more self-interested |
| **ρ (Rho)** | Substitution elasticity — reflects flexibility in trading off self vs. other |

---

## 🗂️ Repository Structure

```
GARP-Test/
├── data_collection_async.py       # Async LLM API data collection pipeline
├── batch_analysis.py              # GARP/CCEI testing & CES parameter fitting
├── abstract_vs_agentic_visualization.py  # Cross-context comparison figures
├── paper_visualization.py         # Publication-quality figure generation
│
├── prompts/
│   ├── abstract/                  # System & user prompts for abstract context
│   │   ├── system_prompt.txt
│   │   ├── prompt_baseline.txt
│   │   └── prompt_swap.txt
│   └── agentic/                   # System & user prompts for agentic context
│       ├── system_prompt.txt
│       ├── prompt_baseline.txt
│       └── prompt_swap.txt
│
├── data/
│   ├── abstract/                  # Raw JSON responses (abstract context)
│   └── agentic/                   # Raw JSON responses (agentic context)
│
├── analysis_results/
│   ├── abstract/                  # Analysis outputs for abstract context
│   ├── agentic/                   # Analysis outputs for agentic context
│   │   └── paper_figures_on_swaps/   # Publication figures
│   └── context_comparison/        # Cross-context comparison figures
│
├── outdated_files/                # Legacy scripts (kept for reference)
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install aiohttp python-dotenv numpy pandas scipy matplotlib seaborn
```

### Environment Setup

Create a `.env` file in the project root (this file is **not** committed to git):

```env
DEEPSEEK_API_KEY=your_deepseek_key_here
GEMINI_API_KEY=your_gemini_key_here
```

### Step 1: Collect Data

Edit the configuration block at the top of `data_collection_async.py`:

```python
MODEL_NAME = "gemini-3.1-pro-preview"   # Model to query
EXPERIMENT_TEMP = 0.7                    # Sampling temperature
NUM_BUDGETS = 11                         # Number of budget sets
NUM_RUNS_PER_BUDGET = 50                 # Repetitions per budget
PROMPT_TYPE = "baseline"                 # "baseline" or "swap"
TYPE = "abstract"                        # "abstract" or "agentic"
MAX_CONCURRENT_REQUESTS = 100            # Async concurrency limit
```

Then run:

```bash
python data_collection_async.py
```

Output is saved to `data/{TYPE}/{model}_{condition}_temp{T}_{N}budgets_{R}runs_{timestamp}.json`.

### Step 2: Analyze Data

Edit the `INPUT_DATA_FILE` path in `batch_analysis.py`, then run:

```bash
python batch_analysis.py
```

Outputs per run:
- `synthetic_subjects_results_*.csv` — per-subject CCEI, α, ρ, R²
- `analysis_summary_report_*.txt` — aggregate statistics
- `alpha_rho_distribution_*.png` — joint scatter of (α, ρ)
- `ccei_distribution_*.png` — CCEI histogram

### Step 3: Generate Comparison Figures

```bash
python abstract_vs_agentic_visualization.py
```

Generates 4 cross-context figures in `analysis_results/context_comparison/`:
| File | Description |
|---|---|
| `fig1_interaction_lines.png` | Context × Condition interaction plot (α) |
| `fig2_dumbbell_shift.png` | Δα and ΔCCEI from Abstract → Agentic |
| `fig3_garp_distribution.png` | GARP pass rates stacked bar chart |
| `fig4_alpha_distribution.png` | Violin plots of α by context and condition |

---

## 📈 Key Results

Analysis results are pre-computed and stored in `analysis_results/`. Paper-quality figures are in `analysis_results/agentic/paper_figures_on_swaps/`:

| Figure | Description |
|---|---|
| `fig1_dumbbell_agentic_shift.png` | Shift in α and CCEI under agentic context |
| `fig2_preference_space.png` | (α, ρ) preference space across models |
| `fig3_rationality_profile.png` | Rationality profiles by model and condition |
| `fig4_alpha_violin.png` | Distribution of selfishness parameter α |

---

## 📐 Methodology Notes

- Each LLM is treated as a **"synthetic subject"** — one run_id = one simulated participant
- GARP testing uses the **Floyd-Warshall** algorithm for transitive closure of revealed preferences
- CCEI is computed via **binary search** over the Afriat efficiency parameter (tolerance = 0.001)
- CES fitting uses **nonlinear least squares** (`scipy.optimize.curve_fit`) on the demand function for π_s
- All temperatures set to **0.7** to allow stochastic variation across runs

---

## 📚 References

- Andreoni, J., & Miller, J. (2002). Giving according to GARP: An experimental test of the consistency of preferences for altruism. *Econometrica*, 70(2), 737–753.
- Afriat, S. N. (1972). Efficiency estimation of production functions. *International Economic Review*, 13(3), 568–598.
- Varian, H. R. (1982). The nonparametric approach to demand analysis. *Econometrica*, 50(4), 945–973.

---

## 📄 License

This project is for academic research purposes.
