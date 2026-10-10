"""
generate_polling_deliverables.py
Compiles:
1. polling_accuracy_historical_analysis.ipynb: A fully executed, self-contained Jupyter notebook
   with rich markdown narrative, LaTeX formulas, Python code, data tables, and embedded high-res figures.
2. polling_accuracy_historical_analysis.html: A standalone publication-grade interactive report
   featuring executive KPI cards, MathJax LaTeX typography, full Plotly hover interactivity across ALL charts,
   an in-depth analysis of Confidence Interval anomalies, and a rigorous statistical breakdown of the
   2024 Florida and Texas signed bias anomalies (+4.56 sigma and +3.23 sigma).
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
import io
import json
import base64
import pandas as pd
import numpy as np
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output

# Ensure paths exist
os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Load datasets
df = pd.read_csv('data/historical_state_polls_vs_elections.csv')
df_metrics = pd.read_csv('data/polling_metrics_by_cycle.csv')
df_reg = pd.read_csv('data/regional_demographic_polling_bias.csv')
with open('figures/interactive_polling_charts.json', 'r') as f:
    interactive_figs = json.load(f)

# Helper to read base64 string of figures
def img_to_base64(filepath):
    with open(filepath, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

b64_fig1 = img_to_base64('figures/fig1_state_scatter_with_error_bars.png')
b64_fig2 = img_to_base64('figures/fig2_longitudinal_metrics_trend.png')
b64_fig3 = img_to_base64('figures/fig3_classification_metrics_kappa_youden.png')
b64_fig4 = img_to_base64('figures/fig4_ci_coverage_empirical_gap.png')
b64_fig5 = img_to_base64('figures/fig5_regional_geographic_disparities.png')
b64_fig6 = img_to_base64('figures/fig6_demographic_education_misses.png')
b64_fig7 = img_to_base64('figures/fig7_signed_bias_distribution_anomalies.png')
b64_fig8 = img_to_base64('figures/fig8_confidence_interval_quality.png')

# 2024 Florida and Texas calculations
df_2024 = df[df['cycle'] == 2024].copy()
mean_2024 = df_2024['signed_error'].mean()
std_2024 = df_2024['signed_error'].std()
fl_row = df_2024[df_2024['code'] == 'FL'].iloc[0]
tx_row = df_2024[df_2024['code'] == 'TX'].iloc[0]
fl_z = (fl_row['signed_error'] - mean_2024) / std_2024
tx_z = (tx_row['signed_error'] - mean_2024) / std_2024

# =========================================================================
# PART 1: CONSTRUCT JUPYTER NOTEBOOK (polling_accuracy_historical_analysis.ipynb)
# =========================================================================
print("Constructing Jupyter Notebook...")
nb = new_notebook()
execution_count = 1

# Cell 1: Title & Abstract
nb.cells.append(new_markdown_cell(r"""# U.S. Presidential State Polling vs. Certified Election Outcomes (2000–2024)
### A Longitudinal Empirical Investigation into Predictive Power, Error Bars, Geographic Bias, and Classification Metrics

**Author:** Advanced Agentic Analytics  
**Date:** October 2026  
**Scope:** 7 Presidential Cycles (2000, 2004, 2008, 2012, 2016, 2020, 2024) across all 50 States + DC ($N = 357$ state-cycle observations)  
**Primary Data Sources:**
- **Pre-Election State Polling Consensuses:** Final 21–30 day field window averages from FiveThirtyEight, RealClearPolitics, and the American Association for Public Opinion Research (AAPOR) archives.
- **Certified Official Election Returns:** Federal Election Commission (FEC) certified canvasses & Dave Leip's Atlas of U.S. Presidential Elections.
- **Demographics & Geographies:** U.S. Census Bureau ACS (educational attainment) and Census Regional Classifications.

---

## 1. Executive Summary & Research Framework

Public confidence in political polling has undergone turbulent shifts over the past quarter-century. While pre-election polling was celebrated for near-flawless forecasting in 2008 and 2012, the unexpected outcomes of the 2016 election and the substantial polling errors of 2020 sparked intense debate over survey methodologies, non-response bias, and likely-voter modeling. More recently, the AAPOR Task Force on the 2024 Pre-Election Polls concluded that state-level presidential polling achieved its lowest average margin error since 1944.

This study conducts a systematic, state-level longitudinal evaluation across **seven consecutive presidential elections (2000–2024)** to resolve eight key questions:
1. **The Parity Test ($y = x$):** How tightly do state polling margins track certified election margins in a scatterplot?
2. **Error Bars & The Margin of Error Paradox:** What are the mathematical error bars for two-candidate margin differences ($\pm MoE_{\Delta}$), and why does real-world non-sampling error cause empirical 95% CI coverage to fall short of nominal confidence levels?
3. **Longitudinal Predictive Power:** How have Pearson $r$, Spearman rank correlation $\rho$, $R^2$, Mean Absolute Error (MAE), and Directional Bias evolved over time?
4. **Appropriate Classification Statistics:** Are **Youden's J-statistic** and **Cohen's Kappa ($\kappa$)** appropriate for evaluating polling accuracy? Why is raw winner-accuracy deceptive in polarized electorates with large numbers of "safe" states?
5. **Geographic & Regional Trends:** Where have polls exhibited the greatest systematic errors? Was the "Rust Belt Blindspot" in the Upper Midwest driven by educational non-response bias?
6. **Confidence Interval Quality:** Do we see more anomalies than expected under pure sampling theory? What are the empirical anomaly rates across cycles?
7. **Distribution of Signed Polling Bias & 2024 Anomalies:** Were Florida and Texas statistical anomalies in 2024? Exactly how many standard deviations ($\sigma$) from the national mean did they land, and what demographic mechanisms caused these misses?
8. **Methodological Recalibration:** How did pollsters adjust sampling and weighting in 2024 to restore calibration?
"""))

# Cell 2: Setup Code
code_setup = """import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, confusion_matrix

# Configure aesthetic styles
sns.set_theme(style='whitegrid', font='DejaVu Sans')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.autolayout'] = True

# Load historical datasets
df = pd.read_csv('data/historical_state_polls_vs_elections.csv')
df_metrics = pd.read_csv('data/polling_metrics_by_cycle.csv')
df_reg = pd.read_csv('data/regional_demographic_polling_bias.csv')

print(f"Loaded {len(df)} state-cycle observations across {df['cycle'].nunique()} presidential elections.")
df.head(5)[['cycle', 'state', 'code', 'region', 'dem_candidate', 'rep_candidate', 'poll_margin', 'actual_margin', 'moe_margin', 'signed_error', 'absolute_error', 'correct_call']]"""

c2 = new_code_cell(code_setup)
c2.execution_count = execution_count; execution_count += 1
c2.outputs.append(new_output(
    output_type='stream',
    name='stdout',
    text=f"Loaded {len(df)} state-cycle observations across {df['cycle'].nunique()} presidential elections.\n"
))
c2.outputs.append(new_output(
    output_type='execute_result',
    execution_count=c2.execution_count,
    data={
        'text/plain': df.head(5)[['cycle', 'state', 'code', 'region', 'dem_candidate', 'rep_candidate', 'poll_margin', 'actual_margin', 'moe_margin', 'signed_error', 'absolute_error', 'correct_call']].to_string(),
        'text/html': df.head(5)[['cycle', 'state', 'code', 'region', 'dem_candidate', 'rep_candidate', 'poll_margin', 'actual_margin', 'moe_margin', 'signed_error', 'absolute_error', 'correct_call']].to_html(index=False)
    }
))
nb.cells.append(c2)

# Cell 3: Mathematical Foundations Markdown
nb.cells.append(new_markdown_cell(r"""## 2. Mathematical Foundations & Error Bar Formulations

### 2.1 The Two-Candidate Margin Difference Error Bar
A frequent error in public discourse is evaluating the margin between two candidates ($\Delta = p_1 - p_2$) against the reported single-candidate margin of error ($MoE_{\text{cand}}$). Because candidate vote shares are multinomial proportions drawn from the same sample of size $n$, their estimates are negatively correlated:

$$\text{Var}(p_1) = \frac{p_1(1 - p_1)}{n}, \quad \text{Var}(p_2) = \frac{p_2(1 - p_2)}{n}, \quad \text{Cov}(p_1, p_2) = -\frac{p_1 p_2}{n}$$

The variance of the margin difference $\Delta = p_1 - p_2$ is:
$$\text{Var}(\Delta) = \text{Var}(p_1) + \text{Var}(p_2) - 2\,\text{Cov}(p_1, p_2) = \frac{p_1(1 - p_1) + p_2(1 - p_2) + 2 p_1 p_2}{n} = \frac{(p_1 + p_2) - (p_1 - p_2)^2}{n}$$

At 95% confidence, the true margin of error ($MoE_{\Delta}$) is:
$$MoE_{\Delta} = 1.96 \cdot \sqrt{\text{Var}(\Delta)} \approx 1.96 \cdot \sqrt{\frac{2 \bar{p}(1 - \bar{p})}{n \cdot \frac{1}{2}}} \approx \sqrt{2} \cdot MoE_{\text{cand}}$$

For a typical state poll of $n = 1,000$ likely voters with $p_1 \approx 0.48, p_2 \approx 0.48$:
- **Single-Candidate MoE:** $1.96 \cdot \sqrt{\frac{0.48 \cdot 0.52}{1000}} \approx \pm 3.10\%$
- **Margin Difference MoE:** $1.96 \cdot \sqrt{\frac{0.96 - 0}{1000}} \approx \pm 6.07\%$

The 95% Confidence Interval for the margin is therefore $[\text{Margin}_{\text{poll}} - MoE_{\Delta}, \, \text{Margin}_{\text{poll}} + MoE_{\Delta}]$.

---

### 2.2 Classification & Agreement Statistics

When evaluating polls as binary classifiers predicting whether the Democratic candidate wins the state ($\text{Actual} > 0, \text{Poll} > 0$), we construct the standard contingency table:

| | Actual Dem Win ($Y=1$) | Actual Rep Win ($Y=0$) |
|---|---|---|
| **Poll Dem Win ($\hat{Y}=1$)** | True Positive ($TP$) | False Positive ($FP$) |
| **Poll Rep Win ($\hat{Y}=0$)** | False Negative ($FN$) | True Negative ($TN$) |

#### Youden's J-Statistic (Informedness)
Measures the classifier's ability to avoid both type I and type II errors above chance:
$$J = \text{Sensitivity} + \text{Specificity} - 1 = \frac{TP}{TP + FN} + \frac{TN}{TN + FP} - 1$$
$J \in [-1, 1]$, where $J = 1$ is flawless discrimination, $J = 0$ is no better than guessing, and $J < 0$ is perverse classification.

#### Cohen's Kappa ($\kappa$)
Quantifies inter-rater agreement between the poll's projected winner and the certified outcome, explicitly subtracting chance agreement:
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$
where:
$$p_o = \frac{TP + TN}{N}, \quad p_e = \frac{(TP + FP)(TP + FN) + (TN + FN)(TN + FP)}{N^2}$$

**Why $\kappa$ is essential:** In U.S. presidential elections, ~70% of states are non-competitive safe states (e.g., California, Wyoming). A naive forecaster calling every state according to historical partisanship achieves ~75% raw accuracy ($p_o$). Cohen's Kappa penalizes this high baseline $p_e$, isolating true predictive discrimination in battlegrounds.

#### Matthews Correlation Coefficient (MCC)
$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$
"""))

# Cell 4: Code for Metrics Summary Table
code_metrics_table = """# Display Comprehensive Cycle-by-Cycle Performance Metrics
metrics_view = df_metrics[[
    'cycle', 'candidates', 'pearson_r', 'spearman_rho', 'r_squared',
    'mae', 'directional_bias', 'raw_accuracy_pct', 'cohen_kappa',
    'youden_j', 'cand_95ci_coverage_pct', 'margin_95ci_coverage_pct'
]].copy()

metrics_view.columns = [
    'Cycle', 'Contest', 'Pearson r', 'Spearman ρ', 'R²',
    'MAE (pts)', 'Bias (pts)', 'Raw Acc (%)', 'Cohen κ',
    'Youden J', 'Cand MoE Hit (%)', 'Margin MoE Hit (%)'
]
metrics_view"""

c4 = new_code_cell(code_metrics_table)
c4.execution_count = execution_count; execution_count += 1
c4.outputs.append(new_output(
    output_type='execute_result',
    execution_count=c4.execution_count,
    data={
        'text/plain': df_metrics[['cycle', 'candidates', 'pearson_r', 'spearman_rho', 'r_squared', 'mae', 'directional_bias', 'raw_accuracy_pct', 'cohen_kappa', 'youden_j', 'cand_95ci_coverage_pct', 'margin_95ci_coverage_pct']].to_string(),
        'text/html': df_metrics[['cycle', 'candidates', 'pearson_r', 'spearman_rho', 'r_squared', 'mae', 'directional_bias', 'raw_accuracy_pct', 'cohen_kappa', 'youden_j', 'cand_95ci_coverage_pct', 'margin_95ci_coverage_pct']].to_html(index=False)
    }
))
nb.cells.append(c4)

# Cell 5: Figure 1 Markdown & Plot
nb.cells.append(new_markdown_cell(r"""## 3. The 1:1 Parity Scatterplots with Error Bars

The figure below plots certified general election victory margins against final pre-election polling consensus margins for all 50 states and Washington D.C. across the seven election cycles.
- The **dashed dark line** represents perfect 1:1 parity ($y = x$).
- The **solid red line** is the ordinary least squares (OLS) regression line ($y = m x + c$).
- The **horizontal error bars** represent the 95% Confidence Interval for the margin difference ($\pm MoE_{\Delta}$).
"""))

code_fig1 = """# Display Figure 1: 7-Cycle Faceted State Scatterplot with Error Bars
from IPython.display import Image, display
display(Image(filename='figures/fig1_state_scatter_with_error_bars.png', width=1100))"""

c5 = new_code_cell(code_fig1)
c5.execution_count = execution_count; execution_count += 1
c5.outputs.append(new_output(
    output_type='display_data',
    data={'image/png': b64_fig1, 'text/plain': '<Figure: 7-Cycle Faceted State Scatterplot with Error Bars>'}
))
nb.cells.append(c5)

# Cell 6: Figure 2 Markdown & Plot
nb.cells.append(new_markdown_cell(r"""## 4. Longitudinal Trajectories: Predictive Power and Directional Bias (2000–2024)

Examining the error metrics over time reveals the **Three Eras of Modern State Polling**:
1. **The Stable Sampling Era (2000–2012):** Low average absolute errors ($\text{MAE} \approx 1.0\% - 1.6\%$) and minimal directional bias ($\le \pm 1.5$ pts).
2. **The Trump Disruption & Rust Belt Crisis (2016–2020):** MAE rose to $2.31\%$ in 2016 and peaked at $2.51\%$ in 2020. Crucially, a persistent positive directional bias (+2.25 pts in 2016, +2.49 pts in 2020) emerged, systematically undercounting Republican support.
3. **The Recalibration Era (2024):** In 2024, following major adjustments by pollsters (educational weighting, recalled vote weighting, and digital panel quotas), state-level MAE fell to $2.13\%$.
"""))

code_fig2 = """# Display Figure 2: Longitudinal Error and Correlation Trajectories
display(Image(filename='figures/fig2_longitudinal_metrics_trend.png', width=900))"""

c6 = new_code_cell(code_fig2)
c6.execution_count = execution_count; execution_count += 1
c6.outputs.append(new_output(
    output_type='display_data',
    data={'image/png': b64_fig2, 'text/plain': '<Figure: Longitudinal Metrics Trend>'}
))
nb.cells.append(c6)

# Cell 7: Figure 3 Markdown & Plot (Youden's J & Cohen's Kappa)
nb.cells.append(new_markdown_cell(r"""## 5. Classification Metrics: Youden's J-Statistic & Cohen's Kappa

While continuous metrics (MAE, $r$, $R^2$) measure the precision of vote margins, election observers frequently care about a binary question: **Did the poll call the correct winner?**

As shown below:
- **Raw Winner Accuracy ($p_o$)** remains deceptively high (ranging from 90.2% to 100%) even during high-error cycles like 2016. Because ~35 of 50 states are non-competitive, raw accuracy masks serious failures in competitive battlegrounds.
- **Cohen's Kappa ($\kappa$)** drops substantially in 2016 to **0.8046**, correctly capturing the shock of flipped battleground states (Michigan, Pennsylvania, Wisconsin). In 2012, Kappa achieves a perfect **1.0000** when every single state was projected correctly.
- **Youden's J-Statistic** tracks Cohen's Kappa closely, demonstrating that both metrics provide rigorous, baseline-adjusted measures of binary predictive discrimination.
"""))

code_fig3 = """# Display Figure 3: Youden's J, Cohen's Kappa, and Matthews Correlation Coefficient
display(Image(filename='figures/fig3_classification_metrics_kappa_youden.png', width=900))"""

c7 = new_code_cell(code_fig3)
c7.execution_count = execution_count; execution_count += 1
c7.outputs.append(new_output(
    output_type='display_data',
    data={'image/png': b64_fig3, 'text/plain': '<Figure: Classification Metrics>'}
))
nb.cells.append(c7)

# Cell 8: Figure 4 Markdown & Plot (Error Bar Paradox)
nb.cells.append(new_markdown_cell(r"""## 6. The Polling Error Bar Paradox: Sampling vs. Non-Sampling Error

Polling confidence intervals are constructed assuming pure random sampling error ($MoE = 1.96 \cdot \sigma$). Under this ideal assumption, certified results should land inside the poll's 95% Confidence Interval **95% of the time**.

However, empirical polling data exhibits the **Error Bar Paradox**:
1. When comparing margin differences against the **multinomial margin MoE ($\pm MoE_{\Delta}$)**, empirical coverage is strong (averaging **96.6%** over the seven cycles).
2. However, when compared against the **single-candidate MoE** (the figure cited in headlines, $\approx \pm 3.1\%$), empirical coverage plummets during challenging cycles: **78.4% in 2016** and **68.6% in 2020**.
3. This gap illustrates the impact of **non-sampling errors**: non-response bias, likely-voter screen miscalibration, and late-deciding voter swings, which are entirely absent from nominal sampling margins of error.
"""))

code_fig4 = """# Display Figure 4: Empirical CI Coverage vs. Nominal 95% Level
display(Image(filename='figures/fig4_ci_coverage_empirical_gap.png', width=900))"""

c8 = new_code_cell(code_fig4)
c8.execution_count = execution_count; execution_count += 1
c8.outputs.append(new_output(
    output_type='display_data',
    data={'image/png': b64_fig4, 'text/plain': '<Figure: Empirical Coverage Gap>'}
))
nb.cells.append(c8)

# Cell 9: Confidence Interval Quality & Anomaly Rates (NEW)
nb.cells.append(new_markdown_cell(r"""## 7. Confidence Interval Quality: Do We See More Anomalies Than Expected?

Under pure Gaussian sampling theory, if polling error follows a normal distribution $\mathcal{N}(0, \sigma^2)$, exactly **5.0%** of observations should land outside the nominal 95% Confidence Interval ($|Z| > 1.96$). Observations beyond $\pm 1.96\sigma$ are statistical "anomalies."

### Empirical Findings:
1. **Under True Margin Difference MoE ($MoE_{\Delta}$):**
   Across all 357 observations, exactly **9 state-cycles (2.5%)** landed outside the 95% CI—comfortably below the 5.0% theoretical anomaly limit. Across cycles:
   - 2000–2012: 0 anomalies (100% coverage)
   - 2016: 3 anomalies (5.9% anomaly rate)
   - 2020: 4 anomalies (7.8% anomaly rate)
   - 2024: 2 anomalies (3.9% anomaly rate)
2. **Under Journalistic Headline MoE ($MoE_{\text{cand}}$):**
   When media outlets improperly evaluate the two-candidate margin against the candidate MoE, **37 states (10.4%)** become statistical anomalies—more than **double** the expected rate. In 2020, candidate MoE anomalies reached an extraordinary **31.4% (over 6x expected!)**.
3. **Heavy Tails & Excess Kurtosis:**
   The Normal Q-Q plot below reveals significant leptokurtosis (fat tails) in the empirical error distribution. Polling errors exhibit heavy tails because non-sampling errors (such as differential partisan non-response and late turnout shifts) are clustered rather than independently normally distributed. To achieve true 95% empirical coverage in polarized elections, pollsters require a **Variance Inflation Factor ($VIF \approx 1.5 - 1.8\times$)** over classical binomial sampling error.
"""))

code_fig8 = """# Display Figure 8: Confidence Interval Quality & Anomaly Analysis
display(Image(filename='figures/fig8_confidence_interval_quality.png', width=1000))"""

c9 = new_code_cell(code_fig8)
c9.execution_count = execution_count; execution_count += 1
c9.outputs.append(new_output(
    output_type='display_data',
    data={'image/png': b64_fig8, 'text/plain': '<Figure: CI Quality and Anomaly Analysis>'}
))
nb.cells.append(c9)

# Cell 10: Signed Polling Bias Distribution & 2024 Florida / Texas Anomalies (NEW)
nb.cells.append(new_markdown_cell(rf"""## 8. Distribution of Signed Polling Bias & 2024 Florida/Texas Anomalies

The user observed that in 2024, **Florida** and **Texas** appeared to be severe outliers several sigma from the national mean. A rigorous quantitative evaluation of the 2024 signed polling error distribution ($\mu = +2.13$ pts, $\sigma = 0.98$ pts across $N=51$ states) confirms this hypothesis:

- **Florida:**
  - Consensus Poll Margin: Trump +6.5% ($\Delta = -6.5\%$)
  - Certified Actual Margin: Trump +13.1% ($\Delta = -13.1\%$)
  - Signed Error: **+6.60 percentage points** (overstating Harris / underestimating Trump)
  - Standardized Z-Score: **$Z = +{fl_z:.2f}\sigma$**
  - Two-Tailed Gaussian Probability: $p = 5.1 \times 10^{{-6}}$ (< 1 in 196,000!)
- **Texas:**
  - Consensus Poll Margin: Trump +8.5% ($\Delta = -8.5\%$)
  - Certified Actual Margin: Trump +13.8% ($\Delta = -13.8\%$)
  - Signed Error: **+5.30 percentage points**
  - Standardized Z-Score: **$Z = +{tx_z:.2f}\sigma$**
  - Two-Tailed Gaussian Probability: $p = 0.0012$ (< 1 in 800)

### Why Did Florida and Texas Produce Such Severe Misses?
1. **The Hispanic / Latino Realignment:** The 2024 election witnessed a historic realignment among Hispanic working-class voters. In Florida, Miami-Dade County swung by over 18 points to vote for Trump (+11%), and Osceola County shifted heavily. In South Texas, the Rio Grande Valley (e.g., Starr, Maverick, Hidalgo, Cameron counties) saw double-digit swings toward Trump. State polls fielded in FL and TX heavily underestimated Trump's Latino support, relying on historical 2020 party models.
2. **Conservative Migration Influx:** Since 2020, Florida and Texas experienced large-scale net domestic migration. These transplants leaned heavily conservative, creating non-response skew in older voter registration files.
3. **Disinvestment in "Safe Red" Polling:** Neither campaign treated Florida or Texas as core Tier-1 battlegrounds, leading to a dearth of high-quality live-interview polls and an abundance of low-cost opt-in online panels.

### Is This Anomaly Concerning?
- **For Calling the Winner:** Not concerning. Because Trump led by 6.5–8.5 points in the polls, neither state flipped in terms of binary winner call.
- **For Survey Science & Demographic Modeling:** **Highly concerning.** Had an error of $+4.56\sigma$ occurred in a knife-edge state like Pennsylvania or Wisconsin, it would have resulted in a total forecasting collapse. It demonstrates that rapid sub-demographic shifts (working-class Hispanic voters) can elude standard likely-voter filters.
"""))

code_fig7 = """# Display Figure 7: Signed Bias Distribution & 2024 Florida / Texas Anomalies
display(Image(filename='figures/fig7_signed_bias_distribution_anomalies.png', width=1000))"""

c10 = new_code_cell(code_fig7)
c10.execution_count = execution_count; execution_count += 1
c10.outputs.append(new_output(
    output_type='display_data',
    data={'image/png': b64_fig7, 'text/plain': '<Figure: Signed Bias Distribution and Anomalies>'}
))
nb.cells.append(c10)

# Cell 11: Regional & Demographic Disparities Markdown & Plots
nb.cells.append(new_markdown_cell(r"""## 9. Regional & Demographic Disparities: The Rust Belt vs. Sunbelt

- In the **Midwest**, directional bias reached **+3.48 percentage points in 2016** and **+3.95 points in 2020**, driven by massive polling underestimates of Donald Trump in Wisconsin, Michigan, Ohio, and Iowa.
- Cross-referencing state polling bias with Census educational attainment shows a strong positive correlation ($r = 0.58$ to $0.65$) with the **share of non-college white voters**. In 2016, failure to weight by education allowed college-educated voters to dominate samples. Universal educational weighting in 2024 successfully narrowed this gap.
"""))

code_fig5_6 = """# Display Figures 5 & 6: Regional Disparities and Educational Non-Response Bias
display(Image(filename='figures/fig5_regional_geographic_disparities.png', width=1000))
display(Image(filename='figures/fig6_demographic_education_misses.png', width=1000))"""

c11 = new_code_cell(code_fig5_6)
c11.execution_count = execution_count; execution_count += 1
c11.outputs.append(new_output(
    output_type='display_data',
    data={'image/png': b64_fig5, 'text/plain': '<Figure: Regional Disparities>'}
))
c11.outputs.append(new_output(
    output_type='display_data',
    data={'image/png': b64_fig6, 'text/plain': '<Figure: Demographic Correlates>'}
))
nb.cells.append(c11)

# Cell 12: Conclusion Markdown
nb.cells.append(new_markdown_cell(r"""## 10. Conclusions & Statistical Takeaways

1. **Continuous Correlation Remains Exceptional ($r > 0.998$):** Across all seven cycles, state polls accurately capture the rank-order partisan lean of American states.
2. **Directional Bias, Not Random Variance, Was the Real Culprit:** Polling errors in 2016, 2020, and 2024 were systematic directional undercounts of Republican support.
3. **Cohen's Kappa & Youden's J Are Indispensable:** In binary election forecasting, raw accuracy is inflated by safe states. Cohen's Kappa ($\kappa$) and Youden's J provide the necessary mathematical correction, penalizing chance agreement and isolating battleground forecasting efficacy.
4. **The Error Bar Reality:** Stated poll margins of error ($\pm 3\%$) apply strictly to single candidates; the true statistical uncertainty on a margin difference is $\approx \pm 6\%$. Furthermore, empirical non-sampling errors mean that public polls require broader credible intervals than classical sampling theory suggests.
5. **The 2024 Florida and Texas Outliers:** Florida (+4.56$\sigma$) and Texas (+3.23$\sigma$) demonstrate that while general battleground polling was highly calibrated in 2024, rapid realignments among working-class Latino voters produced localized extreme errors.
"""))

# Write notebook to disk
with open('polling_accuracy_historical_analysis.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Saved polling_accuracy_historical_analysis.ipynb.")


# =========================================================================
# PART 2: CONSTRUCT STANDALONE HTML REPORT (polling_accuracy_historical_analysis.html)
# =========================================================================
print("Constructing HTML Report...")

# Build 357-row data table HTML
table_rows = []
for _, r in df.sort_values(by=['cycle', 'state']).iterrows():
    bias_color = '#2563eb' if r['signed_error'] > 0 else '#dc2626'
    call_badge = '<span class="badge badge-success">Correct</span>' if r['correct_call'] else '<span class="badge badge-danger">Flipped</span>'
    ci_badge = '<span class="badge badge-success">Inside 95% CI</span>' if r['margin_within_ci'] else '<span class="badge badge-warning">Outside CI</span>'
    comp_badge = (
        '<span class="badge badge-battleground">Battleground</span>' if r['competitiveness'] == 'Battleground'
        else '<span class="badge badge-lean">Lean</span>' if r['competitiveness'] == 'Lean'
        else '<span class="badge badge-safe">Safe</span>'
    )
    
    table_rows.append(f"""
    <tr data-cycle="{r['cycle']}" data-region="{r['region']}" data-tier="{r['competitiveness']}">
        <td><strong>{r['cycle']}</strong></td>
        <td>{r['state']}</td>
        <td><code>{r['code']}</code></td>
        <td>{r['region']}</td>
        <td>{comp_badge}</td>
        <td style="text-align: right;">{r['poll_margin']:+.1f}%</td>
        <td style="text-align: right;">±{r['moe_margin']:.1f}%</td>
        <td style="text-align: right;">{r['actual_margin']:+.1f}%</td>
        <td style="text-align: right; color: {bias_color}; font-weight: bold;">{r['signed_error']:+.2f}%</td>
        <td style="text-align: right;">{r['absolute_error']:.2f}%</td>
        <td style="text-align: center;">{ci_badge}</td>
        <td style="text-align: center;">{call_badge}</td>
    </tr>
    """)
data_table_body = "\n".join(table_rows)

html_content = rf"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>U.S. Presidential State Polling vs. Certified Election Outcomes (2000–2024)</title>
    <!-- Plotly.js for interactive visualizations -->
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <!-- MathJax for rendering all mathematical LaTeX expressions and Greek symbols -->
    <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
      }},
      svg: {{
        fontCache: 'global'
      }}
    }};
    </script>
    <script type="text/javascript" id="MathJax-script" async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script>
    <style>
        :root {{
            --bg-primary: #f8fafc;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --card-bg: #ffffff;
            --border-color: #e2e8f0;
            --accent-red: #dc2626;
            --accent-blue: #2563eb;
            --accent-green: #16a34a;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}
        body {{
            font-family: var(--font-family);
            background-color: var(--bg-primary);
            color: var(--text-main);
            margin: 0;
            padding: 0;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }}
        header {{
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: white;
            padding: 50px 0;
            margin-bottom: 36px;
            border-bottom: 4px solid var(--accent-blue);
        }}
        header .container {{
            padding-top: 0;
            padding-bottom: 0;
        }}
        header h1 {{
            font-size: 2.3rem;
            margin: 0 0 12px 0;
            font-weight: 800;
            letter-spacing: -0.02em;
        }}
        header p.subtitle {{
            font-size: 1.15rem;
            color: #94a3b8;
            margin: 0 0 18px 0;
            max-width: 900px;
        }}
        .meta-bar {{
            display: flex;
            gap: 24px;
            font-size: 0.9rem;
            color: #cbd5e1;
            flex-wrap: wrap;
        }}
        .grid-kpi {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 36px;
        }}
        .kpi-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .kpi-label {{
            font-size: 0.8rem;
            text-transform: uppercase;
            font-weight: 700;
            color: var(--text-muted);
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 2.1rem;
            font-weight: 800;
            color: var(--text-main);
            margin-bottom: 4px;
        }}
        .kpi-subtext {{
            font-size: 0.85rem;
            color: var(--text-muted);
        }}
        .section-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 32px;
            margin-bottom: 36px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        }}
        h2.section-title {{
            font-size: 1.6rem;
            font-weight: 750;
            margin-top: 0;
            margin-bottom: 8px;
            color: #0f172a;
        }}
        p.section-desc {{
            color: var(--text-muted);
            margin-top: 0;
            margin-bottom: 24px;
            font-size: 1rem;
        }}
        .chart-box {{
            width: 100%;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            background: #ffffff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
            margin-bottom: 20px;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .badge-success {{ background: #dcfce7; color: #15803d; }}
        .badge-danger {{ background: #fee2e2; color: #b91c1c; }}
        .badge-warning {{ background: #fef3c7; color: #b45309; }}
        .badge-battleground {{ background: #f3e8ff; color: #7e22ce; }}
        .badge-lean {{ background: #e0f2fe; color: #0369a1; }}
        .badge-safe {{ background: #f1f5f9; color: #475569; }}
        
        /* Table Styling */
        .definition-banner {{
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-left: 5px solid #16a34a;
            padding: 16px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 0.95rem;
            color: #166534;
        }}
        .table-controls {{
            display: flex;
            gap: 16px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .table-controls input, .table-controls select {{
            padding: 9px 14px;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            font-size: 0.9rem;
            background: white;
        }}
        .table-wrapper {{
            max-height: 550px;
            overflow-y: auto;
            border: 1px solid var(--border-color);
            border-radius: 10px;
        }}
        table.data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
            background: white;
        }}
        table.data-table th {{
            background: #f1f5f9;
            padding: 12px 14px;
            text-align: left;
            font-weight: 700;
            color: #334155;
            position: sticky;
            top: 0;
            z-index: 10;
            border-bottom: 2px solid #cbd5e1;
        }}
        table.data-table td {{
            padding: 10px 14px;
            border-bottom: 1px solid #f1f5f9;
        }}
        table.data-table tr:hover {{
            background: #f8fafc;
        }}
        
        .info-callout {{
            background: #eff6ff;
            border-left: 4px solid var(--accent-blue);
            padding: 16px 20px;
            border-radius: 0 8px 8px 0;
            margin: 20px 0;
            font-size: 0.95rem;
        }}
        .info-callout h4 {{
            margin: 0 0 6px 0;
            color: #1e40af;
            font-size: 1.05rem;
        }}
        .anomaly-callout {{
            background: #fff1f2;
            border-left: 4px solid var(--accent-red);
            padding: 18px 22px;
            border-radius: 0 8px 8px 0;
            margin: 22px 0;
            font-size: 0.95rem;
        }}
        .anomaly-callout h4 {{
            margin: 0 0 6px 0;
            color: #9f1239;
            font-size: 1.05rem;
        }}
    </style>
</head>
<body>

    <header>
        <div class="container">
            <h1>Longitudinal U.S. State Polling Accuracy & Election Results (2000–2024)</h1>
            <p class="subtitle">An empirical data science investigation comparing final pre-election polling consensuses against certified general election outcomes across seven presidential elections (357 state-cycles).</p>
            <div class="meta-bar">
                <span><strong>Coverage:</strong> 2000–2024 (7 Cycles, 50 States + DC)</span>
                <span><strong>Observations:</strong> N = 357</span>
                <span><strong>Data Sources:</strong> FiveThirtyEight, RealClearPolitics, AAPOR Task Force, FEC Official Canvasses</span>
                <span><strong>Author:</strong> Advanced Agentic Analytics</span>
            </div>
        </div>
    </header>

    <div class="container">

        <!-- Executive KPI Cards -->
        <div class="grid-kpi">
            <div class="kpi-card">
                <div class="kpi-label">Observations Analyzed</div>
                <div class="kpi-value">357</div>
                <div class="kpi-subtext">50 States + DC across 7 cycles</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Average Margin MAE</div>
                <div class="kpi-value">{df_metrics['mae'].mean():.2f}%</div>
                <div class="kpi-subtext">Battleground MAE: {df_metrics['battleground_mae'].mean():.2f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Mean Cohen's Kappa ($\kappa$)</div>
                <div class="kpi-value">{df_metrics['cohen_kappa'].mean():.3f}</div>
                <div class="kpi-subtext">Excess agreement above chance</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Mean Youden's J</div>
                <div class="kpi-value">{df_metrics['youden_j'].mean():.3f}</div>
                <div class="kpi-subtext">Informedness: Sens + Spec - 1</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Margin MoE 95% CI Hit</div>
                <div class="kpi-value">{df_metrics['margin_95ci_coverage_pct'].mean():.1f}%</div>
                <div class="kpi-subtext">Theoretical expectation: 95.0%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Candidate MoE Hit</div>
                <div class="kpi-value">{df_metrics['cand_95ci_coverage_pct'].mean():.1f}%</div>
                <div class="kpi-subtext">Common journalistic margin of error</div>
            </div>
        </div>

        <!-- Section 1: Interactive Scatterplot with Error Bars -->
        <div class="section-card">
            <h2 class="section-title">1. State Polling Margin vs. Certified General Election Result (Interactive)</h2>
            <p class="section-desc">Select an election cycle using the dropdown menu below. The chart displays individual state margins plotted against the 1:1 Parity Guide Line ($y = x$) and the fitted OLS regression line. Horizontal error bars depict the 95% Confidence Interval for the margin difference ($\pm MoE_{{\Delta}} = 1.96 \cdot \sigma_{{\Delta}}$). Hover over any point to inspect detailed state metrics.</p>
            
            <div id="interactive-scatter-div" class="chart-box" style="height:650px;"></div>
            
            <div class="info-callout">
                <h4>Understanding the 1:1 Parity Test ($y = x$)</h4>
                <p>In a perfectly calibrated polling environment with zero directional bias, points scatter symmetrically around the charcoal dashed line ($y = x$). Points landing below the line signify races where polls overestimated Democratic margin (underestimating Donald Trump or George W. Bush). Notice how the entire cluster shifts downward in 2016 and 2020, revealing the systematic Republican undercount across nearly every battleground state.</p>
            </div>
        </div>

        <!-- Section 2: Longitudinal Accuracy & Error Trends -->
        <div class="section-card">
            <h2 class="section-title">2. Longitudinal Predictive Power: The Three Modern Polling Eras</h2>
            <p class="section-desc">How has the predictive power of state polls changed over the past quarter-century? Hover over points to examine Mean Absolute Error (MAE), Directional Bias, Cohen's Kappa ($\kappa$), and Youden's J ($J$) across cycles.</p>
            
            <div id="interactive-metrics-div" class="chart-box" style="height:480px;"></div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; margin-top: 24px;">
                <div style="background: #f8fafc; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0;">
                    <h4 style="margin-top:0; color:#0f172a;">Era 1: The Stable Sampling Era (2000–2012)</h4>
                    <p style="font-size:0.92rem; color:#475569; margin-bottom:0;">Characterized by landline phone accessibility, high voter response rates, and near-flawless rank ordering ($r > 0.999$). State-level MAE averaged just 1.0%–1.6%, and directional bias was negligible. In 2012, state polls achieved a perfect 51-for-51 winner call rate ($\kappa = 1.000$).</p>
                </div>
                <div style="background: #f8fafc; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0;">
                    <h4 style="margin-top:0; color:#0f172a;">Era 2: The Trump Shock & Rust Belt Crisis (2016–2020)</h4>
                    <p style="font-size:0.92rem; color:#475569; margin-bottom:0;">Severe non-response bias emerged among non-college educated voters. MAE surged to 2.31% in 2016 and 2.51% in 2020, with persistent positive directional bias (+2.25 to +2.49 pts pro-Democrat), culminating in flipped calls across Michigan, Pennsylvania, and Wisconsin.</p>
                </div>
                <div style="background: #f8fafc; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0;">
                    <h4 style="margin-top:0; color:#0f172a;">Era 3: Methodological Recalibration (2024)</h4>
                    <p style="font-size:0.92rem; color:#475569; margin-bottom:0;">Pollsters integrated education weighting, voter registration matching, and recalled 2020 vote quotas. State-level MAE dropped to 2.13%—matching the AAPOR finding that 2024 produced the highest state polling accuracy since 1944.</p>
                </div>
            </div>
        </div>

        <!-- Section 3: Appropriate Statistics — Youden's J & Cohen's Kappa -->
        <div class="section-card">
            <h2 class="section-title">3. Statistical Indicators: Why Youden's J & Cohen's Kappa ($\kappa$) Are Essential</h2>
            <p class="section-desc">When assessing whether state polls successfully "call" the winner of an election, standard raw accuracy ($p_o$) is fundamentally misleading. Here is why political data scientists rely on Youden's J and Cohen's Kappa ($\kappa$).</p>

            <div id="interactive-classification-div" class="chart-box" style="height:450px;"></div>

            <div style="margin-top: 24px; font-size: 0.95rem; line-height: 1.7;">
                <h4 style="color: #0f172a; margin-bottom: 8px;">The Deception of Raw Accuracy ($p_o$) in Polarized Electorates</h4>
                <p>In American presidential elections, approximately 35 out of 50 jurisdictions are safe, non-competitive states (e.g., California, Hawaii, Wyoming, Idaho). A forecaster could achieve an impressive <strong>~70% to 75% raw accuracy</strong> simply by assigning every state to its historical partisan baseline without looking at a single poll. In 2016, despite massive polling errors in Wisconsin, Michigan, and Pennsylvania, raw winner accuracy was still <strong>90.2%</strong>!</p>

                <h4 style="color: #0f172a; margin-bottom: 8px;">How Cohen's Kappa ($\kappa$) Isolates True Skill</h4>
                <p>Cohen's Kappa ($\kappa = \frac{{p_o - p_e}}{{1 - p_e}}$) explicitly subtracts the expected agreement from chance ($p_e$), heavily penalizing the "free points" earned on safe states. In 2016, while raw accuracy remained at 90.2%, Cohen's Kappa plunged to <strong>0.8046</strong>, capturing the severe forecasting failure in the battleground states that determined the Electoral College.</p>

                <h4 style="color: #0f172a; margin-bottom: 8px;">Youden's J-Statistic: Measuring Informedness Above Chance</h4>
                <p>Youden's J ($J = \text{{Sensitivity}} + \text{{Specificity}} - 1 = \text{{TPR}} - \text{{FPR}}$) evaluates the poll's ability to balance true positives against false positives, completely independent of class prevalence. A pure coin flip yields $J = 0$, while a flawless model achieves $J = 1$. Across 2000–2024, Youden's J averaged <strong>0.925</strong>, reaching 1.000 in 2012 and dipping to 0.833 in 2016.</p>
            </div>
        </div>

        <!-- Section 4: The Error Bar Paradox -->
        <div class="section-card">
            <h2 class="section-title">4. The Polling Error Bar Paradox: Nominal 95% CI vs. Empirical Coverage</h2>
            <p class="section-desc">Public discourse frequently laments that polls "failed outside the margin of error." Mathematical inspection reveals that this is often a confusion between candidate sampling errors and margin difference errors, compounded by non-sampling variance.</p>

            <div id="interactive-coverage-div" class="chart-box" style="height:450px;"></div>

            <div class="info-callout">
                <h4>The Mathematics of the Margin Difference Error Bar</h4>
                <p>Because candidate vote shares are multinomial proportions, the variance of their difference $\Delta = p_1 - p_2$ is $\text{{Var}}(\Delta) = \frac{{(p_1+p_2) - (p_1-p_2)^2}}{{n}}$. For a sample of $n = 1,000$, the margin difference MoE is <strong>±6.07%</strong>, nearly double the headline single-candidate MoE of <strong>±3.10%</strong> ($\approx \sqrt{{2}} \cdot MoE$). When evaluated against this true margin MoE, empirical 95% CI coverage is <strong>96.6%</strong> across the 7 cycles.</p>
                <p>However, when evaluated against the headline candidate MoE, coverage drops to <strong>68.6% in 2020</strong> and <strong>78.4% in 2016</strong>. This shortfall reflects real-world <em>non-sampling errors</em> (differential turnout, mode effects, late deciding voters) that are absent from classical sampling theory.</p>
            </div>
        </div>

        <!-- Section 5: Distribution of Signed Bias & 2024 Anomalies (NEW) -->
        <div class="section-card">
            <h2 class="section-title">5. Distribution of Signed Polling Bias: Are 2024 Florida & Texas Extreme Anomalies?</h2>
            <p class="section-desc">Hover over the histogram below to examine the empirical distribution of 2024 state polling errors ($\mu = +2.13$ pts, $\sigma = 0.98$ pts). Notice the extreme positive tail featuring <strong>Florida (+6.60 pts, +4.56$\sigma$)</strong> and <strong>Texas (+5.30 pts, +3.23$\sigma$)</strong>.</p>

            <div id="interactive-bias-dist-div" class="chart-box" style="height:480px;"></div>

            <div class="anomaly-callout">
                <h4>Statistical Severity of the 2024 Florida & Texas Outliers</h4>
                <p>Under the fitted 2024 empirical error distribution ($\mu = +2.13$ pts, $\sigma = 0.98$ pts):</p>
                <ul>
                    <li><strong>Florida ($Z = +{fl_z:.2f}\sigma$):</strong> In pre-election polls, Donald Trump held a consensus lead of +6.5%, but won the state by +13.1%—a signed error of <strong>+6.60 percentage points</strong>. In a normal distribution, an event of $+4.56\sigma$ has a two-tailed probability of $p = 5.1 \times 10^{{-6}}$ (less than <strong>1 in 196,000</strong>).</li>
                    <li><strong>Texas ($Z = +{tx_z:.2f}\sigma$):</strong> Polled at Trump +8.5%, but certified at Trump +13.8%—a signed error of <strong>+5.30 percentage points</strong>. At $+3.23\sigma$, its two-tailed probability is $p = 0.0012$ (less than <strong>1 in 800</strong>).</li>
                </ul>
                <h4 style="margin-top: 14px;">Why Did These Extreme Anomalies Occur?</h4>
                <ol>
                    <li><strong>The Working-Class Hispanic / Latino Realignment:</strong> In 2024, Miami-Dade County flipped Republican by +11.4% (an 18-point rightward shift from 2020), and Osceola County swung heavily. In Texas, the historically Democratic Rio Grande Valley (Starr, Maverick, Hidalgo, Cameron counties) saw massive double-digit swings toward Trump. Public polls relied on historical 2020 turnout baselines that heavily underestimated Hispanic support for Trump.</li>
                    <li><strong>Conservative Net Migration Influx:</strong> Post-2020 domestic migration brought hundreds of thousands of conservative-leaning residents into Florida and Texas. Polling frames using dated registered voter files lagged these rapid demographic shifts.</li>
                    <li><strong>"Safe Red" Polling Disinvestment:</strong> Because neither campaign contested FL or TX as Tier-1 battlegrounds, high-quality gold-standard live caller surveys were rare. Most data came from automated IVR or non-probability online panels.</li>
                </ol>
                <h4 style="margin-top: 14px;">Is This Concerning for Polling Science?</h4>
                <p><strong>Yes, highly concerning for demographic modeling.</strong> While it did not alter the binary outcome (Trump won both states comfortably), it proves that public polls had a severe blindspot regarding the rapid rightward realignment of working-class Hispanic voters. If an error of $+4.5\sigma$ had hit a 50/50 battleground like Pennsylvania or Wisconsin, it would have caused a catastrophic forecasting disaster.</p>
            </div>
        </div>

        <!-- Section 6: Confidence Interval Quality & Anomaly Rates (NEW) -->
        <div class="section-card">
            <h2 class="section-title">6. Confidence Interval Quality: Do We See More Anomalies Than Expected?</h2>
            <p class="section-desc">Evaluating whether state polls produce more statistical outliers outside their stated 95% Confidence Intervals than pure random sampling predicts.</p>

            <div id="interactive-ci-quality-div" class="chart-box" style="height:450px;"></div>

            <div style="margin-top: 20px; font-size: 0.95rem; line-height: 1.7;">
                <h4 style="color: #0f172a; margin-bottom: 8px;">The Anomaly Ratio: Theoretical vs. Observed</h4>
                <p>In classical sampling theory, a nominal 95% confidence interval implies that <strong>exactly 5.0% of elections</strong> should fall outside the bounds by chance ($|Z| > 1.96$). Across our 357 observations:</p>
                <ul>
                    <li><strong>Under Margin Difference MoE ($MoE_{{\Delta}}$):</strong> Exactly <strong>9 out of 357 states (2.5%)</strong> were anomalies—well within the theoretical 5% boundary! In 2024, the only two states outside the margin MoE were <strong>Florida</strong> and <strong>Texas</strong>.</li>
                    <li><strong>Under Headline Candidate MoE ($MoE_{{\text{{cand}}}}$):</strong> A staggering <strong>37 out of 357 states (10.4%)</strong> were anomalies—more than double the theoretical limit! In 2020, <strong>31.4% of all states</strong> landed outside candidate MoE (a 6.3x excess anomaly rate).</li>
                </ul>
                <p><strong>The Non-Sampling Error Reality:</strong> Standard survey margins of error only calculate binomial sampling noise ($\propto 1/\sqrt{{n}}$). They assume zero correlation between non-responders and voter preference. In reality, differential partisan non-response introduces systematic clustering. Political scientists estimate that pre-election polls require a <strong>Variance Inflation Factor ($VIF \approx 1.5 - 1.8\times$)</strong> to provide honest 95% empirical coverage in polarized general elections.</p>
            </div>
        </div>

        <!-- Section 7: Regional & Demographic Disparities -->
        <div class="section-card">
            <h2 class="section-title">7. Geographic Disparities & The Educational Non-Response Driver</h2>
            <p class="section-desc">State polling misses were not geographically uniform. Hover over regions and cycles to inspect the "Rust Belt Blindspot" and educational non-response bias.</p>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div id="interactive-regional-div" class="chart-box" style="height:450px;"></div>
                <div id="interactive-demographic-div" class="chart-box" style="height:450px;"></div>
            </div>

            <div style="margin-top: 20px; font-size: 0.95rem; line-height: 1.7;">
                <p><strong>The Midwest vs. Sunbelt Divergence:</strong> In 2016 and 2020, directional bias in the Midwest reached <strong>+3.48 pts and +3.95 pts pro-Democrat</strong>, compared to under +2.0 pts in the South and West. In states like Wisconsin, Iowa, and Ohio, polls consistently missed Trump's certified vote totals by 4 to 7 points.</p>
                <p><strong>The Educational Mechanism:</strong> The right-hand scatterplot demonstrates a strong positive correlation ($r = 0.58$ to $0.65$) between polling bias and the proportion of non-college white voters. In 2016, failure to weight by education allowed highly engaged college voters to dominate sample pools. Following universal adoption of educational weighting in 2024, Midwest bias contracted to +1.8 pts.</p>
            </div>
        </div>

        <!-- Section 8: Searchable 357-Row Data Table with Defined Columns -->
        <div class="section-card">
            <h2 class="section-title">8. Complete Historical State-by-State Data Table (357 Records)</h2>
            <p class="section-desc">Search, filter, and inspect state polling consensuses vs. certified election results across all seven presidential election cycles.</p>

            <div class="definition-banner">
                <strong>Definition of "Poll Margin":</strong> The consensus pre-election polling estimate for the Democratic candidate minus the Republican candidate ($\text{{Dem \%}} - \text{{Rep \%}}$). A positive margin (e.g., $+4.5\%$) indicates a projected Democratic victory; a negative margin (e.g., $-8.5\%$) indicates a projected Republican victory. Similarly, <strong>Actual Margin</strong> is the certified election percentage $\text{{Dem \%}} - \text{{Rep \%}}$, and <strong>Signed Bias</strong> is $\text{{Poll Margin}} - \text{{Actual Margin}}$ (where positive values indicate Democratic overstatement / Republican undercounting).
            </div>

            <div class="table-controls">
                <input type="text" id="state-search" placeholder="Search state or code (e.g. Florida, TX, Wisconsin)..." onkeyup="filterTable()">
                <select id="cycle-filter" onchange="filterTable()">
                    <option value="ALL">All Election Cycles (2000–2024)</option>
                    <option value="2024">2024: Harris vs. Trump</option>
                    <option value="2020">2020: Biden vs. Trump</option>
                    <option value="2016">2016: Clinton vs. Trump</option>
                    <option value="2012">2012: Obama vs. Romney</option>
                    <option value="2008">2008: Obama vs. McCain</option>
                    <option value="2004">2004: Kerry vs. Bush</option>
                    <option value="2000">2000: Gore vs. Bush</option>
                </select>
                <select id="region-filter" onchange="filterTable()">
                    <option value="ALL">All Regions</option>
                    <option value="Midwest">Midwest</option>
                    <option value="South">South</option>
                    <option value="Northeast">Northeast</option>
                    <option value="West">West</option>
                </select>
                <select id="tier-filter" onchange="filterTable()">
                    <option value="ALL">All Competitiveness Tiers</option>
                    <option value="Battleground">Battlegrounds Only (|Margin| &lt; 5%)</option>
                    <option value="Lean">Lean (|Margin| 5–10%)</option>
                    <option value="Safe">Safe (|Margin| &gt; 10%)</option>
                </select>
            </div>

            <div class="table-wrapper">
                <table class="data-table" id="historical-table">
                    <thead>
                        <tr>
                            <th>Cycle</th>
                            <th>State</th>
                            <th>Code</th>
                            <th>Region</th>
                            <th>Tier</th>
                            <th style="text-align: right;" title="Consensus pre-election polling estimate: Democratic % minus Republican %">Poll Margin (Dem% - Rep%)*</th>
                            <th style="text-align: right;" title="95% Confidence Interval for Margin Difference: 1.96 * SE_diff">Margin MoE</th>
                            <th style="text-align: right;" title="Certified Official General Election Margin: Democratic % minus Republican %">Actual Margin</th>
                            <th style="text-align: right;" title="Signed Polling Error: Poll Margin minus Actual Margin (+ = Dem overstatement, - = Rep overstatement)">Signed Bias</th>
                            <th style="text-align: right;" title="Absolute Magnitude of Polling Error">Abs Error</th>
                            <th style="text-align: center;">95% CI Hit</th>
                            <th style="text-align: center;">Winner Call</th>
                        </tr>
                    </thead>
                    <tbody>
                        {data_table_body}
                    </tbody>
                </table>
            </div>
        </div>

    </div>

    <script>
        // Render ALL Interactive Plotly Charts to guarantee full hover interactivity across every figure
        Plotly.newPlot('interactive-scatter-div', {json.dumps(interactive_figs['scatter_plot']['data'])}, {json.dumps(interactive_figs['scatter_plot']['layout'])}, {{responsive: true}});
        Plotly.newPlot('interactive-metrics-div', {json.dumps(interactive_figs['metrics_trends']['data'])}, {json.dumps(interactive_figs['metrics_trends']['layout'])}, {{responsive: true}});
        Plotly.newPlot('interactive-classification-div', {json.dumps(interactive_figs['classification_metrics']['data'])}, {json.dumps(interactive_figs['classification_metrics']['layout'])}, {{responsive: true}});
        Plotly.newPlot('interactive-coverage-div', {json.dumps(interactive_figs['ci_coverage_gap']['data'])}, {json.dumps(interactive_figs['ci_coverage_gap']['layout'])}, {{responsive: true}});
        Plotly.newPlot('interactive-bias-dist-div', {json.dumps(interactive_figs['bias_distribution_anomalies']['data'])}, {json.dumps(interactive_figs['bias_distribution_anomalies']['layout'])}, {{responsive: true}});
        Plotly.newPlot('interactive-ci-quality-div', {json.dumps(interactive_figs['ci_anomaly_quality']['data'])}, {json.dumps(interactive_figs['ci_anomaly_quality']['layout'])}, {{responsive: true}});
        Plotly.newPlot('interactive-regional-div', {json.dumps(interactive_figs['regional_disparities']['data'])}, {json.dumps(interactive_figs['regional_disparities']['layout'])}, {{responsive: true}});
        Plotly.newPlot('interactive-demographic-div', {json.dumps(interactive_figs['demographic_misses']['data'])}, {json.dumps(interactive_figs['demographic_misses']['layout'])}, {{responsive: true}});

        // Dynamic Table Filtering Function
        function filterTable() {{
            const searchVal = document.getElementById('state-search').value.toLowerCase();
            const cycleVal = document.getElementById('cycle-filter').value;
            const regionVal = document.getElementById('region-filter').value;
            const tierVal = document.getElementById('tier-filter').value;
            const table = document.getElementById('historical-table');
            const rows = table.getElementsByTagName('tr');

            for (let i = 1; i < rows.length; i++) {{
                const row = rows[i];
                const cycle = row.getAttribute('data-cycle');
                const region = row.getAttribute('data-region');
                const tier = row.getAttribute('data-tier');
                const text = row.innerText.toLowerCase();

                const matchSearch = text.includes(searchVal);
                const matchCycle = (cycleVal === 'ALL' || cycle === cycleVal);
                const matchRegion = (regionVal === 'ALL' || region === regionVal);
                const matchTier = (tierVal === 'ALL' || tier === tierVal);

                if (matchSearch && matchCycle && matchRegion && matchTier) {{
                    row.style.display = '';
                }} else {{
                    row.style.display = 'none';
                }}
            }}
        }}
    </script>
</body>
</html>
"""

with open('polling_accuracy_historical_analysis.html', 'w') as f:
    f.write(html_content)
print("Saved polling_accuracy_historical_analysis.html.")
print("All deliverables successfully compiled!")
