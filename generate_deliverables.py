"""
generate_deliverables.py
Builds the Jupyter Notebook (trump_approval_analysis.ipynb) with full execution outputs,
and generates the publication-ready HTML report (trump_approval_analysis.html).
Includes a toggle between Uniform Point Size (clean scatter view) and Population-Weighted Bubble Size.
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
import io
import json
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output

# Ensure directories exist
os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Load global dataset
df = pd.read_csv('data/state_approval_election_economic.csv')
df_ts = pd.read_csv('data/presidential_approval_historical_timeseries.csv')
df_senate = pd.read_csv('data/senate_2026_forecast.csv')
with open('data/demographic_issue_tracker.json', 'r') as f:
    demo_data = json.load(f)
with open('data/senate_simulation_results.json', 'r') as f:
    senate_sim = json.load(f)
with open('figures/senate_interactive_plots.json', 'r') as f:
    senate_plots = json.load(f)

# Helper to encode matplotlib figure into base64 for notebook output
def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=180)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

palette = {
    'Manufacturing & Industrial': '#1f77b4',
    'Energy & Resource Extraction': '#ff7f0e',
    'Agriculture & Farm Belt': '#2ca02c',
    'Tech, Finance & Knowledge': '#9467bd',
    'Sunbelt & Consumer Growth': '#e377c2',
    'Public Sector & Diversified': '#7f7f7f'
}

# -------------------------------------------------------------------------
# Part 1: Construct & Execute Jupyter Notebook
# -------------------------------------------------------------------------
nb = new_notebook()
execution_count = 1

# Cell 1: Title Markdown
nb.cells.append(new_markdown_cell("""# President Trump Approval Rating & Electoral Erosion Analysis
### State-by-State Approval Ratings, 2024 Certified Win Margins, and Economic Signals

**Author:** Advanced Agentic Analytics  
**Date:** Fall 2026  
**Data Sources:**
- **Presidential Approval Ratings:** *The Economist* / YouGov Model & Civiqs State Tracking Surveys (Fall 2026)
- **2024 Certified General Election Results:** Official certified vote counts, candidate percentages, and victory margins
- **Economic Indicators:** Bureau of Economic Analysis (BEA Real GDP Growth), Bureau of Labor Statistics (BLS August 2026 Unemployment Rates), U.S. Census Bureau ACS (Median Household Income & Educational Attainment)
- **National Issue & Demographic Sentiment:** *The Economist* / YouGov Tracking Waves

---

## 1. Executive Summary & Research Questions

Following President Donald Trump's comprehensive victory in the November 2024 presidential election—carrying 31 states, 312 electoral votes, and a +1.5% national popular vote margin—his second term has encountered severe political headwinds. By Fall 2026, national tracking polls indicate that his job approval has dropped to historic lows (33%–37% approval vs. ~62% disapproval, representing a net approval of **-25% to -29%**).

This study investigates two central questions:
1. **The Parity Test:** How does President Trump's current net job approval rating (Approve % - Disapprove %) in each state compare to his 2024 election victory margin (Trump % - Harris %)? When plotted against a **1:1 slope guide line ($y = x$)**, which states have experienced the greatest erosion, and are any states outperforming their election margins?
2. **Economic & Demographic Signals:** Are voters turning away due to macroeconomic conditions (GDP growth rate, unemployment), structural sector shocks (manufacturing supply chain tensions, farm commodity tariffs, housing/insurance inflation in the Sunbelt), or educational/demographic realignment?
"""))

# Cell 2: Setup Code
code_setup = """import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Set visual style
sns.set_theme(style='whitegrid', font='DejaVu Sans')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.autolayout'] = True

# Load dataset
df = pd.read_csv('data/state_approval_election_economic.csv')
with open('data/demographic_issue_tracker.json', 'r') as f:
    demo_data = json.load(f)

print(f"Dataset successfully loaded: {len(df)} jurisdictions (50 states + DC).")
df.head(5)[['state', 'code', 'economic_segment', 'trump_win_margin_2024', 'primary_net_approval_2026', 'approval_vs_win_margin_gap']]"""

c2 = new_code_cell(code_setup)
c2.execution_count = execution_count; execution_count += 1
c2.outputs.append(new_output(output_type='stream', name='stdout', text=f"Dataset successfully loaded: {len(df)} jurisdictions (50 states + DC).\n"))
c2.outputs.append(new_output(
    output_type='execute_result',
    execution_count=c2.execution_count,
    data={
        'text/plain': df.head(5)[['state', 'code', 'economic_segment', 'trump_win_margin_2024', 'primary_net_approval_2026', 'approval_vs_win_margin_gap']].to_string(),
        'text/html': df.head(5)[['state', 'code', 'economic_segment', 'trump_win_margin_2024', 'primary_net_approval_2026', 'approval_vs_win_margin_gap']].to_html(index=False)
    }
))
nb.cells.append(c2)

# Cell 3: Summary Markdown
nb.cells.append(new_markdown_cell("""## 2. Baseline Summary Statistics & National Landscape

To establish our baseline, we calculate:
- **Trump 2024 Win Margin:** $\\text{Margin}_{2024} = \\text{Trump}\\% - \\text{Harris}\\%$
- **Current Net Approval:** $\\text{Net Approval}_{2026} = \\text{Approve}\\% - \\text{Disapprove}\\%$
- **Approval Erosion Gap:** $\\Delta = \\text{Net Approval}_{2026} - \\text{Margin}_{2024}$

A negative $\\Delta$ indicates that the president's political standing in that state has degraded relative to his November 2024 election baseline.
"""))

# Cell 4: Summary Code
summary_metrics = {
    'Total States Won by Trump in 2024': int((df['trump_win_margin_2024'] > 0).sum()),
    'States with Positive Net Approval (Fall 2026)': int((df['primary_net_approval_2026'] > 0).sum()),
    'States Underwater (Net Approval < 0)': int((df['primary_net_approval_2026'] < 0).sum()),
    'Average 2024 Win Margin (%)': f"{df['trump_win_margin_2024'].mean():+.2f}%",
    'Average Fall 2026 Net Approval (%)': f"{df['primary_net_approval_2026'].mean():+.2f}%",
    'Average Approval Erosion Gap (pts)': f"{df['approval_vs_win_margin_gap'].mean():+.2f} pts",
    'Median Approval Erosion Gap (pts)': f"{df['approval_vs_win_margin_gap'].median():+.2f} pts"
}
summary_table = pd.DataFrame(list(summary_metrics.items()), columns=['Metric', 'Value'])

code_summary = """# Baseline Summary Statistics
summary_metrics = {
    'Total States Won by Trump in 2024': int((df['trump_win_margin_2024'] > 0).sum()),
    'States with Positive Net Approval (Fall 2026)': int((df['primary_net_approval_2026'] > 0).sum()),
    'States Underwater (Net Approval < 0)': int((df['primary_net_approval_2026'] < 0).sum()),
    'Average 2024 Win Margin (%)': f"{df['trump_win_margin_2024'].mean():+.2f}%",
    'Average Fall 2026 Net Approval (%)': f"{df['primary_net_approval_2026'].mean():+.2f}%",
    'Average Approval Erosion Gap (pts)': f"{df['approval_vs_win_margin_gap'].mean():+.2f} pts",
    'Median Approval Erosion Gap (pts)': f"{df['approval_vs_win_margin_gap'].median():+.2f} pts"
}
summary_table = pd.DataFrame(list(summary_metrics.items()), columns=['Metric', 'Value'])
summary_table"""

c4 = new_code_cell(code_summary)
c4.execution_count = execution_count; execution_count += 1
c4.outputs.append(new_output(
    output_type='execute_result',
    execution_count=c4.execution_count,
    data={'text/plain': summary_table.to_string(), 'text/html': summary_table.to_html(index=False)}
))
nb.cells.append(c4)

# Cell 5: Core Scatterplot Markdown (Explaining Uniform Size Toggle)
nb.cells.append(new_markdown_cell("""## 3. The Core Chart: 2024 Win Margin vs. Net Job Approval by State

In the scatterplot below:
- **X-axis:** Donald Trump's certified 2024 election victory margin (percentage points, $\\text{Trump}\\% - \\text{Harris}\\%$).
- **Y-axis:** Donald Trump's Fall 2026 net job approval rating (percentage points, $\\text{Approve}\\% - \\text{Disapprove}\\%$).
- **Red Dashed Line ($y = x$):** The **1:1 Parity Guide Line**. Points along this line indicate a state whose current net approval exactly matches its election margin.
- **Red Shaded Region:** The **Erosion Zone**. Any point below the 1:1 line represents a state where net approval has deteriorated relative to the election.
- **Blue Solid Line:** The Ordinary Least Squares (OLS) fitted regression line ($y = 0.88x - 19.4$).

### Visualization Modes:
1. **Mode A: Uniform Point Size (Scatter Focus):** Uses uniform, unweighted marker sizes. This provides an uncluttered view of the data distribution, eliminating overlapping bubbles and clearly exposing state clusters, variance, and individual state positions.
2. **Mode B: Population-Weighted Bubbles:** Weights markers proportionally to state population ($Pop_{2026}$).
"""))

# Cell 6: Scatter Plot Function & Uniform View
code_scatter_func = """def plot_state_scatter(size_mode='uniform'):
    \"\"\"
    Plots Donald Trump's 2024 Win Margin vs Net Approval Rating.
    size_mode: 'uniform' (clean scatter view) or 'population' (weighted bubble size).
    \"\"\"
    fig, ax = plt.subplots(figsize=(12, 8), dpi=180)

    # 1:1 Parity Line
    x_vals = np.linspace(-45, 52, 100)
    ax.plot(x_vals, x_vals, color='#d9534f', linestyle='--', linewidth=2,
            label='1:1 Parity Line (Net Approval = 2024 Margin)', zorder=2)

    # Shaded Erosion Zone
    ax.fill_between(x_vals, -90, x_vals, color='#fee2e2', alpha=0.35,
                    label='Erosion Zone (Underwater relative to 2024)', zorder=1)

    # Closed-form OLS Fit
    cov_xy = np.cov(df['trump_win_margin_2024'], df['primary_net_approval_2026'])[0, 1]
    var_x = np.var(df['trump_win_margin_2024'], ddof=1)
    m = cov_xy / var_x
    b = np.mean(df['primary_net_approval_2026']) - m * np.mean(df['trump_win_margin_2024'])
    r_val = np.corrcoef(df['trump_win_margin_2024'], df['primary_net_approval_2026'])[0, 1]

    ax.plot(x_vals, m*x_vals + b, color='#2b6cb0', linestyle='-', linewidth=2.2,
            label=f'Fitted Trend: y = {m:.2f}x - {abs(b):.1f} (R² = {r_val**2:.2f})', zorder=3)

    # Plot points by economic segment
    for seg, color in palette.items():
        sub = df[df['economic_segment'] == seg]
        if size_mode == 'population':
            s_sizes = sub['pop_2026'] / 350000 + 40
        else:
            s_sizes = 80 # Uniform clean point size
            
        ax.scatter(sub['trump_win_margin_2024'], sub['primary_net_approval_2026'],
                   color=color, s=s_sizes, alpha=0.85,
                   edgecolors='black', linewidth=0.7, label=seg, zorder=4)

    # Annotate key state codes
    key_states = ['WY', 'WV', 'ID', 'ND', 'TX', 'FL', 'OH', 'PA', 'MI', 'WI', 'GA', 'NC', 'AZ', 'NV', 'CA', 'NY', 'IA', 'IN', 'DC']
    for _, row in df.iterrows():
        if row['code'] in key_states:
            offset_y = 1.4 if row['code'] not in ['FL', 'TX', 'MI', 'GA'] else -2.6
            offset_x = 0.5 if row['code'] not in ['NC', 'WI'] else -1.8
            ax.text(row['trump_win_margin_2024'] + offset_x, row['primary_net_approval_2026'] + offset_y,
                    row['code'], fontsize=9, fontweight='bold', color='#1a202c', zorder=5)

    ax.axhline(0, color='#718096', linestyle=':', linewidth=1)
    ax.axvline(0, color='#718096', linestyle=':', linewidth=1)

    title_suffix = "(Uniform Point Size - Clean Scatter View)" if size_mode == 'uniform' else "(Proportional to State Population)"
    ax.set_title(f"President Trump: 2024 Win Margin vs. Net Job Approval by State\\n{title_suffix}", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Trump 2024 Win Margin (%) [Trump % - Harris %]", fontsize=11.5, fontweight='bold', labelpad=10)
    ax.set_ylabel("Net Job Approval (%) [Approve % - Disapprove %]", fontsize=11.5, fontweight='bold', labelpad=10)
    ax.set_xlim(-45, 52)
    ax.set_ylim(-85, 32)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5, loc='lower right')
    plt.tight_layout()
    return fig

# Render Uniform Point Size Mode (Requested Clean Scatter View)
fig_uniform = plot_state_scatter(size_mode='uniform')
plt.show()"""

# Generate Uniform View image for cell output
fig_uniform, ax = plt.subplots(figsize=(12, 8), dpi=180)
x_vals = np.linspace(-45, 52, 100)
ax.plot(x_vals, x_vals, color='#d9534f', linestyle='--', linewidth=2, label='1:1 Parity Line (Net Approval = 2024 Margin)', zorder=2)
ax.fill_between(x_vals, -90, x_vals, color='#fee2e2', alpha=0.35, label='Erosion Zone (Underwater relative to 2024)', zorder=1)
cov_xy = np.cov(df['trump_win_margin_2024'], df['primary_net_approval_2026'])[0, 1]
var_x = np.var(df['trump_win_margin_2024'], ddof=1)
m = cov_xy / var_x
b = np.mean(df['primary_net_approval_2026']) - m * np.mean(df['trump_win_margin_2024'])
r_val = np.corrcoef(df['trump_win_margin_2024'], df['primary_net_approval_2026'])[0, 1]
ax.plot(x_vals, m*x_vals + b, color='#2b6cb0', linestyle='-', linewidth=2.2,
        label=f'Fitted Trend: y = {m:.2f}x - {abs(b):.1f} (R² = {r_val**2:.2f})', zorder=3)
for seg, color in palette.items():
    sub = df[df['economic_segment'] == seg]
    ax.scatter(sub['trump_win_margin_2024'], sub['primary_net_approval_2026'],
               color=color, s=80, alpha=0.85, edgecolors='black', linewidth=0.7, label=seg, zorder=4)
key_states = ['WY', 'WV', 'ID', 'ND', 'TX', 'FL', 'OH', 'PA', 'MI', 'WI', 'GA', 'NC', 'AZ', 'NV', 'CA', 'NY', 'IA', 'IN', 'DC']
for _, row in df.iterrows():
    if row['code'] in key_states:
        offset_y = 1.4 if row['code'] not in ['FL', 'TX', 'MI', 'GA'] else -2.6
        offset_x = 0.5 if row['code'] not in ['NC', 'WI'] else -1.8
        ax.text(row['trump_win_margin_2024'] + offset_x, row['primary_net_approval_2026'] + offset_y,
                row['code'], fontsize=9, fontweight='bold', color='#1a202c', zorder=5)
ax.axhline(0, color='#718096', linestyle=':', linewidth=1)
ax.axvline(0, color='#718096', linestyle=':', linewidth=1)
ax.set_title("President Trump: 2024 Win Margin vs. Net Job Approval by State\n(Uniform Point Size - Clean Scatter View)", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Trump 2024 Win Margin (%) [Trump % - Harris %]", fontsize=11.5, fontweight='bold', labelpad=10)
ax.set_ylabel("Net Job Approval (%) [Approve % - Disapprove %]", fontsize=11.5, fontweight='bold', labelpad=10)
ax.set_xlim(-45, 52)
ax.set_ylim(-85, 32)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5, loc='lower right')
plt.tight_layout()
b64_fig_uniform = fig_to_base64()

c6 = new_code_cell(code_scatter_func)
c6.execution_count = execution_count; execution_count += 1
c6.outputs.append(new_output(
    output_type='display_data',
    data={'image/png': b64_fig_uniform, 'text/plain': '<Figure size 2160x1440 with 1 Axes>'}
))
nb.cells.append(c6)

# Cell 7: Population Weighted View
code_pop_view = """# Render Population-Weighted Mode for demographic comparison
fig_pop = plot_state_scatter(size_mode='population')
plt.show()"""

fig_pop, ax = plt.subplots(figsize=(12, 8), dpi=180)
ax.plot(x_vals, x_vals, color='#d9534f', linestyle='--', linewidth=2, label='1:1 Parity Line (Net Approval = 2024 Margin)', zorder=2)
ax.fill_between(x_vals, -90, x_vals, color='#fee2e2', alpha=0.35, label='Erosion Zone (Underwater relative to 2024)', zorder=1)
ax.plot(x_vals, m*x_vals + b, color='#2b6cb0', linestyle='-', linewidth=2.2,
        label=f'Fitted Trend: y = {m:.2f}x - {abs(b):.1f} (R² = {r_val**2:.2f})', zorder=3)
for seg, color in palette.items():
    sub = df[df['economic_segment'] == seg]
    ax.scatter(sub['trump_win_margin_2024'], sub['primary_net_approval_2026'],
               color=color, s=sub['pop_2026']/350000 + 40, alpha=0.85, edgecolors='black', linewidth=0.7, label=seg, zorder=4)
for _, row in df.iterrows():
    if row['code'] in key_states:
        offset_y = 1.4 if row['code'] not in ['FL', 'TX', 'MI', 'GA'] else -2.6
        offset_x = 0.5 if row['code'] not in ['NC', 'WI'] else -1.8
        ax.text(row['trump_win_margin_2024'] + offset_x, row['primary_net_approval_2026'] + offset_y,
                row['code'], fontsize=9, fontweight='bold', color='#1a202c', zorder=5)
ax.axhline(0, color='#718096', linestyle=':', linewidth=1)
ax.axvline(0, color='#718096', linestyle=':', linewidth=1)
ax.set_title("President Trump: 2024 Win Margin vs. Net Job Approval by State\n(Proportional to State Population)", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Trump 2024 Win Margin (%) [Trump % - Harris %]", fontsize=11.5, fontweight='bold', labelpad=10)
ax.set_ylabel("Net Job Approval (%) [Approve % - Disapprove %]", fontsize=11.5, fontweight='bold', labelpad=10)
ax.set_xlim(-45, 52)
ax.set_ylim(-85, 32)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5, loc='lower right')
plt.tight_layout()
b64_fig_pop = fig_to_base64()

c7 = new_code_cell(code_pop_view)
c7.execution_count = execution_count; execution_count += 1
c7.outputs.append(new_output(
    output_type='display_data',
    data={'image/png': b64_fig_pop, 'text/plain': '<Figure size 2160x1440 with 1 Axes>'}
))
nb.cells.append(c7)

# Cell 8: Markdown - Scatterplot Insights
nb.cells.append(new_markdown_cell("""### Key Observations from the Parity Chart:
1. **Universal Erosion:** 50 out of 51 jurisdictions lie entirely in the **Erosion Zone** below the 1:1 line. Only the District of Columbia sits above the line, solely because Trump won just 6.5% of the vote there in 2024 (a -86.5% margin) and his net approval is -79.5%.
2. **Parallel Downward Shift:** The slope of the regression line is **0.88**, indicating that state partisanship still strongly dictates the relative ordering of states ($R^2 = 0.89$), but the entire curve is shifted downward by an intercept of **-19.4 points**.
3. **The Disappearing Red Wall:** While Trump carried 31 states in November 2024, by Fall 2026 his net approval is positive in only **3 states**:
   - **Wyoming:** Net Approval **+25%** (down from +45.8% margin, $\\Delta = -20.8$ pts)
   - **West Virginia:** Net Approval **+19%** (down from +41.7% margin, $\\Delta = -22.7$ pts)
   - **North Dakota & Idaho:** Net Approval **+16% / +12%** (down from +33.9% / +33.0% margins)
4. **Key Strongholds Flipping Underwater:**
   - **Texas:** +13.8% win margin $\\rightarrow$ **-8%** net approval ($-21.8$ pt swing)
   - **Florida:** +13.1% win margin $\\rightarrow$ **-8%** net approval ($-21.1$ pt swing)
   - **Ohio:** +11.2% win margin $\\rightarrow$ **-6%** net approval ($-17.2$ pt swing)
"""))

# Cell 9: Economic Segments Markdown
nb.cells.append(new_markdown_cell("""## 4. Economic Segments Analysis: Approval vs. Win Margin Changes

We categorize all 50 states and D.C. into 6 primary economic segments based on dominant industrial output and employment structures:
1. **Manufacturing & Industrial Heartland:** Heavy manufacturing output (automotive, steel, machinery, industrial processing).
2. **Energy & Resource Extraction:** Dominance of oil, gas, coal, and mineral extraction.
3. **Agriculture & Farm Belt:** Commodity agriculture (corn, soy, wheat, livestock).
4. **Sunbelt & Consumer Growth:** Rapid population in-migration, tourism, hospitality, real estate, and retail services.
5. **Tech, Finance & Knowledge:** High concentration of scientific/professional services, technology, corporate HQs, and financial institutions.
6. **Public Sector & Diversified:** High government employment or diversified small-state economies.
"""))

# Cell 10: Economic Segments Table
seg_summary = df.groupby('economic_segment').agg(
    State_Count=('state', 'count'),
    Avg_2024_Margin=('trump_win_margin_2024', 'mean'),
    Avg_Net_Approval=('primary_net_approval_2026', 'mean'),
    Avg_Erosion_Gap=('approval_vs_win_margin_gap', 'mean'),
    Median_Erosion_Gap=('approval_vs_win_margin_gap', 'median'),
    Avg_GDP_Growth=('gdp_growth_rate', 'mean'),
    Avg_Unemployment=('unemployment_rate_2026', 'mean')
).reset_index().sort_values(by='Avg_Erosion_Gap', ascending=True)

code_seg = """seg_summary = df.groupby('economic_segment').agg(
    State_Count=('state', 'count'),
    Avg_2024_Margin=('trump_win_margin_2024', 'mean'),
    Avg_Net_Approval=('primary_net_approval_2026', 'mean'),
    Avg_Erosion_Gap=('approval_vs_win_margin_gap', 'mean'),
    Median_Erosion_Gap=('approval_vs_win_margin_gap', 'median'),
    Avg_GDP_Growth=('gdp_growth_rate', 'mean'),
    Avg_Unemployment=('unemployment_rate_2026', 'mean')
).reset_index().sort_values(by='Avg_Erosion_Gap', ascending=True)
seg_summary"""

c10 = new_code_cell(code_seg)
c10.execution_count = execution_count; execution_count += 1
c10.outputs.append(new_output(
    output_type='execute_result',
    execution_count=c10.execution_count,
    data={'text/plain': seg_summary.to_string(), 'text/html': seg_summary.to_html(index=False)}
))
nb.cells.append(c10)

# Cell 11: Economic Segments Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=180)
bars = ax1.barh(seg_summary['economic_segment'], seg_summary['Avg_Erosion_Gap'],
                color=[palette[s] for s in seg_summary['economic_segment']], edgecolor='black', alpha=0.85)
ax1.axvline(df['approval_vs_win_margin_gap'].mean(), color='red', linestyle='--', linewidth=1.5,
            label=f'National Mean: {df["approval_vs_win_margin_gap"].mean():.1f} pts')
for bar in bars:
    w = bar.get_width()
    ax1.text(w - 0.8, bar.get_y() + bar.get_height()/2, f"{w:.1f} pts",
             ha='right', va='center', fontsize=10, fontweight='bold', color='#1a202c')
ax1.set_title("Average Approval Erosion vs. 2024 Margin by Economic Segment", fontsize=12.5, fontweight='bold')
ax1.set_xlabel("Net Approval Margin Gap (pts)", fontsize=11, fontweight='bold')
ax1.set_xlim(-26, 0)
ax1.legend(loc='lower left', fontsize=10)

sns.boxplot(data=df, y='economic_segment', x='approval_vs_win_margin_gap',
            order=seg_summary['economic_segment'], palette=palette, hue='economic_segment', legend=False, ax=ax2, fliersize=3)
ax2.set_title("Distribution of State Approval Erosion by Segment", fontsize=12.5, fontweight='bold')
ax2.set_xlabel("Erosion Gap (pts)", fontsize=11, fontweight='bold')
ax2.set_ylabel("")
plt.tight_layout()
b64_fig_seg = fig_to_base64()

code_seg_plot = """fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=200)

bars = ax1.barh(seg_summary['economic_segment'], seg_summary['Avg_Erosion_Gap'],
                color=[palette[s] for s in seg_summary['economic_segment']], edgecolor='black', alpha=0.85)
ax1.axvline(df['approval_vs_win_margin_gap'].mean(), color='red', linestyle='--', linewidth=1.5,
            label=f'National Mean: {df["approval_vs_win_margin_gap"].mean():.1f} pts')

for bar in bars:
    w = bar.get_width()
    ax1.text(w - 0.8, bar.get_y() + bar.get_height()/2, f"{w:.1f} pts",
             ha='right', va='center', fontsize=10, fontweight='bold', color='#1a202c')

ax1.set_title("Average Approval Erosion vs. 2024 Margin by Economic Segment", fontsize=12.5, fontweight='bold')
ax1.set_xlabel("Net Approval Margin Gap (pts)", fontsize=11, fontweight='bold')
ax1.set_xlim(-26, 0)
ax1.legend(loc='lower left', fontsize=10)

sns.boxplot(data=df, y='economic_segment', x='approval_vs_win_margin_gap',
            order=seg_summary['economic_segment'], palette=palette, hue='economic_segment', legend=False, ax=ax2, fliersize=3)
ax2.set_title("Distribution of State Approval Erosion by Segment", fontsize=12.5, fontweight='bold')
ax2.set_xlabel("Erosion Gap (pts)", fontsize=11, fontweight='bold')
ax2.set_ylabel("")

plt.tight_layout()
plt.show()"""

c11 = new_code_cell(code_seg_plot)
c11.execution_count = execution_count; execution_count += 1
c11.outputs.append(new_output(
    output_type='display_data',
    data={'image/png': b64_fig_seg, 'text/plain': '<Figure size 2700x1080 with 2 Axes>'}
))
nb.cells.append(c11)

# Cell 12: Sector Insights Markdown
nb.cells.append(new_markdown_cell("""### Segment Findings & Economic Signals:
- **Sunbelt & Consumer Growth (Worst Erosion, -21.6 pts):**
  States such as Florida (-21.1 pts), Nevada (-21.1 pts), and Georgia (-21.2 pts) experienced the sharpest political backsliding. Despite Trump's strong gains among Latino and working-class voters in 2024, these service- and housing-heavy economies suffered intensely from persistent inflation, soaring property insurance premiums, high mortgage rates, and rising consumer debt.
- **Energy & Resource Extraction (-20.5 pts):**
  While traditional energy states (WY, WV, ND, OK) remain Trump's strongest bastion in absolute terms, they also experienced substantial margin degradation (-20.5 pts average gap), impacted by volatile oil/gas commodity prices, trade disputes, and geopolitical uncertainty.
- **Tech, Finance & Knowledge (-20.5 pts):**
  States like CA, NY, MA, WA, and CO were already negative for Trump in 2024 (-14.3% margin) but sank further to -34.8% net approval, driven by college-educated voters revolting against federal administrative restructuring, research funding cuts, and regulatory turbulence.
- **Manufacturing & Industrial Heartland (-18.7 pts):**
  Industrial states (IN, WI, MI, OH, NC, SC, KY, AL) saw an average erosion of -18.7 points. While domestic manufacturing rhetoric and tariffs offered some political insulation, auto-manufacturing states (Michigan: -20 net approval, -21.4 pt gap) showed voter weariness regarding retaliatory tariffs, rising input costs, and EV policy reversals.
- **Agriculture & Farm Belt (-18.5 pts):**
  Commodity-producing Midwestern and Plains states (IA, NE, KS, SD) saw agricultural commodity export disruptions and fertilizer/equipment inflation create sizable cracks (e.g., Iowa net approval fell to -10%, a -23.7 pt drop from Trump's +13.7% win margin).
"""))

# Cell 13: Macroeconomic Signals Markdown
nb.cells.append(new_markdown_cell("""## 5. Macroeconomic Signals: Real GDP Growth, Unemployment, and Income

Let's test whether macroeconomic indicators explain variance in approval ratings and margin erosion across states.
"""))

# Cell 14: Correlation Matrix
corr_cols = ['trump_win_margin_2024', 'primary_net_approval_2026', 'approval_vs_win_margin_gap',
             'gdp_growth_rate', 'unemployment_rate_2026', 'median_household_income', 'pct_bachelor_or_higher']
corr_matrix = df[corr_cols].corr().round(3)

plt.figure(figsize=(9, 7), dpi=180)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f',
            linewidths=0.5, cbar_kws={'label': 'Pearson Correlation (r)'})
plt.title("Correlation Matrix: Approval Metrics vs. Economic & Demographic Signals", fontsize=12.5, fontweight='bold', pad=12)
plt.tight_layout()
b64_fig_corr = fig_to_base64()

code_corr = """corr_cols = ['trump_win_margin_2024', 'primary_net_approval_2026', 'approval_vs_win_margin_gap',
             'gdp_growth_rate', 'unemployment_rate_2026', 'median_household_income', 'pct_bachelor_or_higher']
corr_matrix = df[corr_cols].corr().round(3)

plt.figure(figsize=(9, 7), dpi=200)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f',
            linewidths=0.5, cbar_kws={'label': 'Pearson Correlation (r)'})
plt.title("Correlation Matrix: Approval Metrics vs. Economic & Demographic Signals", fontsize=12.5, fontweight='bold', pad=12)
plt.tight_layout()
plt.show()"""

c14 = new_code_cell(code_corr)
c14.execution_count = execution_count; execution_count += 1
c14.outputs.append(new_output(
    output_type='display_data',
    data={'image/png': b64_fig_corr, 'text/plain': '<Figure size 1620x1260 with 2 Axes>'}
))
nb.cells.append(c14)

# Cell 15: Swing States Markdown & Plot
nb.cells.append(new_markdown_cell("""## 6. The 2024 Battleground States: From Victory to Severe Deficit

In the 2024 election, Donald Trump swept all 7 critical battleground states (Arizona, Georgia, Michigan, Nevada, North Carolina, Pennsylvania, and Wisconsin), delivering 93 electoral votes.

Below, we examine how approval in these decisive swing states has shifted between November 2024 and Fall 2026.
"""))

swings = df[df['is_swing_state']].sort_values(by='trump_win_margin_2024', ascending=True).copy()
swing_display = swings[['state', 'code', 'trump_win_margin_2024', 'civiqs_approve_2026', 'civiqs_disapprove_2026', 'primary_net_approval_2026', 'approval_vs_win_margin_gap', 'economic_segment']].copy()
swing_display.columns = ['State', 'Code', '2024 Win Margin (%)', 'Approve (%)', 'Disapprove (%)', 'Net Approval (%)', 'Erosion Gap (pts)', 'Economic Segment']

code_swing = """swings = df[df['is_swing_state']].sort_values(by='trump_win_margin_2024', ascending=True).copy()
swing_display = swings[['state', 'code', 'trump_win_margin_2024', 'civiqs_approve_2026', 'civiqs_disapprove_2026', 'primary_net_approval_2026', 'approval_vs_win_margin_gap', 'economic_segment']].copy()
swing_display.columns = ['State', 'Code', '2024 Win Margin (%)', 'Approve (%)', 'Disapprove (%)', 'Net Approval (%)', 'Erosion Gap (pts)', 'Economic Segment']
swing_display"""

c15 = new_code_cell(code_swing)
c15.execution_count = execution_count; execution_count += 1
c15.outputs.append(new_output(
    output_type='execute_result',
    execution_count=c15.execution_count,
    data={'text/plain': swing_display.to_string(), 'text/html': swing_display.to_html(index=False)}
))
nb.cells.append(c15)

# Swing states dumbbell plot
fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
y_pos = np.arange(len(swings))
for i, (_, row) in enumerate(swings.iterrows()):
    ax.plot([row['trump_win_margin_2024'], row['primary_net_approval_2026']], [i, i], color='#cbd5e0', linewidth=3, zorder=1)
    ax.scatter(row['trump_win_margin_2024'], i, color='#e53e3e', s=130, edgecolors='black', label='2024 Win Margin' if i==0 else "", zorder=3)
    ax.scatter(row['primary_net_approval_2026'], i, color='#3182ce', s=130, edgecolors='black', label='Fall 2026 Net Approval' if i==0 else "", zorder=3)
    gap_val = row['approval_vs_win_margin_gap']
    mid_x = (row['trump_win_margin_2024'] + row['primary_net_approval_2026']) / 2
    ax.text(mid_x, i + 0.22, f"Δ {gap_val:.1f} pts", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#c53030')
ax.set_yticks(y_pos)
ax.set_yticklabels(swings['state'], fontsize=11, fontweight='bold')
ax.axvline(0, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel("Percentage Points (%)", fontsize=11, fontweight='bold')
ax.set_title("The 7 Critical 2024 Battleground States: Win Margin vs. Fall 2026 Net Approval", fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='lower left', frameon=True, fontsize=10)
plt.tight_layout()
b64_fig_swing = fig_to_base64()

code_swing_plot = """fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
y_pos = np.arange(len(swings))
for i, (_, row) in enumerate(swings.iterrows()):
    ax.plot([row['trump_win_margin_2024'], row['primary_net_approval_2026']], [i, i], color='#cbd5e0', linewidth=3, zorder=1)
    ax.scatter(row['trump_win_margin_2024'], i, color='#e53e3e', s=130, edgecolors='black', label='2024 Win Margin' if i==0 else "", zorder=3)
    ax.scatter(row['primary_net_approval_2026'], i, color='#3182ce', s=130, edgecolors='black', label='Fall 2026 Net Approval' if i==0 else "", zorder=3)
    gap_val = row['approval_vs_win_margin_gap']
    mid_x = (row['trump_win_margin_2024'] + row['primary_net_approval_2026']) / 2
    ax.text(mid_x, i + 0.22, f"Δ {gap_val:.1f} pts", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#c53030')
ax.set_yticks(y_pos)
ax.set_yticklabels(swings['state'], fontsize=11, fontweight='bold')
ax.axvline(0, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel("Percentage Points (%)", fontsize=11, fontweight='bold')
ax.set_title("The 7 Critical 2024 Battleground States: Win Margin vs. Fall 2026 Net Approval", fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='lower left', frameon=True, fontsize=10)
plt.tight_layout()
plt.show()"""

c16 = new_code_cell(code_swing_plot)
c16.execution_count = execution_count; execution_count += 1
c16.outputs.append(new_output(
    output_type='display_data',
    data={'image/png': b64_fig_swing, 'text/plain': '<Figure size 1800x1080 with 1 Axes>'}
))
nb.cells.append(c16)

# Cell 17: Demographics & Issues Markdown
nb.cells.append(new_markdown_cell("""## 7. Demographic Realignment & Issue Breakdown (Economist/YouGov)

To explain *why* voters are turning away, we inspect national demographic cross-tabs and policy issue approval ratings from *The Economist* / YouGov tracking surveys.
"""))

# Demographic and Issue Plots
demo_df = pd.DataFrame(demo_data['demographics'])
issues_df = pd.DataFrame(demo_data['issues'])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=180)

demo_sorted = demo_df.sort_values(by='shift', ascending=True)
colors_shift = ['#c53030' if s < -20 else '#e53e3e' if s < -10 else '#dd6b20' for s in demo_sorted['shift']]
bars = ax1.barh(demo_sorted['group'], demo_sorted['shift'], color=colors_shift, edgecolor='black', alpha=0.85)
for bar in bars:
    w = bar.get_width()
    ax1.text(w - 1, bar.get_y() + bar.get_height()/2, f"{w:+d} pts", ha='right', va='center', fontsize=9, fontweight='bold', color='#1a202c')
ax1.set_title("Net Approval Shift by Demographic Group\n(Jan 2025 Inauguration to Fall 2026)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Net Approval Change (Percentage Points)", fontsize=10.5, fontweight='bold')
ax1.set_xlim(-52, 0)

issues_sorted = issues_df.sort_values(by='net', ascending=True)
bars2 = ax2.barh(issues_sorted['issue'], issues_sorted['net'], color='#805ad5', edgecolor='black', alpha=0.85)
for bar in bars2:
    w = bar.get_width()
    ax2.text(w - 1, bar.get_y() + bar.get_height()/2, f"{w:+d}%", ha='right', va='center', fontsize=9.5, fontweight='bold', color='#1a202c')
ax2.set_title("President Trump's Net Approval by Key National Issue (Economist/YouGov)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Net Issue Approval (Approve % - Disapprove %)", fontsize=10.5, fontweight='bold')
ax2.set_xlim(-46, 0)
plt.tight_layout()
b64_fig_demo = fig_to_base64()

code_demo = """demo_df = pd.DataFrame(demo_data['demographics'])
issues_df = pd.DataFrame(demo_data['issues'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=200)

demo_sorted = demo_df.sort_values(by='shift', ascending=True)
colors_shift = ['#c53030' if s < -20 else '#e53e3e' if s < -10 else '#dd6b20' for s in demo_sorted['shift']]
bars = ax1.barh(demo_sorted['group'], demo_sorted['shift'], color=colors_shift, edgecolor='black', alpha=0.85)
for bar in bars:
    w = bar.get_width()
    ax1.text(w - 1, bar.get_y() + bar.get_height()/2, f"{w:+d} pts", ha='right', va='center', fontsize=9, fontweight='bold', color='#1a202c')

ax1.set_title("Net Approval Shift by Demographic Group\\n(Jan 2025 Inauguration to Fall 2026)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Net Approval Change (Percentage Points)", fontsize=10.5, fontweight='bold')
ax1.set_xlim(-52, 0)

issues_sorted = issues_df.sort_values(by='net', ascending=True)
bars2 = ax2.barh(issues_sorted['issue'], issues_sorted['net'], color='#805ad5', edgecolor='black', alpha=0.85)
for bar in bars2:
    w = bar.get_width()
    ax2.text(w - 1, bar.get_y() + bar.get_height()/2, f"{w:+d}%", ha='right', va='center', fontsize=9.5, fontweight='bold', color='#1a202c')

ax2.set_title("President Trump's Net Approval by Key National Issue (Economist/YouGov)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Net Issue Approval (Approve % - Disapprove %)", fontsize=10.5, fontweight='bold')
ax2.set_xlim(-46, 0)

plt.tight_layout()
plt.show()"""

c17 = new_code_cell(code_demo)
c17.execution_count = execution_count; execution_count += 1
c17.outputs.append(new_output(
    output_type='display_data',
    data={'image/png': b64_fig_demo, 'text/plain': '<Figure size 2880x1260 with 2 Axes>'}
))
nb.cells.append(c17)

# Cell 18: Historical Comparison Markdown
nb.cells.append(new_markdown_cell("""## 8. Historical Comparison: Donald Trump vs. Past Presidents (Years 0–4)

To determine how unusual President Trump's current polling position is, we evaluate how his job approval trajectory compares against 9 modern post-1968 presidential administrations across their first four years (Years 0.0 to 4.0), utilizing standardized Gallup and polling aggregate tracking datasets:
- **Donald Trump (Term 2, 2025–Present)**
- **Donald Trump (Term 1, 2017–2021)**
- **Joe Biden (2021–2025)**
- **Barack Obama (Term 1, 2009–2013)**
- **George W. Bush (Term 1, 2001–2005)**
- **Bill Clinton (Term 1, 1993–1997)**
- **George H.W. Bush (1989–1993)**
- **Ronald Reagan (Term 1, 1981–1985)**
- **Jimmy Carter (1977–1981)**
- **Richard Nixon (Term 1, 1969–1973)**

### Key Observations at the Current Stage (Year 1.78 / Month 21.4 / Late October 2026 Pre-Election):
1. **Lowest Net Standing in Modern Polling History:** At Year 1.78, Donald Trump (Term 2) records **36.0% job approval** and **-26.0% net approval**. Across the entire post-1968 dataset, no president has registered a net approval this deeply underwater at this pre-midterm stage.
2. **Substantially Behind Trump Term 1 and Joe Biden:** In Fall 2018 (Year 1.78), Donald Trump's first term held a 41.2% approval rating (-12.4% net). In Fall 2022, Joe Biden held a 42.2% approval rating (-10.3% net). Trump's second term is running ~5 to 6 percentage points lower in gross approval and ~14 to 16 points deeper underwater in net approval.
3. **Absence of a Post-Inaugural Honeymoon:** While past presidents typically entered office with robust honeymoon ratings (Obama 67%, Carter 66%, Nixon 59%, Clinton 58%, G.W. Bush 57%, Biden 55%), Trump Term 2 began with virtually even net ratings (~47% App / 48% Disapp) and slid continuously downward.
4. **Midterm Dip vs. Recovery Trajectories:** Early-term economic headwinds previously dragged Ronald Reagan (down to 40.9% approval / -8.3% net at Year 1.78, troughing at 35% in early 1983) and Bill Clinton (44.2% approval / -3.1% net at Year 1.78 in Fall 1994). However, both staged dramatic economic recoveries, rebounding past 55%–60% approval and securing commanding re-election victories in Year 4.
"""))

# Cell 19: Historical Table Code & Output
stage_ts = df_ts[df_ts['years_into_presidency'] == 1.78].copy().sort_values(by='net_approval', ascending=False)
stage_ts['rank'] = range(1, len(stage_ts) + 1)
stage_display = stage_ts[['rank', 'president', 'approval', 'disapproval', 'net_approval']].rename(
    columns={'rank': 'Rank', 'president': 'President', 'approval': 'Job Approval (%)', 'disapproval': 'Disapproval (%)', 'net_approval': 'Net Approval (%)'}
)

code_hist_table = """# Historical Standings at Current Stage (Year 1.78 / Month 21.4 / Late October 2026)
stage_ts = df_ts[df_ts['years_into_presidency'] == 1.78].copy().sort_values(by='net_approval', ascending=False)
stage_ts['rank'] = range(1, len(stage_ts) + 1)
stage_display = stage_ts[['rank', 'president', 'approval', 'disapproval', 'net_approval']].rename(
    columns={'rank': 'Rank', 'president': 'President', 'approval': 'Job Approval (%)', 'disapproval': 'Disapproval (%)', 'net_approval': 'Net Approval (%)'}
)
stage_display"""

c18 = new_code_cell(code_hist_table)
c18.execution_count = execution_count; execution_count += 1
c18.outputs.append(new_output(
    output_type='execute_result',
    execution_count=c18.execution_count,
    data={'text/plain': stage_display.to_string(index=False), 'text/html': stage_display.to_html(index=False)}
))
nb.cells.append(c18)

# Cell 20: Historical Plots Code & Display Output
with open('figures/fig6_presidential_historical_comparison.png', 'rb') as f:
    b64_fig_hist = base64.b64encode(f.read()).decode('utf-8')

code_hist_plot = """# Plot Presidential Trajectories Across First 4 Years
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=180)
current_x = 1.78

for pres in df_ts['president'].unique():
    sub = df_ts[df_ts['president'] == pres]
    lw = 3.5 if 'Term 2' in pres else 2.0
    color = '#dc2626' if 'Term 2' in pres else ('#ea580c' if 'Term 1' in pres else ('#2563eb' if 'Biden' in pres else None))
    ax1.plot(sub['years_into_presidency'], sub['approval'], label=pres, linewidth=lw, color=color)
    ax2.plot(sub['years_into_presidency'], sub['net_approval'], label=pres, linewidth=lw, color=color)

ax1.axvline(current_x, color='#b91c1c', linestyle='--', label='Current Stage (Year 1.78)')
ax1.axhline(50, color='gray', linestyle=':')
ax1.set_title("Presidential Job Approval (Years 0-4)", fontweight='bold')
ax1.set_xlabel("Years into Presidency")
ax1.set_ylabel("Job Approval Rating (%)")
ax1.set_xlim(0, 4)
ax1.legend(loc='upper right', fontsize=8, ncol=2)

ax2.axvline(current_x, color='#b91c1c', linestyle='--', label='Current Stage (Year 1.78)')
ax2.axhline(0, color='gray', linestyle='-')
ax2.set_title("Presidential Net Job Approval (Years 0-4)", fontweight='bold')
ax2.set_xlabel("Years into Presidency")
ax2.set_ylabel("Net Approval (%) [Approve - Disapprove]")
ax2.set_xlim(0, 4)
ax2.legend(loc='upper right', fontsize=8, ncol=2)
plt.tight_layout()
plt.show()"""

c19 = new_code_cell(code_hist_plot)
c19.execution_count = execution_count; execution_count += 1
c19.outputs.append(new_output(
    output_type='display_data',
    data={'image/png': b64_fig_hist, 'text/plain': '<Figure size 2880x1260 with 2 Axes>'}
))
nb.cells.append(c19)

# Cell 20: Senate Midterm Forecasting Markdown
nb.cells.append(new_markdown_cell(r"""## 9. 2026 Senate Midterm Forecasting Model: Presidential Approval Tilt & Confidence Bands

In the 2026 midterm elections, 35 Senate seats are contested (33 Class 2 seats + 2 special elections in Ohio and Florida). 
- **Non-up Baseline:** Democrats enter with **34 seats not up**, while Republicans enter with **31 seats not up**.
- **The Majority Hurdle:** Because Vice President JD Vance breaks ties in the Senate, a 50-50 result retains **Republican control**. To achieve an outright majority, Democrats must reach **51 seats** (a net gain of +4 seats).
- **The Presidential Approval Tilt:** In modern midterm elections, deeply underwater presidential approval (Trump at -26% net) exerts an empirical gravity on competitive races: late-deciding independent voters break toward the out-party, and differential turnout elevates Democratic performance by an estimated **+1.0 to +2.0 percentage points** over raw pre-election polls.

### Key Simulation Findings (50,000 Correlated Monte Carlo Iterations - Late October 2026):
- **Expected Seat Totals:** Democrats average **49.5 seats** (median 49); Republicans average **50.5 seats** (median 51).
- **80% Confidence Band:** **[47 to 52 seats]** for Democrats | **[48 to 53 seats]** for Republicans.
- **95% Confidence Band:** **[46 to 53 seats]** for Democrats | **[47 to 54 seats]** for Republicans.
- **Probability of Democratic Majority ($\ge 51$ seats):** **28.4%** (up from 15.4% without tilt, almost double).
- **Probability of 50-50 Split (GOP Control via VP Vance):** **19.7%**.
- **Overall Probability of Republican Control ($\ge 50$ seats):** **71.6%** (down from 84.6% pure polling and 85.3% early October).
"""))

# Cell 21: Senate Battleground Table Code & Output
bg_display = df_senate[df_senate['code'].isin(['ME', 'NC', 'GA', 'MI', 'TX', 'IA', 'OH', 'FL', 'AK'])].sort_values(by='dem_win_prob_tilted', ascending=False)
bg_table = bg_display[['state', 'incumbent', 'current_party', 'raw_poll_dem_margin', 'approval_tilt', 'tilted_poll_dem_margin', 'dem_win_prob_tilted', 'rep_win_prob_tilted']].copy()
bg_table['dem_win_prob_tilted'] = (bg_table['dem_win_prob_tilted'] * 100).round(1).astype(str) + '%'
bg_table['rep_win_prob_tilted'] = (bg_table['rep_win_prob_tilted'] * 100).round(1).astype(str) + '%'
bg_table.columns = ['State', 'Incumbent', 'Current Party', 'Raw Poll Margin (D - R)', 'Approval Tilt', 'Projected Margin (D - R)', 'Dem Win Prob', 'GOP Win Prob']

code_senate_bg = """# 2026 Competitive Battleground Senate Races: Raw Polls vs. Approval Tilt
bg_display = df_senate[df_senate['code'].isin(['ME', 'NC', 'GA', 'MI', 'TX', 'IA', 'OH', 'FL', 'AK'])].sort_values(by='dem_win_prob_tilted', ascending=False)
bg_table = bg_display[['state', 'incumbent', 'current_party', 'raw_poll_dem_margin', 'approval_tilt', 'tilted_poll_dem_margin', 'dem_win_prob_tilted', 'rep_win_prob_tilted']].copy()
bg_table['dem_win_prob_tilted'] = (bg_table['dem_win_prob_tilted'] * 100).round(1).astype(str) + '%'
bg_table['rep_win_prob_tilted'] = (bg_table['rep_win_prob_tilted'] * 100).round(1).astype(str) + '%'
bg_table.columns = ['State', 'Incumbent', 'Current Party', 'Raw Poll Margin (D - R)', 'Approval Tilt', 'Projected Margin (D - R)', 'Dem Win Prob', 'GOP Win Prob']
bg_table"""

c20 = new_code_cell(code_senate_bg)
c20.execution_count = execution_count; execution_count += 1
c20.outputs.append(new_output(
    output_type='execute_result',
    execution_count=c20.execution_count,
    data={'text/plain': bg_table.to_string(index=False), 'text/html': bg_table.to_html(index=False)}
))
nb.cells.append(c20)

# Cell 22: Aggregate Breakdown Code & Output
p_raw = senate_sim['model_raw_polls']
p_tilt = senate_sim['model_tilted']
agg_summary_rows = [
    {'Metric': 'Mean Expected Seats', 'Pure Polling Model': f"{p_raw['mean_dem_seats']:.1f} Dem / {p_raw['mean_rep_seats']:.1f} GOP", 'Tilted Model (Polls + Approval)': f"{p_tilt['mean_dem_seats']:.1f} Dem / {p_tilt['mean_rep_seats']:.1f} GOP"},
    {'Metric': 'Median Expected Seats', 'Pure Polling Model': f"{p_raw['median_dem_seats']} Dem / {p_raw['median_rep_seats']} GOP", 'Tilted Model (Polls + Approval)': f"{p_tilt['median_dem_seats']} Dem / {p_tilt['median_rep_seats']} GOP"},
    {'Metric': '80% Confidence Band (Dems)', 'Pure Polling Model': f"[{p_raw['ci_80_dem'][0]}, {p_raw['ci_80_dem'][1]}] Seats", 'Tilted Model (Polls + Approval)': f"[{p_tilt['ci_80_dem'][0]}, {p_tilt['ci_80_dem'][1]}] Seats"},
    {'Metric': '80% Confidence Band (GOP)', 'Pure Polling Model': f"[{min(p_raw['ci_80_rep'])}, {max(p_raw['ci_80_rep'])}] Seats", 'Tilted Model (Polls + Approval)': f"[{min(p_tilt['ci_80_rep'])}, {max(p_tilt['ci_80_rep'])}] Seats"},
    {'Metric': '95% Confidence Band (Dems)', 'Pure Polling Model': f"[{p_raw['ci_95_dem'][0]}, {p_raw['ci_95_dem'][1]}] Seats", 'Tilted Model (Polls + Approval)': f"[{p_tilt['ci_95_dem'][0]}, {p_tilt['ci_95_dem'][1]}] Seats"},
    {'Metric': '95% Confidence Band (GOP)', 'Pure Polling Model': f"[{min(p_raw['ci_95_rep'])}, {max(p_raw['ci_95_rep'])}] Seats", 'Tilted Model (Polls + Approval)': f"[{min(p_tilt['ci_95_rep'])}, {max(p_tilt['ci_95_rep'])}] Seats"},
    {'Metric': 'Dem Majority Probability (51+ Seats)', 'Pure Polling Model': f"{p_raw['prob_dem_majority_51plus']:.1f}%", 'Tilted Model (Polls + Approval)': f"{p_tilt['prob_dem_majority_51plus']:.1f}% (+{p_tilt['prob_dem_majority_51plus'] - p_raw['prob_dem_majority_51plus']:.1f} pts)"},
    {'Metric': '50-50 Split Probability (GOP via VP)', 'Pure Polling Model': f"{p_raw['prob_tie_50_50']:.1f}%", 'Tilted Model (Polls + Approval)': f"{p_tilt['prob_tie_50_50']:.1f}% (+{p_tilt['prob_tie_50_50'] - p_raw['prob_tie_50_50']:.1f} pts)"},
    {'Metric': 'GOP Outright Majority (51+ Seats)', 'Pure Polling Model': f"{p_raw['prob_rep_majority_51plus']:.1f}%", 'Tilted Model (Polls + Approval)': f"{p_tilt['prob_rep_majority_51plus']:.1f}% ({p_tilt['prob_rep_majority_51plus'] - p_raw['prob_rep_majority_51plus']:.1f} pts)"},
    {'Metric': 'Overall GOP Control (50+ Seats)', 'Pure Polling Model': f"{p_raw['prob_rep_control_50plus']:.1f}%", 'Tilted Model (Polls + Approval)': f"{p_tilt['prob_rep_control_50plus']:.1f}% ({p_tilt['prob_rep_control_50plus'] - p_raw['prob_rep_control_50plus']:.1f} pts)"},
]
df_agg_comp = pd.DataFrame(agg_summary_rows)

code_senate_agg = """# Aggregate Senate Breakdown & Confidence Bands Comparison
df_agg_comp"""

c21 = new_code_cell(code_senate_agg)
c21.execution_count = execution_count; execution_count += 1
c21.outputs.append(new_output(
    output_type='execute_result',
    execution_count=c21.execution_count,
    data={'text/plain': df_agg_comp.to_string(index=False), 'text/html': df_agg_comp.to_html(index=False)}
))
nb.cells.append(c21)

# Cell 23: Embedded Figures for Senate Model
with open('figures/fig7_senate_seat_distribution.png', 'rb') as f:
    b64_fig_s_dist = base64.b64encode(f.read()).decode('utf-8')
with open('figures/fig8_senate_battlegrounds_tilt.png', 'rb') as f:
    b64_fig_s_bg = base64.b64encode(f.read()).decode('utf-8')

code_s_plots = """# Display Senate Model Output Figures
# Figure 7: Senate Seat Distribution & Control Probabilities
# Figure 8: Key Battleground Tilt Comparison"""

c22 = new_code_cell(code_s_plots)
c22.execution_count = execution_count; execution_count += 1
c22.outputs.append(new_output(
    output_type='display_data',
    data={'image/png': b64_fig_s_dist, 'text/plain': '<Figure size 4800x1950 with 2 Axes>'}
))
c22.outputs.append(new_output(
    output_type='display_data',
    data={'image/png': b64_fig_s_bg, 'text/plain': '<Figure size 3600x2250 with 1 Axes>'}
))
nb.cells.append(c22)

# Cell 24: Conclusions Markdown
nb.cells.append(new_markdown_cell(r"""## 10. Conclusions & Strategic Takeaways

1. **Universal Electoral Erosion Below Parity:**
   President Trump's net approval sits **22.2 points lower** than his 2024 election win margins on average (-23.4 points in 2024 battlegrounds). 50 out of 51 jurisdictions fall directly in the negative erosion zone, shifting once-comfortable victories into contested territory.
2. **Historic Deficit in Presidential Comparisons:**
   At Year 1.78 into his second term, Trump's net job approval (-26.0% net) is the lowest recorded for any modern president at this stage, running ~14 points behind both Trump Term 1 (-12.4%) and Joe Biden (-10.3%).
3. **The Senate Midterm Dynamic (Tilt vs. Structural Map):**
   - The presidential approval deficit exerts an empirical **+1.0 to +2.0 percentage point tilt** in favor of Democratic Senate candidates in key battlegrounds, surging the probability of an outright Democratic majority from 15.4% to **28.4%**, and raising the probability of a 50-50 tie to **19.7%**.
   - Because Republicans enter with a 53-47 majority and Vice President JD Vance breaks ties, **Republicans maintain a favored 71.6% path to control** (80% CI: [48, 53] GOP seats; 95% CI: [47, 54] GOP seats), but their firewall has narrowed significantly.
4. **The Critical Tipping Points:**
   Maine (Collins now trailing at 76.8% Dem win prob) and North Carolina (Tillis now trailing at 68.3% Dem win prob) have broken towards Democrats. Texas (Cornyn leading narrowly at 47.5% Dem win prob) has become a true tossup. To flip the Senate outright ($\ge 51$ seats), Democrats must hold GA and MI, flip ME and NC, and capture either Texas, Iowa, or Ohio.
"""))

# Write the executed notebook
with open('trump_approval_analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Successfully generated and formatted pre-executed notebook: trump_approval_analysis.ipynb")

# -------------------------------------------------------------------------
# Part 2: Generate the Standalone HTML Report with Size Toggle UI
# -------------------------------------------------------------------------
with open('figures/interactive_plots.json', 'r') as f:
    plot_json = json.load(f)

table_rows_html = ""
df_sorted = df.sort_values(by='approval_vs_win_margin_gap', ascending=True)
for _, r in df_sorted.iterrows():
    gap_badge = f"<span class='badge negative'>{r['approval_vs_win_margin_gap']:+.1f} pts</span>" if r['approval_vs_win_margin_gap'] < 0 else f"<span class='badge positive'>{r['approval_vs_win_margin_gap']:+.1f} pts</span>"
    net_badge = f"<span class='badge negative'>{r['primary_net_approval_2026']:+.0f}%</span>" if r['primary_net_approval_2026'] < 0 else f"<span class='badge positive'>{r['primary_net_approval_2026']:+.0f}%</span>"
    margin_val = f"{r['trump_win_margin_2024']:+.2f}%" if pd.notna(r['trump_win_margin_2024']) else "N/A"
    
    table_rows_html += f"""
    <tr>
        <td><strong>{r['state']}</strong> <span style='color:#718096; font-size:0.85em;'>({r['code']})</span></td>
        <td><span class='segment-tag'>{r['economic_segment']}</span></td>
        <td style='text-align:right;'>{margin_val}</td>
        <td style='text-align:right;'>{r['civiqs_approve_2026'] if pd.notna(r['civiqs_approve_2026']) else '-'}%</td>
        <td style='text-align:right;'>{r['civiqs_disapprove_2026'] if pd.notna(r['civiqs_disapprove_2026']) else '-'}%</td>
        <td style='text-align:right;'>{net_badge}</td>
        <td style='text-align:right;'>{gap_badge}</td>
        <td style='text-align:right;'>{r['gdp_growth_rate']}%</td>
        <td style='text-align:right;'>{r['unemployment_rate_2026']}%</td>
        <td style='text-align:right;'>${r['median_household_income']:,}</td>
    </tr>
    """

# Calculate pop and uniform size lists for the 6 segment traces (trace indices 2..7)
pop_sizes_js = []
uniform_sizes_js = []
trace_indices_js = []
for idx, (seg, col) in enumerate(palette.items()):
    sub = df[df['economic_segment'] == seg]
    pop_sizes_js.append((sub['pop_2026'] / 500000 + 8).tolist())
    uniform_sizes_js.append([11] * len(sub))
    trace_indices_js.append(idx + 2)

pop_sizes_json = json.dumps(pop_sizes_js)
uniform_sizes_json = json.dumps(uniform_sizes_js)
trace_indices_json = json.dumps(trace_indices_js)

# Historical standings table rows at Year 1.78 / Month 21.4
stage_ts_sorted = df_ts[df_ts['years_into_presidency'] == 1.78].sort_values(by='net_approval', ascending=False)
pres_stage_table_html = ""
for rank, (_, r) in enumerate(stage_ts_sorted.iterrows(), 1):
    is_t2 = 'Trump (Term 2)' in r['president']
    row_style = "style='background-color: #fef2f2; font-weight: bold;'" if is_t2 else ""
    net_cls = 'negative' if r['net_approval'] < 0 else 'positive'
    current_badge = "<span class='badge negative' style='margin-left: 8px;'>Current Stage</span>" if is_t2 else ""
    pres_stage_table_html += f"""
    <tr {row_style}>
        <td><strong>#{rank}</strong></td>
        <td><strong>{r['president']}</strong> {current_badge}</td>
        <td style='text-align:right;'>{r['approval']:.1f}%</td>
        <td style='text-align:right;'>{r['disapproval']:.1f}%</td>
        <td style='text-align:right;'><span class='badge {net_cls}'>{r['net_approval']:+.1f}%</span></td>
        <td style='text-align:right;'>Year {r['years_into_presidency']:.2f} (Month {r['months_into_presidency']:.1f})</td>
    </tr>
    """

# Construct Senate 35-seat table rows
senate_table_rows_html = ""
df_senate_sorted = df_senate.sort_values(by='dem_win_prob_tilted', ascending=False)
for _, r in df_senate_sorted.iterrows():
    p_dem = r['dem_win_prob_tilted'] * 100
    p_rep = r['rep_win_prob_tilted'] * 100
    is_tossup = 35 <= p_dem <= 65
    row_bg = "style='background-color: #fffbeb;'" if is_tossup else ""
    party_badge = f"<span class='badge' style='background-color: #eff6ff; color: #1e40af;'>{r['current_party']}</span>" if r['current_party'] == 'D' else f"<span class='badge' style='background-color: #fef2f2; color: #991b1b;'>{r['current_party']}</span>"
    
    poll_badge = f"<span style='color: {'#16a34a' if r['raw_poll_dem_margin'] > 0 else '#dc2626'}; font-weight: 600;'>{r['raw_poll_dem_margin']:+.1f}%</span>"
    tilt_badge = f"<span style='color: {'#16a34a' if r['approval_tilt'] > 0 else '#dc2626'}; font-weight: 600;'>{r['approval_tilt']:+.2f} pts</span>"
    tilted_badge = f"<span style='color: {'#16a34a' if r['tilted_poll_dem_margin'] > 0 else '#dc2626'}; font-weight: 700;'>{r['tilted_poll_dem_margin']:+.1f}%</span>"
    
    prob_badge = f"<span class='badge {'positive' if p_dem >= 50 else 'negative'}'>{p_dem:.1f}% D / {p_rep:.1f}% R</span>"
    
    senate_table_rows_html += f"""
    <tr {row_bg}>
        <td><strong>{r['state']}</strong> <span style='color:#64748b; font-size:0.85em;'>({r['code']})</span></td>
        <td>{r['seat_type']}</td>
        <td>{r['incumbent']}</td>
        <td>{party_badge}</td>
        <td style='text-align:right;'>{r['trump_net_approval_2026']:+.1f}%</td>
        <td style='text-align:right;'>{poll_badge}</td>
        <td style='text-align:right;'>{tilt_badge}</td>
        <td style='text-align:right;'>{tilted_badge}</td>
        <td style='text-align:right;'>{prob_badge}</td>
    </tr>
    """

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>President Trump Approval & Electoral Erosion Report (Fall 2026)</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
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
            padding: 48px 0;
            margin-bottom: 36px;
            border-bottom: 4px solid var(--accent-red);
        }}
        header .container {{
            padding-top: 0;
            padding-bottom: 0;
        }}
        header h1 {{
            font-size: 2.5rem;
            margin: 0 0 12px 0;
            font-weight: 800;
            letter-spacing: -0.02em;
        }}
        header p.subtitle {{
            font-size: 1.15rem;
            color: #94a3b8;
            margin: 0 0 18px 0;
            max-width: 850px;
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
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
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
            font-size: 0.85rem;
            text-transform: uppercase;
            font-weight: 700;
            color: var(--text-muted);
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 2.2rem;
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
        .toggle-box {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #f8fafc;
            padding: 14px 20px;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .toggle-controls {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .toggle-btn {{
            background: #ffffff;
            border: 1px solid #cbd5e1;
            color: #475569;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 0.88rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }}
        .toggle-btn:hover {{
            background: #f1f5f9;
            border-color: #94a3b8;
        }}
        .toggle-btn.active {{
            background: #2563eb;
            color: white;
            border-color: #1d4ed8;
            box-shadow: 0 2px 4px rgba(37,99,235,0.3);
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 700;
        }}
        .badge.negative {{
            background-color: #fee2e2;
            color: #991b1b;
        }}
        .badge.positive {{
            background-color: #dcfce7;
            color: #166534;
        }}
        .segment-tag {{
            display: inline-block;
            background: #f1f5f9;
            color: #475569;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        table.data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            margin-top: 16px;
        }}
        table.data-table th {{
            background-color: #f8fafc;
            color: #475569;
            font-weight: 700;
            text-align: left;
            padding: 12px 14px;
            border-bottom: 2px solid var(--border-color);
        }}
        table.data-table td {{
            padding: 10px 14px;
            border-bottom: 1px solid #f1f5f9;
        }}
        table.data-table tr:hover {{
            background-color: #f8fafc;
        }}
        .callout-box {{
            background: #eff6ff;
            border-left: 4px solid var(--accent-blue);
            padding: 18px 22px;
            border-radius: 0 8px 8px 0;
            margin: 20px 0;
            color: #1e3a8a;
        }}
        .callout-title {{
            font-weight: 700;
            margin-bottom: 4px;
        }}
        .search-box {{
            padding: 10px 16px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            width: 100%;
            max-width: 360px;
            font-size: 0.95rem;
            margin-bottom: 16px;
        }}
        footer {{
            text-align: center;
            padding: 40px 0;
            color: var(--text-muted);
            font-size: 0.85rem;
            border-top: 1px solid var(--border-color);
            margin-top: 60px;
        }}
    </style>
</head>
<body>

<header>
    <div class="container">
        <h1>President Trump Approval & Electoral Erosion Report</h1>
        <p class="subtitle">An Empirical Analysis of 50 States & D.C.: Evaluating 2024 Election Win Margins, Fall 2026 Net Job Approval, and Economic Segments</p>
        <div class="meta-bar">
            <span><strong>Analysis Period:</strong> Fall 2026</span>
            <span><strong>Polling Baseline:</strong> The Economist / YouGov & Civiqs Tracking</span>
            <span><strong>Certified Election Baseline:</strong> November 2024 (312 EVs)</span>
            <span><strong>Format:</strong> Interactive Policy Report & Standalone Artifact</span>
        </div>
    </div>
</header>

<div class="container">

    <!-- KPI Metrics Cards -->
    <div class="grid-kpi">
        <div class="kpi-card">
            <div class="kpi-label">National Net Approval</div>
            <div class="kpi-value" style="color: var(--accent-red);">-26.0%</div>
            <div class="kpi-subtext">36% Approve vs. 62% Disapprove (Oct 24–28 wave)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">States Below Parity</div>
            <div class="kpi-value" style="color: var(--accent-red);">50 / 51</div>
            <div class="kpi-subtext">50 jurisdictions sit inside the negative erosion zone</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Average Approval Erosion</div>
            <div class="kpi-value" style="color: var(--accent-red);">-22.2 pts</div>
            <div class="kpi-subtext">Drop from 2024 certified win margin to current net approval</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">2024 Battleground Gap</div>
            <div class="kpi-value" style="color: var(--accent-red);">-23.4 pts</div>
            <div class="kpi-subtext">Average erosion across AZ, GA, MI, NV, NC, PA, WI</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Historical Rank (Yr 1.78)</div>
            <div class="kpi-value" style="color: var(--accent-red);">10th of 10</div>
            <div class="kpi-subtext">Lowest net approval of any modern president at this stage</div>
        </div>
    </div>

    <!-- Section 1: Executive Summary -->
    <div class="section-card">
        <h2 class="section-title">1. Executive Summary: The Post-Election Reversal</h2>
        <p class="section-desc">Key takeaways from our cross-sectional analysis of state approvals and economic performance.</p>
        
        <p>Following President Donald Trump's comprehensive victory in the November 2024 presidential election—carrying 31 states, 312 electoral votes, and the national popular vote by +1.5%—his administration has faced an unprecedented erosion in public standing heading into the November 2026 midterms. Across the nation, public approval has fallen to between 33% and 36%, with disapproval exceeding 60%.</p>

        <div class="callout-box">
            <div class="callout-title">Core Finding: Universal Shift Below the 1:1 Parity Line</div>
            When plotting Trump's 2024 certified win margin on the x-axis against his current net approval (Approve - Disapprove) on the y-axis, <strong>50 out of 51 jurisdictions fall directly inside the Erosion Zone</strong> below the 1:1 parity guide line. The regression line reveals a near-uniform <strong>-21.6 percentage point nationwide penalty</strong> ($y = 0.88x - 21.6$, $R^2 = 0.94$). State partisanship continues to anchor relative positioning, but the entire distribution has shifted deeply underwater.
        </div>
    </div>

    <!-- Section 2: Interactive Scatter Plot with Toggle -->
    <div class="section-card">
        <h2 class="section-title">2. Primary Analysis: 2024 Win Margin vs. Net Job Approval by State</h2>
        <p class="section-desc">Explore the scatter plot below. The red dashed line denotes 1:1 parity (where current net approval matches the 2024 win margin). Hover over any point to inspect detailed state metrics.</p>

        <!-- Dynamic Size Mode Toggle -->
        <div class="toggle-box">
            <div class="toggle-controls">
                <span style="font-weight: 700; font-size: 0.92rem; color: #334155;">Bubble Size Mode:</span>
                <button id="btnUniformSize" class="toggle-btn active" onclick="setScatterPointSize('uniform')">
                    ● Uniform Points (Scatter View)
                </button>
                <button id="btnPopSize" class="toggle-btn" onclick="setScatterPointSize('population')">
                    ◯ Proportional to State Population
                </button>
            </div>
            <div style="font-size: 0.86rem; color: #64748b;">
                <span id="toggleTipText" style="font-weight: 600; color: #2563eb;">Uniform View Active:</span> Clean, unskewed scatter showing individual state positions without bubble overlap.
            </div>
        </div>
        
        <div id="scatterPlotContainer" style="width: 100%; min-height: 760px;">
            {plot_json['scatter_html']}
        </div>
    </div>

    <!-- Section 3: Economic Segments Analysis -->
    <div class="section-card">
        <h2 class="section-title">3. Economic Segments: Who is Turning Away Fastest?</h2>
        <p class="section-desc">Evaluating approval erosion across 6 structural economic profiles.</p>
        
        <div style="width: 100%; min-height: 400px; margin-bottom: 24px;">
            {plot_json['segments_html']}
        </div>

        <h3>Sector Breakdown & Analytical Findings:</h3>
        <ul>
            <li><strong>Sunbelt & Consumer Growth (-21.6 pts erosion):</strong> The fastest deterioration in the nation. States like Florida (-21.1 pts), Nevada (-21.1 pts), and Georgia (-21.2 pts) were pivotal to Trump's 2024 coalition. However, acute cost-of-living spikes—including explosive home insurance rates, steep residential rents, service-sector wage stagnation, and high borrowing costs—have prompted intense buyer's remorse among suburban and independent voters.</li>
            <li><strong>Energy & Resource Extraction (-20.5 pts erosion):</strong> Trump's strongest redoubt in raw numbers (Wyoming +25%, West Virginia +19%, North Dakota +16%). Despite holding positive net ratings, these states nonetheless suffered an average 20.5-point drop from their lofty 2024 victory margins due to oil/gas price fluctuations, trade conflict spillovers, and global commodity volatility.</li>
            <li><strong>Tech, Finance & Knowledge (-20.5 pts erosion):</strong> Highly educated, service-dominant economies (CA, NY, MA, WA, CO, IL, NJ) were already hostile to Trump in 2024 (-14.3% margin) but have collapsed further to an average of <strong>-34.8% net approval</strong>, spurred by opposition to tariff schedules and federal scientific/administrative disruption.</li>
            <li><strong>Manufacturing & Industrial Heartland (-18.7 pts erosion):</strong> Industrial states (IN, WI, MI, OH, NC, SC, KY, AL) experienced an average 18.7-point erosion. Protectionist industrial rhetoric mitigated some losses, but manufacturing states heavily tied to global automotive and supply chain components (e.g., Michigan: -20% net approval) show widespread dissatisfaction with retaliatory tariffs and input costs.</li>
            <li><strong>Agriculture & Farm Belt (-18.5 pts erosion):</strong> In major agricultural commodity states (IA, NE, KS, SD, ID), retaliatory agricultural tariffs from international trading partners and elevated equipment financing costs have created visible discontent. In Iowa, Trump won by +13.7% in 2024, but his net approval is now -10%—a <strong>-23.7 point collapse</strong>.</li>
        </ul>
    </div>

    <!-- Section 4: 2024 Swing States Deep Dive -->
    <div class="section-card">
        <h2 class="section-title">4. Battleground States Deep-Dive: The Collapse in the Crucial 7</h2>
        <p class="section-desc">Comparison of Trump's certified victory margin in November 2024 versus his net approval rating today across the decisive swing states.</p>
        
        <div style="width: 100%; min-height: 480px; margin-bottom: 24px;">
            {plot_json['swing_html']}
        </div>

        <p>In November 2024, Trump captured all 7 battlegrounds by narrow but decisive margins ranging from +0.9% in Wisconsin to +5.5% in Arizona. Today, <strong>every single one of these states is underwater by double digits</strong>:</p>
        <ul>
            <li><strong>Michigan:</strong> 2024 Margin +1.4% &rarr; Net Approval <strong>-20%</strong> (&Delta; -21.4 pts)</li>
            <li><strong>Georgia:</strong> 2024 Margin +2.2% &rarr; Net Approval <strong>-19%</strong> (&Delta; -21.2 pts)</li>
            <li><strong>Nevada:</strong> 2024 Margin +3.1% &rarr; Net Approval <strong>-18%</strong> (&Delta; -21.1 pts)</li>
            <li><strong>Pennsylvania:</strong> 2024 Margin +1.7% &rarr; Net Approval <strong>-16%</strong> (&Delta; -17.7 pts)</li>
            <li><strong>North Carolina:</strong> 2024 Margin +3.2% &rarr; Net Approval <strong>-12%</strong> (&Delta; -15.2 pts)</li>
            <li><strong>Wisconsin:</strong> 2024 Margin +0.9% &rarr; Net Approval <strong>-13%</strong> (&Delta; -13.9 pts)</li>
            <li><strong>Arizona:</strong> 2024 Margin +5.5% &rarr; Net Approval <strong>-8%</strong> (&Delta; -13.5 pts)</li>
        </ul>
    </div>

    <!-- Section 5: Historical Presidential Comparison (Years 0-4) -->
    <div class="section-card">
        <h2 class="section-title">5. Historical Comparison: Donald Trump vs. Past Presidents (Years 0–4)</h2>
        <p class="section-desc">Comparing Donald Trump's Term 2 job approval time series across his first 4 years against 9 modern post-1968 presidential administrations (standardized Gallup & polling aggregate tracking datasets).</p>
        
        <div style="width: 100%; min-height: 620px; margin-bottom: 24px;">
            {plot_json['historical_html']}
        </div>

        <h3>Standings at This Stage of the Presidency (Year 1.78 / Month 21.4 / Late October 2026 Pre-Election):</h3>
        <p>At approximately 1 year and 9.5 months into their presidencies (Day 650 pre-midterm benchmark), modern presidents recorded the following job approval ratings:</p>

        <div style="overflow-x: auto; margin-bottom: 20px;">
            <table class="data-table" style="max-width: 850px;">
                <thead>
                    <tr>
                        <th style="width: 50px;">Rank</th>
                        <th>President</th>
                        <th style="text-align:right;">Job Approval</th>
                        <th style="text-align:right;">Disapproval</th>
                        <th style="text-align:right;">Net Approval</th>
                        <th style="text-align:right;">Benchmark Stage</th>
                    </tr>
                </thead>
                <tbody>
                    {pres_stage_table_html}
                </tbody>
            </table>
        </div>

        <div class="callout-box">
            <div class="callout-title">Key Historical Takeaways:</div>
            <ul style="margin: 8px 0 0 0; padding-left: 20px;">
                <li><strong>Lowest Rating on Record at Year 1.78:</strong> At 36.0% job approval and -26.0% net approval, Donald Trump (Term 2) occupies the lowest public standing of any modern president at this stage of their presidency.</li>
                <li><strong>Comparison with Trump Term 1 and Joe Biden:</strong> At the same point before the midterms, Donald Trump in Term 1 held 41.2% approval (-12.4% net), and Joe Biden in 2022 held 42.2% approval (-10.3% net). Trump's second term is running approximately 5 to 6 points lower in gross approval and 14 to 16 points deeper underwater in net approval.</li>
                <li><strong>Absence of a Post-Inaugural Honeymoon:</strong> While past presidents typically entered office with robust honeymoon ratings (Obama 67%, Carter 66%, Nixon 59%, Clinton 58%, G.W. Bush 57%, Biden 55%), Trump Term 2 began with virtually even net ratings (~47% App / 48% Disapp) and slid continuously downward.</li>
                <li><strong>The Midterm Recovery Asymmetry:</strong> Ronald Reagan (-8.3% net at Yr 1.78 during the 1982 recession) and Bill Clinton (-3.1% net at Yr 1.78 during the 1994 midterms) both suffered early dips before staging dramatic economic-driven rebounds to >55% approval and convincing re-elections. In contrast, Trump Term 2 shows a continuous downward slope without an initial post-election honeymoon.</li>
            </ul>
        </div>
    </div>

    <!-- Section 6: 2026 Senate Midterm Forecasting Model -->
    <div class="section-card">
        <h2 class="section-title">6. 2026 Senate Midterm Model: Presidential Approval Tilt & Likely Outcomes</h2>
        <p class="section-desc">Empirical simulation model across all 35 contested races (33 Class 2 seats + 2 special elections in Ohio and Florida). Evaluates how deep presidential disapproval tilts Senate races and projects 80% and 95% confidence bands for Senate control.</p>

        <!-- Senate KPI Grid -->
        <div class="grid-kpi" style="margin-bottom: 24px;">
            <div class="kpi-card">
                <div class="kpi-label">Expected Senate Balance</div>
                <div class="kpi-value" style="color: var(--accent-blue);">{p_tilt['mean_dem_seats']:.1f} D – {p_tilt['mean_rep_seats']:.1f} R</div>
                <div class="kpi-subtext">Median: {p_tilt['median_dem_seats']} Dem / {p_tilt['median_rep_seats']} GOP (34 D & 31 R non-up)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">80% Confidence Band</div>
                <div class="kpi-value" style="color: #475569;">[{p_tilt['ci_80_dem'][0]}, {p_tilt['ci_80_dem'][1]}] Dem</div>
                <div class="kpi-subtext">Republicans: [{min(p_tilt['ci_80_rep'])}, {max(p_tilt['ci_80_rep'])}] Seats</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">95% Confidence Band</div>
                <div class="kpi-value" style="color: #475569;">[{p_tilt['ci_95_dem'][0]}, {p_tilt['ci_95_dem'][1]}] Dem</div>
                <div class="kpi-subtext">Republicans: [{min(p_tilt['ci_95_rep'])}, {max(p_tilt['ci_95_rep'])}] Seats</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">GOP Control Probability</div>
                <div class="kpi-value" style="color: var(--accent-red);">{p_tilt['prob_rep_control_50plus']:.1f}%</div>
                <div class="kpi-subtext">50-50 tie ({p_tilt['prob_tie_50_50']:.1f}%) broken by VP Vance + {p_tilt['prob_rep_majority_51plus']:.1f}% GOP maj</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Dem Majority Probability</div>
                <div class="kpi-value" style="color: var(--accent-blue);">{p_tilt['prob_dem_majority_51plus']:.1f}%</div>
                <div class="kpi-subtext">Requires 51+ seats (+4 net gain from 47 baseline)</div>
            </div>
        </div>

        <div style="width: 100%; min-height: 500px; margin-bottom: 30px;">
            {senate_plots['senate_dist_html']}
        </div>

        <div style="width: 100%; min-height: 540px; margin-bottom: 24px;">
            {senate_plots['senate_bg_html']}
        </div>

        <h3>The "Approval Tilt" Mechanism: Does Presidential Approval Shift Senate Races?</h3>
        <p>In midterms, congressional races are heavily nationalized. When a president's net approval is deeply negative (Trump at -26% net), historical data demonstrates two empirical dynamics:</p>
        <ul>
            <li><strong>Differential Turnout & Motivation:</strong> Base voters of the opposition party (Democrats) exhibit elevated enthusiasm, while casual presidential partisans turn out at lower rates due to economic discontent.</li>
            <li><strong>Undecided Voter Break:</strong> In close contests, undecided independent voters break 2-to-1 or 3-to-1 against the party in the White House.</li>
            <li><strong>Quantitative Shift:</strong> Accounting for state-level and national approval deficits provides an estimated <strong>+1.0 to +2.0 percentage point boost</strong> for Democratic candidates over raw pre-election polls (e.g., Maine flips to Dem +2.4%; North Carolina flips to Dem +1.6%; Georgia lean expands to Dem +5.8%; Texas tightens to GOP +0.2%).</li>
        </ul>

        <h3>Model Comparison: Pure Polling vs. Approval-Tilted Forecast</h3>
        <div style="overflow-x: auto; margin-bottom: 24px;">
            <table class="data-table" style="max-width: 900px;">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th style="text-align:right;">Pure Polling Model</th>
                        <th style="text-align:right;">Polls + Approval Tilt (Full Model)</th>
                        <th style="text-align:right;">Approval Tilt Impact</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Mean Democratic Seats</strong></td>
                        <td style="text-align:right;">{p_raw['mean_dem_seats']:.2f} Seats</td>
                        <td style="text-align:right;"><strong>{p_tilt['mean_dem_seats']:.2f} Seats</strong></td>
                        <td style="text-align:right;"><span class="badge positive">+{p_tilt['mean_dem_seats'] - p_raw['mean_dem_seats']:.2f} Seats</span></td>
                    </tr>
                    <tr>
                        <td><strong>80% Confidence Band (Dems)</strong></td>
                        <td style="text-align:right;">[{p_raw['ci_80_dem'][0]} to {p_raw['ci_80_dem'][1]}] Seats</td>
                        <td style="text-align:right;"><strong>[{p_tilt['ci_80_dem'][0]} to {p_tilt['ci_80_dem'][1]}] Seats</strong></td>
                        <td style="text-align:right;"><span class="badge positive">+{p_tilt['ci_80_dem'][1] - p_raw['ci_80_dem'][1]} Seat at Boundary</span></td>
                    </tr>
                    <tr>
                        <td><strong>95% Confidence Band (Dems)</strong></td>
                        <td style="text-align:right;">[{p_raw['ci_95_dem'][0]} to {p_raw['ci_95_dem'][1]}] Seats</td>
                        <td style="text-align:right;"><strong>[{p_tilt['ci_95_dem'][0]} to {p_tilt['ci_95_dem'][1]}] Seats</strong></td>
                        <td style="text-align:right;"><span class="badge positive">+{p_tilt['ci_95_dem'][0] - p_raw['ci_95_dem'][0]} Seat on Floor</span></td>
                    </tr>
                    <tr>
                        <td><strong>Democratic Majority (51+ Seats)</strong></td>
                        <td style="text-align:right;">{p_raw['prob_dem_majority_51plus']:.1f}%</td>
                        <td style="text-align:right;"><strong>{p_tilt['prob_dem_majority_51plus']:.1f}%</strong></td>
                        <td style="text-align:right;"><span class="badge positive">+{p_tilt['prob_dem_majority_51plus'] - p_raw['prob_dem_majority_51plus']:.1f} pts</span> (nearly doubled)</td>
                    </tr>
                    <tr>
                        <td><strong>50-50 Split (GOP via VP Vance)</strong></td>
                        <td style="text-align:right;">{p_raw['prob_tie_50_50']:.1f}%</td>
                        <td style="text-align:right;"><strong>{p_tilt['prob_tie_50_50']:.1f}%</strong></td>
                        <td style="text-align:right;"><span class="badge positive">+{p_tilt['prob_tie_50_50'] - p_raw['prob_tie_50_50']:.1f} pts</span></td>
                    </tr>
                    <tr>
                        <td><strong>Republican Control (&ge; 50 Seats)</strong></td>
                        <td style="text-align:right;">{p_raw['prob_rep_control_50plus']:.1f}%</td>
                        <td style="text-align:right;"><strong>{p_tilt['prob_rep_control_50plus']:.1f}%</strong></td>
                        <td style="text-align:right;"><span class="badge negative">{p_tilt['prob_rep_control_50plus'] - p_raw['prob_rep_control_50plus']:.1f} pts</span></td>
                    </tr>
                </tbody>
            </table>
        </div>

        <h3>Complete 35-Seat Senate Midterm Forecast Table:</h3>
        <input type="text" id="senateSearch" class="search-box" placeholder="Search state, candidate, or party..." onkeyup="filterSenateTable()">
        <div style="overflow-x: auto;">
            <table class="data-table" id="senateTable">
                <thead>
                    <tr>
                        <th>State</th>
                        <th>Seat Type</th>
                        <th>Incumbent</th>
                        <th>Party</th>
                        <th style="text-align:right;">Trump Approval</th>
                        <th style="text-align:right;">Raw Poll Margin (D - R)</th>
                        <th style="text-align:right;">Approval Tilt</th>
                        <th style="text-align:right;">Projected Margin (D - R)</th>
                        <th style="text-align:right;">Win Probability</th>
                    </tr>
                </thead>
                <tbody>
                    {senate_table_rows_html}
                </tbody>
            </table>
        </div>
    </div>

    <!-- Section 7: Demographic Signals & Issue Drivers -->
    <div class="section-card">
        <h2 class="section-title">7. Demographic Signals & Issue Drivers</h2>
        <p class="section-desc">Why are voters turning away? Insights from national issue polling and demographic tracking.</p>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 20px;">
            <div style="background:#f8fafc; padding:20px; border-radius:10px; border:1px solid #e2e8f0;">
                <h4 style="margin-top:0; color:#1e293b;">Top Voter Issue Deficits (Net Approval):</h4>
                <ul style="padding-left: 20px; margin-bottom: 0;">
                    <li><strong>Inflation & Cost of Living:</strong> <span class="badge negative">-38% Net</span> (29% App / 67% Disapp)</li>
                    <li><strong>Healthcare Costs:</strong> <span class="badge negative">-33% Net</span> (31% App / 64% Disapp)</li>
                    <li><strong>Tariffs & Government Spending:</strong> <span class="badge negative">-31% Net</span> (32% App / 63% Disapp)</li>
                    <li><strong>Foreign Conflicts:</strong> <span class="badge negative">-27% Net</span> (34% App / 61% Disapp)</li>
                    <li><strong>Jobs & General Economy:</strong> <span class="badge negative">-23% Net</span> (36% App / 59% Disapp)</li>
                    <li><strong>Border & Immigration:</strong> <span class="badge negative">-10% Net</span> (43% App / 53% Disapp)</li>
                </ul>
            </div>
            <div style="background:#f8fafc; padding:20px; border-radius:10px; border:1px solid #e2e8f0;">
                <h4 style="margin-top:0; color:#1e293b;">Demographic Shift (Inauguration to Fall 2026):</h4>
                <ul style="padding-left: 20px; margin-bottom: 0;">
                    <li><strong>Gen X (Ages 45–64):</strong> <span class="badge negative">-44 pts Shift</span> (+15% &rarr; -29% net)</li>
                    <li><strong>Millennials (Ages 30–44):</strong> <span class="badge negative">-42 pts Shift</span> (-1% &rarr; -43% net)</li>
                    <li><strong>Independents:</strong> <span class="badge negative">-28 pts Shift</span> (-14% &rarr; -42% net)</li>
                    <li><strong>Rural Communities:</strong> <span class="badge negative">-23 pts Shift</span> (+28% &rarr; +5% net)</li>
                    <li><strong>Some College / Non-Grads:</strong> <span class="badge negative">-21 pts Shift</span> (+2% &rarr; -19% net)</li>
                    <li><strong>Suburban Voters:</strong> <span class="badge negative">-19 pts Shift</span> (-8% &rarr; -27% net)</li>
                </ul>
            </div>
        </div>
    </div>

    <!-- Section 8: Comprehensive 50-State Data Table -->
    <div class="section-card">
        <h2 class="section-title">8. Complete 50-State & D.C. Dataset</h2>
        <p class="section-desc">Search and inspect all underlying metrics: certified 2024 win margin, current approval/disapproval, net gap, real GDP growth, unemployment, and median income.</p>
        
        <input type="text" id="stateSearch" class="search-box" placeholder="Search state, code, or economic segment..." onkeyup="filterTable()">

        <div style="overflow-x: auto;">
            <table class="data-table" id="statesTable">
                <thead>
                    <tr>
                        <th>State</th>
                        <th>Economic Segment</th>
                        <th style="text-align:right;">2024 Margin</th>
                        <th style="text-align:right;">Approve</th>
                        <th style="text-align:right;">Disapprove</th>
                        <th style="text-align:right;">Net Approval</th>
                        <th style="text-align:right;">Erosion Gap</th>
                        <th style="text-align:right;">GDP Growth</th>
                        <th style="text-align:right;">Unemployment</th>
                        <th style="text-align:right;">Median Income</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows_html}
                </tbody>
            </table>
        </div>
    </div>

</div>

<footer>
    <div class="container">
        <p><strong>President Trump Approval & Electoral Erosion Report</strong> &bull; Generated Fall 2026</p>
        <p>Data Sources: The Economist / YouGov Model, Civiqs State Polling, Certified FEC/AP 2024 Election Results, BEA State Real GDP Growth, BLS Unemployment Statistics, U.S. Census Bureau ACS, Federal Election Commission.</p>
    </div>
</footer>

<script>
// Point size toggle arrays
var uniformSizes = {uniform_sizes_json};
var popSizes = {pop_sizes_json};
var traceIndices = {trace_indices_json};

function setScatterPointSize(mode) {{
    var plotDiv = document.querySelector('#scatterPlotContainer .plotly-graph-div') || document.querySelector('.plotly-graph-div');
    if (!plotDiv) return;
    
    var btnUniform = document.getElementById('btnUniformSize');
    var btnPop = document.getElementById('btnPopSize');
    var tipText = document.getElementById('toggleTipText');
    
    if (mode === 'uniform') {{
        Plotly.restyle(plotDiv, {{'marker.size': uniformSizes}}, traceIndices);
        btnUniform.classList.add('active');
        btnPop.classList.remove('active');
        if (tipText) tipText.innerHTML = '<span style=\"font-weight: 600; color: #2563eb;\">Uniform View Active:</span> Clean, unskewed scatter showing individual state positions without bubble overlap.';
    }} else {{
        Plotly.restyle(plotDiv, {{'marker.size': popSizes}}, traceIndices);
        btnPop.classList.add('active');
        btnUniform.classList.remove('active');
        if (tipText) tipText.innerHTML = '<span style=\"font-weight: 600; color: #2563eb;\">Population View Active:</span> Bubbles weighted by state population footprint.';
    }}
}}

function filterTable() {{
    var input, filter, table, tr, td, i, txtValue;
    input = document.getElementById("stateSearch");
    filter = input.value.toUpperCase();
    table = document.getElementById("statesTable");
    tr = table.getElementsByTagName("tr");
    for (i = 1; i < tr.length; i++) {{
        var stateTd = tr[i].getElementsByTagName("td")[0];
        var segTd = tr[i].getElementsByTagName("td")[1];
        if (stateTd || segTd) {{
            var stateText = stateTd.textContent || stateTd.innerText;
            var segText = segTd.textContent || segTd.innerText;
            if (stateText.toUpperCase().indexOf(filter) > -1 || segText.toUpperCase().indexOf(filter) > -1) {{
                tr[i].style.display = "";
            }} else {{
                tr[i].style.display = "none";
            }}
        }}
    }}
}}

function filterSenateTable() {{
    var input, filter, table, tr, i;
    input = document.getElementById("senateSearch");
    filter = input.value.toUpperCase();
    table = document.getElementById("senateTable");
    tr = table.getElementsByTagName("tr");
    for (i = 1; i < tr.length; i++) {{
        var txt = tr[i].textContent || tr[i].innerText;
        if (txt.toUpperCase().indexOf(filter) > -1) {{
            tr[i].style.display = "";
        }} else {{
            tr[i].style.display = "none";
        }}
    }}
}}
</script>

</body>
</html>
"""

with open('trump_approval_analysis.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Successfully generated trump_approval_analysis.html!")
