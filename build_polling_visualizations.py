"""
build_polling_visualizations.py
Generates publication-quality static figures (PNG, 200 DPI) and Plotly interactive JSON specs
for the longitudinal study of U.S. presidential state-level polling accuracy (2000-2024).

Figures created:
1. figures/fig1_state_scatter_with_error_bars.png: 7-panel faceted scatterplot of Poll Margin vs Actual Margin with 1:1 Parity line, OLS fit, and error bars.
2. figures/fig2_longitudinal_metrics_trend.png: Time series of Pearson r, R^2, MAE, and Directional Bias (2000-2024) [with fixed non-colliding annotations].
3. figures/fig3_classification_metrics_kappa_youden.png: Comparison of Raw Accuracy, Cohen's Kappa, Youden's J, and MCC.
4. figures/fig4_ci_coverage_empirical_gap.png: Nominal 95% CI vs Empirical Coverage (Candidate MoE vs Margin MoE).
5. figures/fig5_regional_geographic_disparities.png: Regional Directional Bias and MAE (Midwest vs South vs Northeast vs West).
6. figures/fig6_demographic_education_misses.png: Polling Error vs Non-College White Share in 2016, 2020, and 2024.
7. figures/fig7_signed_bias_distribution_anomalies.png: Histogram and density of signed bias highlighting Florida (+4.56 sigma) and Texas (+3.23 sigma) anomalies in 2024.
8. figures/fig8_confidence_interval_quality.png: Evaluation of Confidence Interval quality, empirical vs. expected anomaly rates, and Q-Q distribution.
9. figures/interactive_polling_charts.json: Complete Plotly JSON specifications for ALL 8 charts ensuring full hover interactivity.
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('figures', exist_ok=True)

# Visual styling
sns.set_theme(style='whitegrid', font='DejaVu Sans')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.autolayout'] = True

df = pd.read_csv('data/historical_state_polls_vs_elections.csv')
df_metrics = pd.read_csv('data/polling_metrics_by_cycle.csv')
df_reg = pd.read_csv('data/regional_demographic_polling_bias.csv')

# Standardized error calculations
df['se_margin'] = df['moe_margin'] / 1.96
df['z_margin'] = df['signed_error'] / df['se_margin']
df['se_cand'] = df['moe_cand'] / 1.96
df['z_cand'] = df['signed_error'] / df['se_cand']

cycles = [2000, 2004, 2008, 2012, 2016, 2020, 2024]
palette_reg = {'Midwest': '#1f77b4', 'South': '#d62728', 'Northeast': '#2ca02c', 'West': '#ff7f0e'}

# -------------------------------------------------------------------------
# Figure 1: 7-Panel Faceted Scatterplot of Poll Margin vs Actual Margin
# -------------------------------------------------------------------------
print("Generating Figure 1: 7-Panel State Scatter with Error Bars...")
fig, axes = plt.subplots(2, 4, figsize=(22, 11), sharex=True, sharey=True)
axes_flat = axes.flatten()

for idx, cycle in enumerate(cycles):
    ax = axes_flat[idx]
    c_df = df[df['cycle'] == cycle]
    m_row = df_metrics[df_metrics['cycle'] == cycle].iloc[0]
    
    # Parity 1:1 line
    ax.plot([-80, 80], [-80, 80], color='#444444', linestyle='--', linewidth=1.2, label='1:1 Parity ($y = x$)', zorder=2)
    
    # Regression fit line
    x_vals = np.linspace(-75, 75, 100)
    y_vals = m_row['ols_slope'] * x_vals + m_row['ols_intercept']
    ax.plot(x_vals, y_vals, color='#d62728', linestyle='-', linewidth=1.8, label=f"OLS: $y = {m_row['ols_slope']:.2f}x + {m_row['ols_intercept']:.1f}$", zorder=3)
    
    # Scatter points with error bars (moe_margin)
    for _, row in c_df.iterrows():
        pt_color = '#1f77b4' if row['actual_margin'] > 0 else '#d62728'
        edge_col = '#111111' if row['competitiveness'] == 'Battleground' else 'none'
        alpha_val = 0.9 if row['competitiveness'] == 'Battleground' else 0.6
        marker_size = 55 if row['competitiveness'] == 'Battleground' else 35
        
        ax.errorbar(
            row['poll_margin'], row['actual_margin'],
            xerr=row['moe_margin'],
            fmt='o', color=pt_color, markeredgecolor=edge_col, markeredgewidth=1.2,
            ecolor='#888888', elinewidth=0.6, capsize=2, capthick=0.6,
            alpha=alpha_val, markersize=np.sqrt(marker_size), zorder=4
        )
        
        if row['competitiveness'] == 'Battleground' or row['absolute_error'] > 4.5:
            ax.annotate(
                row['code'], (row['poll_margin'], row['actual_margin']),
                xytext=(3, 3), textcoords='offset points', fontsize=7.5,
                weight='bold' if row['competitiveness'] == 'Battleground' else 'normal',
                color='#222222', alpha=0.9
            )
            
    ax.set_title(f"{cycle}: {m_row['candidates']}\n$r = {m_row['pearson_r']:.3f}$ | MAE = {m_row['mae']:.2f}% | Bias = {m_row['directional_bias']:+.2f}%", fontsize=11, fontweight='bold', pad=8)
    ax.set_xlim(-75, 75)
    ax.set_ylim(-75, 75)
    ax.grid(True, linestyle=':', alpha=0.6)
    if idx == 0:
        ax.legend(loc='upper left', fontsize=8.5, framealpha=0.9)

# 8th subplot: Summary KPI Overview
ax_summary = axes_flat[7]
ax_summary.axis('off')
summary_text = (
    "KEY STATISTICAL INSIGHTS (2000-2024)\n"
    "------------------------------------\n"
    f"• Total State-Cycles Analyzed: {len(df)}\n"
    f"• Mean Absolute Error (MAE): {df_metrics['mae'].mean():.2f} pts\n"
    f"• Average Pearson r: {df_metrics['pearson_r'].mean():.4f}\n"
    f"• Average Cohen's Kappa: {df_metrics['cohen_kappa'].mean():.3f}\n"
    f"• Average Youden's J: {df_metrics['youden_j'].mean():.3f}\n\n"
    "THE THREE POLLING ERAS:\n"
    "1. 2000-2012: High Stability\n"
    "   MAE: 1.0% - 1.6% | Zero persistent bias\n"
    "2. 2016 & 2020: The Trump Disruption\n"
    "   MAE: 2.3% - 2.5% | Severe GOP undercount\n"
    "   Midwest Bias: +3.6% to +4.8% pro-Dem\n"
    "3. 2024: Recalibration & Recovery\n"
    "   MAE dropped to 2.13% | Lowest state error\n"
    "   since 1944 (AAPOR Report 2025)"
)
ax_summary.text(0.05, 0.5, summary_text, fontsize=9.5, family='monospace', verticalalignment='center',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#cccccc', linewidth=1.2))

fig.text(0.5, 0.01, 'Consensus Final Pre-Election Poll Margin (Dem % - Rep %)', ha='center', fontsize=12, fontweight='bold')
fig.text(0.01, 0.5, 'Certified Official General Election Margin (Dem % - Rep %)', va='center', rotation='vertical', fontsize=12, fontweight='bold')
plt.suptitle(r'U.S. Presidential State Polling vs. Certified Election Outturns Across Seven Cycles (2000–2024)' + '\n' + r'Horizontal Bars Show 95% Confidence Intervals for Margin Difference ($MoE_{\Delta} = 1.96 \cdot \sigma_{\Delta}$)', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
plt.savefig('figures/fig1_state_scatter_with_error_bars.png', dpi=200)
plt.close()

# -------------------------------------------------------------------------
# Figure 2: Longitudinal Predictive Power & Accuracy Trends (Fixed Collisions)
# -------------------------------------------------------------------------
print("Generating Figure 2: Longitudinal Predictive Power Trends (Cleaned Annotations)...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10.5), sharex=True)

# Panel 1: Error metrics (MAE, RMSE, Directional Bias)
ax1.plot(df_metrics['cycle'], df_metrics['mae'], marker='o', linewidth=2.5, color='#d62728', label='Mean Absolute Error (MAE)')
ax1.plot(df_metrics['cycle'], df_metrics['rmse'], marker='s', linewidth=2.0, linestyle='--', color='#ff7f0e', label='Root Mean Squared Error (RMSE)')
ax1.plot(df_metrics['cycle'], df_metrics['directional_bias'], marker='^', linewidth=2.5, color='#1f77b4', label='Directional Bias (Dem Over/Understatement)')
ax1.plot(df_metrics['cycle'], df_metrics['battleground_mae'], marker='D', linewidth=2.0, linestyle=':', color='#9467bd', label='Battleground State MAE')
ax1.axhline(0, color='#333333', linestyle='-', linewidth=0.8, alpha=0.7)
ax1.set_ylabel('Percentage Points Margin (%)', fontsize=11, fontweight='bold')
ax1.set_title('A. Prediction Error and Directional Bias Trajectory (2000–2024)\n(Positive Bias = Overestimating Democrats / Underestimating Republicans)', fontsize=12, fontweight='bold', pad=18)
ax1.legend(loc='upper left', framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.set_ylim(-0.4, 4.3)  # Expanded headroom so annotations do not collide with title

# Clean, nicely bounded annotations completely within axes
ax1.annotate('2016 Rust Belt Underestimate\nBias: +2.25 pts',
             xy=(2016, 2.25), xytext=(2013.2, 3.4),
             arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.2),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f7ff', edgecolor='#1f77b4', alpha=0.9),
             fontsize=8.5, fontweight='bold')

ax1.annotate('2020 Skew Peak\nBias: +2.49 pts',
             xy=(2020, 2.49), xytext=(2017.5, 3.6),
             arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.2),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f7ff', edgecolor='#1f77b4', alpha=0.9),
             fontsize=8.5, fontweight='bold')

ax1.annotate('2024 Recalibration\nMAE drops to 2.13 pts',
             xy=(2024, 2.13), xytext=(2021.2, 0.9),
             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff5f5', edgecolor='#d62728', alpha=0.9),
             fontsize=8.5, fontweight='bold')

# Panel 2: Correlation & Explained Variance (Pearson r, Spearman rho, R^2)
ax2.plot(df_metrics['cycle'], df_metrics['pearson_r'], marker='o', linewidth=2.5, color='#2ca02c', label='Pearson Correlation ($r$)')
ax2.plot(df_metrics['cycle'], df_metrics['spearman_rho'], marker='v', linewidth=2.0, linestyle='--', color='#17becf', label='Spearman Rank Correlation ($\\rho$)')
ax2.plot(df_metrics['cycle'], df_metrics['r_squared'], marker='x', linewidth=2.0, linestyle='-.', color='#8c564b', label='Variance Explained ($R^2$)')
ax2.set_xlabel('Presidential Election Cycle', fontsize=11, fontweight='bold')
ax2.set_ylabel('Correlation / $R^2$ Metric', fontsize=11, fontweight='bold')
ax2.set_title('B. Correlation and Explained Variance Over Time (2000–2024)', fontsize=12, fontweight='bold', pad=14)
ax2.set_ylim(0.990, 1.0005)
ax2.legend(loc='lower left', framealpha=0.9)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.set_xticks(cycles)

plt.tight_layout()
plt.savefig('figures/fig2_longitudinal_metrics_trend.png', dpi=200)
plt.close()

# -------------------------------------------------------------------------
# Figure 3: Classification & Agreement Statistics (Youden's J & Cohen's Kappa)
# -------------------------------------------------------------------------
print("Generating Figure 3: Classification Metrics (Kappa, Youden's J, MCC)...")
plt.figure(figsize=(12, 6.5))
plt.plot(df_metrics['cycle'], df_metrics['raw_accuracy_pct'] / 100.0, marker='o', linewidth=2.5, color='#7f7f7f', linestyle='--', label='Raw Winner Accuracy ($p_o$)')
plt.plot(df_metrics['cycle'], df_metrics['cohen_kappa'], marker='s', linewidth=2.8, color='#1f77b4', label="Cohen's Kappa ($\\kappa$, Inter-Rater Agreement)")
plt.plot(df_metrics['cycle'], df_metrics['youden_j'], marker='^', linewidth=2.8, color='#2ca02c', label="Youden's J-Statistic ($J = \\text{Sens} + \\text{Spec} - 1$)")
plt.plot(df_metrics['cycle'], df_metrics['mcc'], marker='D', linewidth=2.0, color='#e377c2', linestyle=':', label='Matthews Correlation Coefficient (MCC)')

plt.title("Binary Winner Classification Performance: Youden's J, Cohen's Kappa & MCC (2000–2024)\nEvaluating State Polls as Binary Classifiers of Electoral Victory", fontsize=13, fontweight='bold', pad=14)
plt.xlabel('Presidential Election Cycle', fontsize=11, fontweight='bold')
plt.ylabel('Statistic Value (Scale: -1.0 to 1.0)', fontsize=11, fontweight='bold')
plt.xticks(cycles)
plt.ylim(0.74, 1.05)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower left', fontsize=10, framealpha=0.95)

plt.annotate(
    "2016 'Trump Shock':\nRaw accuracy stays high (90.2%),\nbut Cohen's Kappa plummets to 0.805\nand Youden's J to 0.833 due to\nBattleground flips (MI, PA, WI).",
    xy=(2016, 0.8046), xytext=(2013.0, 0.76),
    arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#eef5fc', edgecolor='#1f77b4', alpha=0.9),
    fontsize=8.5
)
plt.annotate(
    "2012 Perfection:\nAll 51 jurisdictions correctly\ncalled (Kappa = 1.0, J = 1.0).",
    xy=(2012, 1.0), xytext=(2008.8, 0.94),
    arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f9f0', edgecolor='#2ca02c', alpha=0.9),
    fontsize=8.5
)

plt.tight_layout()
plt.savefig('figures/fig3_classification_metrics_kappa_youden.png', dpi=200)
plt.close()

# -------------------------------------------------------------------------
# Figure 4: Empirical Error Bar Coverage Gap (Nominal 95% vs Actual Coverage)
# -------------------------------------------------------------------------
print("Generating Figure 4: The Empirical Error Bar & Coverage Gap...")
fig, ax = plt.subplots(figsize=(12, 6.5))
bar_width = 0.35
x = np.arange(len(cycles))

rects1 = ax.bar(x - bar_width/2, df_metrics['margin_95ci_coverage_pct'], bar_width, label='Margin MoE 95% CI Coverage (Theoretical: 95%)', color='#2ca02c', alpha=0.85)
rects2 = ax.bar(x + bar_width/2, df_metrics['cand_95ci_coverage_pct'], bar_width, label='Candidate MoE Coverage (Common Journalistic MoE)', color='#d62728', alpha=0.85)

ax.axhline(95.0, color='#333333', linestyle='--', linewidth=1.5, label='Nominal 95% Confidence Level')
ax.set_ylabel('Empirical Coverage Rate (%)', fontsize=11, fontweight='bold')
ax.set_title('The Polling Error Bar Paradox: Nominal 95% Confidence vs. Empirical Coverage (2000–2024)\nDemonstrating the Impact of Real-World Non-Sampling Errors on Reported Margins of Error', fontsize=12, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(cycles, fontweight='bold')
ax.set_ylim(50, 105)
ax.legend(loc='lower left', framealpha=0.95, fontsize=10)
ax.grid(True, linestyle=':', alpha=0.6)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1b611b')
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#9e1a1a')

plt.tight_layout()
plt.savefig('figures/fig4_ci_coverage_empirical_gap.png', dpi=200)
plt.close()

# -------------------------------------------------------------------------
# Figure 5: Regional & Geographic Polling Disparities
# -------------------------------------------------------------------------
print("Generating Figure 5: Regional Polling Bias Disparities...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

for region, group in df_reg.groupby('region'):
    ax1.plot(group['cycle'], group['directional_bias'], marker='o', linewidth=2.2, label=region, color=palette_reg[region])

ax1.axhline(0, color='#333333', linestyle='-', linewidth=0.8, alpha=0.7)
ax1.set_title('A. Regional Directional Bias (2000–2024)\nPositive = Pro-Dem Overcount / GOP Undercount', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Presidential Election Cycle', fontsize=11, fontweight='bold')
ax1.set_ylabel('Directional Bias (Percentage Points)', fontsize=11, fontweight='bold')
ax1.set_xticks(cycles)
ax1.legend(loc='upper left', framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.6)

for region, group in df_reg.groupby('region'):
    ax2.plot(group['cycle'], group['mae'], marker='s', linewidth=2.2, label=region, color=palette_reg[region])

ax2.set_title('B. Regional Mean Absolute Error (MAE) (2000–2024)\nAverage Magnitude of State Polling Miss', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Presidential Election Cycle', fontsize=11, fontweight='bold')
ax2.set_ylabel('Mean Absolute Error (Percentage Points)', fontsize=11, fontweight='bold')
ax2.set_xticks(cycles)
ax2.legend(loc='upper left', framealpha=0.9)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('figures/fig5_regional_geographic_disparities.png', dpi=200)
plt.close()

# -------------------------------------------------------------------------
# Figure 6: Demographic Correlates of Polling Error (Non-College White Share)
# -------------------------------------------------------------------------
print("Generating Figure 6: Demographic Correlates of Polling Misses...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
trump_cycles = [2016, 2020, 2024]

for idx, c in enumerate(trump_cycles):
    ax = axes[idx]
    sub_df = df[df['cycle'] == c]
    
    x_data = sub_df['non_college_white_pct'] * 100.0
    y_data = sub_df['signed_error']
    slope, intercept, r_val, p_val, _ = stats.linregress(x_data, y_data)
    
    scatter = ax.scatter(
        x_data, y_data,
        c=sub_df['region'].map(palette_reg),
        s=60, alpha=0.85, edgecolors='#222222', linewidth=0.7
    )
    
    x_grid = np.linspace(x_data.min(), x_data.max(), 50)
    ax.plot(x_grid, slope * x_grid + intercept, color='#d62728', linestyle='--', linewidth=1.8,
            label=f'Slope: {slope:+.2f} pt/10%\n$r = {r_val:+.3f}$ ($p={p_val:.3f}$)')
    
    for _, r in sub_df.iterrows():
        if r['code'] in ['WI', 'MI', 'PA', 'OH', 'IA', 'FL', 'NC', 'AZ', 'GA', 'NV', 'TX']:
            ax.annotate(r['code'], (r['non_college_white_pct'] * 100.0, r['signed_error']),
                        xytext=(3, 3), textcoords='offset points', fontsize=8, fontweight='bold')
            
    ax.axhline(0, color='#333333', linestyle=':', linewidth=0.8)
    ax.set_title(f"{c}: {sub_df['dem_candidate'].iloc[0]} vs {sub_df['rep_candidate'].iloc[0]}\nEducational Non-Response Correlation", fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('Non-College White Population Share (%)', fontsize=10, fontweight='bold')
    if idx == 0:
        ax.set_ylabel('Polling Bias: Dem Overcount (+) / Under (-) (pts)', fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9.5, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.6)

plt.suptitle('Educational Non-Response Bias: State Polling Bias vs. Non-College White Share in Trump Eras (2016–2024)\nDemonstrates How Working-Class White Electorates Were Systematically Undercounted by Pre-Election Polls', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.94])
plt.savefig('figures/fig6_demographic_education_misses.png', dpi=200)
plt.close()

# -------------------------------------------------------------------------
# Figure 7: Signed Bias Distribution & 2024 Florida/Texas Anomalies (NEW)
# -------------------------------------------------------------------------
print("Generating Figure 7: Signed Bias Distribution & 2024 Anomaly Analysis...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

df_2024 = df[df['cycle'] == 2024].copy()
mean_2024 = df_2024['signed_error'].mean()
std_2024 = df_2024['signed_error'].std()

# Panel A: 2024 Histogram with KDE and Gaussian fit
n_bins = 18
counts, bins, patches = ax1.hist(df_2024['signed_error'], bins=n_bins, color='#3b82f6', edgecolor='#1e3a8a', alpha=0.65, density=True, label='Observed 2024 States (N=51)')

# Fitted Normal Curve
x_norm = np.linspace(-1, 8, 200)
y_norm = stats.norm.pdf(x_norm, mean_2024, std_2024)
ax1.plot(x_norm, y_norm, color='#1e3a8a', linewidth=2.2, label=f'Gaussian Fit: N({mean_2024:.2f}, {std_2024:.2f}²)')

# Mark Mean and Sigmas
ax1.axvline(mean_2024, color='#111827', linestyle='-', linewidth=1.5, label=f'Mean Bias ({mean_2024:+.2f} pts)')
ax1.axvline(mean_2024 + std_2024, color='#6b7280', linestyle='--', linewidth=1.0)
ax1.axvline(mean_2024 + 2*std_2024, color='#f59e0b', linestyle='--', linewidth=1.2, label=r'+2$\sigma$ Boundary (4.09 pts)')
ax1.axvline(mean_2024 + 3*std_2024, color='#ef4444', linestyle='--', linewidth=1.5, label=r'+3$\sigma$ Boundary (5.07 pts)')

# Annotate Florida (+6.6 pts, +4.56 sigma) and Texas (+5.3 pts, +3.23 sigma)
fl_row = df_2024[df_2024['code'] == 'FL'].iloc[0]
tx_row = df_2024[df_2024['code'] == 'TX'].iloc[0]
fl_z = (fl_row['signed_error'] - mean_2024) / std_2024
tx_z = (tx_row['signed_error'] - mean_2024) / std_2024

ax1.annotate(
    f"FLORIDA ANOMALY\nBias: +6.60 pts\nZ = +{fl_z:.2f}σ (p < 0.00001)\n(Trump won by 13.1% vs poll 6.5%)",
    xy=(fl_row['signed_error'], 0.04), xytext=(fl_row['signed_error'] - 1.8, 0.22),
    arrowprops=dict(arrowstyle='->', color='#dc2626', lw=2.0),
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#fee2e2', edgecolor='#dc2626', lw=1.5),
    fontsize=8.5, fontweight='bold', color='#991b1b'
)

ax1.annotate(
    f"TEXAS ANOMALY\nBias: +5.30 pts\nZ = +{tx_z:.2f}σ (p = 0.0012)\n(Trump won by 13.8% vs poll 8.5%)",
    xy=(tx_row['signed_error'], 0.06), xytext=(tx_row['signed_error'] - 2.2, 0.38),
    arrowprops=dict(arrowstyle='->', color='#ea580c', lw=2.0),
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffedd5', edgecolor='#ea580c', lw=1.5),
    fontsize=8.5, fontweight='bold', color='#9a3412'
)

ax1.set_title('A. 2024 Signed Polling Bias Distribution\nSevere Positive Skew Driven by Florida & Texas Outliers', fontsize=12, fontweight='bold', pad=12)
ax1.set_xlabel('Signed Polling Error: Dem Overcount / Rep Undercount (pts)', fontsize=10.5, fontweight='bold')
ax1.set_ylabel('Probability Density', fontsize=10.5, fontweight='bold')
ax1.set_xlim(-0.5, 8.0)
ax1.legend(loc='upper right', fontsize=8.5, framealpha=0.95)
ax1.grid(True, linestyle=':', alpha=0.6)

# Panel B: Comparative Violin / Boxplot of Signed Error Across 7 Cycles
cycle_errors = [df[df['cycle'] == c]['signed_error'].values for c in cycles]
bplot = ax2.boxplot(cycle_errors, tick_labels=[str(c) for c in cycles], patch_artist=True,
                    medianprops=dict(color='#111827', linewidth=2.0),
                    flierprops=dict(marker='o', markerfacecolor='#dc2626', markersize=6, alpha=0.8))

colors_box = ['#93c5fd', '#93c5fd', '#86efac', '#86efac', '#fca5a5', '#fca5a5', '#fde047']
for patch, color in zip(bplot['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.axhline(0, color='#333333', linestyle='-', linewidth=0.8, alpha=0.7)
ax2.set_title('B. Multi-Cycle Bias Distribution Comparison (2000–2024)\nBoxplots Show Median, IQR, and Extreme Statistical Fliers', fontsize=12, fontweight='bold', pad=12)
ax2.set_xlabel('Presidential Election Cycle', fontsize=10.5, fontweight='bold')
ax2.set_ylabel('Signed Polling Bias (Percentage Points)', fontsize=10.5, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.6)

# Annotate FL and TX outliers on 2024 box
ax2.annotate('FL (+6.6)', xy=(7, 6.6), xytext=(7.15, 6.4), fontsize=8.5, fontweight='bold', color='#dc2626')
ax2.annotate('TX (+5.3)', xy=(7, 5.3), xytext=(7.15, 5.1), fontsize=8.5, fontweight='bold', color='#ea580c')

plt.tight_layout()
plt.savefig('figures/fig7_signed_bias_distribution_anomalies.png', dpi=200)
plt.close()

# -------------------------------------------------------------------------
# Figure 8: Confidence Interval Quality & Anomaly Rates (NEW)
# -------------------------------------------------------------------------
print("Generating Figure 8: Confidence Interval Quality & Anomaly Analysis...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

# Subplot 1: Observed vs. Expected Anomaly Rate (Outside 95% CI)
x_idx = np.arange(len(cycles))
cand_anom_rates = [100.0 - m for m in df_metrics['cand_95ci_coverage_pct']]
margin_anom_rates = [100.0 - m for m in df_metrics['margin_95ci_coverage_pct']]

ax1.bar(x_idx - 0.2, margin_anom_rates, 0.4, label=r'Margin MoE Anomaly Rate ($|Z_{\Delta}| > 1.96$)', color='#10b981', alpha=0.85)
ax1.bar(x_idx + 0.2, cand_anom_rates, 0.4, label=r'Candidate MoE Anomaly Rate ($|Z_{cand}| > 1.96$)', color='#ef4444', alpha=0.85)
ax1.axhline(5.0, color='#1f2937', linestyle='--', linewidth=1.5, label='Theoretical Expectation (5.0% for 95% CI)')

ax1.set_title('A. Empirical Anomaly Rate vs. 5% Theoretical Expectation\nEvaluating Nominal 95% Confidence Intervals Across Cycles', fontsize=12, fontweight='bold', pad=12)
ax1.set_xlabel('Presidential Election Cycle', fontsize=10.5, fontweight='bold')
ax1.set_ylabel('Percentage of States Outside 95% CI (%)', fontsize=10.5, fontweight='bold')
ax1.set_xticks(x_idx)
ax1.set_xticklabels(cycles, fontweight='bold')
ax1.set_ylim(0, 36)
ax1.legend(loc='upper left', fontsize=9.5, framealpha=0.95)
ax1.grid(True, linestyle=':', alpha=0.6)

# Annotate excess in 2020 and 2016
ax1.annotate('2020 Candidate MoE Failure:\n31.4% outside 95% CI\n(6.3x expected rate!)',
             xy=(5 + 0.2, 31.4), xytext=(3.8, 28),
             arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2', edgecolor='#dc2626', alpha=0.9),
             fontsize=8.5, fontweight='bold')

# Subplot 2: Q-Q Plot of Standardized Margin Residuals (Z_margin)
# Check whether standardized errors are Gaussian N(0, 1)
z_vals = df['z_margin'].sort_values().values
theoretical_quantiles = stats.norm.ppf(np.linspace(0.005, 0.995, len(z_vals)))

ax2.scatter(theoretical_quantiles, z_vals, color='#2563eb', alpha=0.7, edgecolors='#1e40af', s=45, label='Empirical State Residuals (N=357)')
ax2.plot([-3.5, 3.5], [-3.5, 3.5], color='#dc2626', linestyle='--', linewidth=1.8, label='Theoretical Normal (y = x)')
ax2.axvline(1.96, color='#9ca3af', linestyle=':', linewidth=1.0)
ax2.axvline(-1.96, color='#9ca3af', linestyle=':', linewidth=1.0)
ax2.axhline(1.96, color='#9ca3af', linestyle=':', linewidth=1.0)
ax2.axhline(-1.96, color='#9ca3af', linestyle=':', linewidth=1.0)

ax2.set_title(r'B. Normal Q-Q Plot of Standardized Margin Errors ($Z_{\Delta} = Error / SE_{\Delta}$)' + '\n' + 'Demonstrates Leptokurtic Heavy Tails from Non-Sampling Variance', fontsize=12, fontweight='bold', pad=12)
ax2.set_xlabel('Theoretical Standard Normal Quantiles', fontsize=10.5, fontweight='bold')
ax2.set_ylabel('Empirical Standardized Residual Quantiles', fontsize=10.5, fontweight='bold')
ax2.set_xlim(-3.5, 3.5)
ax2.set_ylim(-3.5, 4.2)
ax2.legend(loc='upper left', fontsize=9.5, framealpha=0.95)
ax2.grid(True, linestyle=':', alpha=0.6)

# Annotate extreme tail fliers
ax2.annotate('FL 2024 (+3.52σ)', xy=(stats.norm.ppf(0.995), z_vals[-1]), xytext=(1.2, 3.6),
             arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.2), fontsize=8.5, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/fig8_confidence_interval_quality.png', dpi=200)
plt.close()

# -------------------------------------------------------------------------
# Figure 9: Build Interactive Plotly JSON Specifications for ALL 8 Charts
# -------------------------------------------------------------------------
print("Generating Interactive Plotly JSON specifications for ALL charts...")
interactive_figs = {}

# 1. Interactive Multi-Cycle Scatterplot with Error Bars & Dropdown (Figure 1)
scatter_traces = []
buttons = []

for idx, c in enumerate(cycles):
    c_df = df[df['cycle'] == c]
    m_row = df_metrics[df_metrics['cycle'] == c].iloc[0]
    visible = [False] * len(cycles) * 3
    visible[idx*3] = True
    visible[idx*3 + 1] = True
    visible[idx*3 + 2] = True
    
    trace_parity = {
        'type': 'scatter',
        'x': [-75, 75],
        'y': [-75, 75],
        'mode': 'lines',
        'line': {'color': '#666666', 'dash': 'dash', 'width': 1.5},
        'name': '1:1 Parity Guide Line',
        'hoverinfo': 'skip',
        'visible': (c == 2024)
    }
    
    x_grid = [-75, 75]
    y_grid = [m_row['ols_slope'] * x + m_row['ols_intercept'] for x in x_grid]
    trace_ols = {
        'type': 'scatter',
        'x': x_grid,
        'y': y_grid,
        'mode': 'lines',
        'line': {'color': '#d62728', 'width': 2},
        'name': f"OLS Fit (m={m_row['ols_slope']:.2f}, c={m_row['ols_intercept']:.1f})",
        'hoverinfo': 'skip',
        'visible': (c == 2024)
    }
    
    hover_texts = [
        f"<b>{r['state']} ({r['code']})</b><br>"
        f"Cycle: {c} ({m_row['candidates']})<br>"
        f"Region: {r['region']} | Tier: {r['competitiveness']}<br>"
        f"Poll Margin: {r['poll_margin']:+.1f}% ± {r['moe_margin']:.1f}%<br>"
        f"Actual Margin: {r['actual_margin']:+.1f}%<br>"
        f"Polling Error: {r['signed_error']:+.2f}% ({'Pro-Dem' if r['signed_error'] > 0 else 'Pro-Rep'})<br>"
        f"Inside 95% CI: {'Yes' if r['margin_within_ci'] else 'No'}<br>"
        f"Correct Call: {'Yes' if r['correct_call'] else 'No'}"
        for _, r in c_df.iterrows()
    ]
    
    point_colors = ['#1f77b4' if r['actual_margin'] > 0 else '#d62728' for _, r in c_df.iterrows()]
    
    trace_points = {
        'type': 'scatter',
        'x': c_df['poll_margin'].tolist(),
        'y': c_df['actual_margin'].tolist(),
        'error_x': {
            'type': 'data',
            'array': c_df['moe_margin'].tolist(),
            'visible': True,
            'color': '#888888',
            'thickness': 0.8,
            'width': 3
        },
        'mode': 'markers+text',
        'text': c_df['code'].tolist(),
        'textposition': 'top center',
        'textfont': {'size': 9, 'color': '#222222'},
        'marker': {
            'size': [11 if comp == 'Battleground' else 8 for comp in c_df['competitiveness']],
            'color': point_colors,
            'line': {'color': '#111111', 'width': [1.5 if comp == 'Battleground' else 0.5 for comp in c_df['competitiveness']]},
            'opacity': 0.85
        },
        'texttemplate': '%{text}',
        'hovertext': hover_texts,
        'hoverinfo': 'text',
        'name': f"{c} States",
        'visible': (c == 2024)
    }
    
    scatter_traces.extend([trace_parity, trace_ols, trace_points])
    
    buttons.append({
        'method': 'update',
        'label': str(c),
        'args': [
            {'visible': visible},
            {'title': f"<b>{c} Presidential Election: State Poll Margin vs. Certified Result</b><br><sup>Candidates: {m_row['candidates']} | Pearson r = {m_row['pearson_r']:.3f} | MAE = {m_row['mae']:.2f}% | Bias = {m_row['directional_bias']:+.2f}%</sup>"}
        ]
    })

interactive_figs['scatter_plot'] = {
    'data': scatter_traces,
    'layout': {
        'title': f"<b>2024 Presidential Election: State Poll Margin vs. Certified Result</b><br><sup>Candidates: Kamala Harris vs Donald Trump | Pearson r = 0.999 | MAE = 2.13% | Bias = +2.13%</sup>",
        'xaxis': {'title': 'Consensus Final Poll Margin (Dem % - Rep %)', 'range': [-75, 75], 'zeroline': True, 'zerolinecolor': '#cccccc'},
        'yaxis': {'title': 'Certified General Election Margin (Dem % - Rep %)', 'range': [-75, 75], 'zeroline': True, 'zerolinecolor': '#cccccc'},
        'updatemenus': [{
            'type': 'dropdown',
            'direction': 'down',
            'x': 0.0,
            'y': 1.18,
            'showactive': True,
            'buttons': buttons
        }],
        'height': 650,
        'template': 'plotly_white',
        'hovermode': 'closest'
    }
}

# 2. Interactive Longitudinal Metrics Line Chart (Figure 2)
interactive_figs['metrics_trends'] = {
    'data': [
        {'type': 'scatter', 'x': cycles, 'y': df_metrics['mae'].tolist(), 'name': 'Mean Absolute Error (MAE)', 'line': {'color': '#d62728', 'width': 3}, 'marker': {'size': 8}, 'hoverinfo': 'x+y+name'},
        {'type': 'scatter', 'x': cycles, 'y': df_metrics['directional_bias'].tolist(), 'name': 'Directional Bias (Dem Skew)', 'line': {'color': '#1f77b4', 'width': 3}, 'marker': {'size': 8}, 'hoverinfo': 'x+y+name'},
        {'type': 'scatter', 'x': cycles, 'y': df_metrics['battleground_mae'].tolist(), 'name': 'Battleground State MAE', 'line': {'color': '#9467bd', 'width': 2, 'dash': 'dot'}, 'marker': {'size': 7}, 'hoverinfo': 'x+y+name'},
        {'type': 'scatter', 'x': cycles, 'y': df_metrics['cohen_kappa'].tolist(), 'name': "Cohen's Kappa (κ)", 'line': {'color': '#2ca02c', 'width': 2.5, 'dash': 'dash'}, 'marker': {'size': 8}, 'yaxis': 'y2', 'hoverinfo': 'x+y+name'},
        {'type': 'scatter', 'x': cycles, 'y': df_metrics['youden_j'].tolist(), 'name': "Youden's J-Statistic (J)", 'line': {'color': '#e377c2', 'width': 2.5, 'dash': 'dashdot'}, 'marker': {'size': 8}, 'yaxis': 'y2', 'hoverinfo': 'x+y+name'}
    ],
    'layout': {
        'title': '<b>Longitudinal Polling Accuracy & Classification Statistics (2000–2024)</b>',
        'xaxis': {'title': 'Election Cycle', 'tickvals': cycles},
        'yaxis': {'title': 'Error / Bias (Percentage Points)', 'zeroline': True},
        'yaxis2': {'title': 'Classification Index (Kappa / Youden)', 'overlaying': 'y', 'side': 'right', 'range': [0.75, 1.05]},
        'legend': {'x': 0.02, 'y': 0.05, 'bgcolor': 'rgba(255,255,255,0.9)'},
        'height': 480,
        'template': 'plotly_white',
        'hovermode': 'x unified'
    }
}

# 3. Interactive Classification Metrics (Figure 3)
interactive_figs['classification_metrics'] = {
    'data': [
        {'type': 'scatter', 'x': cycles, 'y': [a/100.0 for a in df_metrics['raw_accuracy_pct']], 'name': 'Raw Winner Accuracy (po)', 'line': {'color': '#7f7f7f', 'width': 2, 'dash': 'dash'}, 'marker': {'size': 7}},
        {'type': 'scatter', 'x': cycles, 'y': df_metrics['cohen_kappa'].tolist(), 'name': "Cohen's Kappa (κ)", 'line': {'color': '#1f77b4', 'width': 3.5}, 'marker': {'size': 9}},
        {'type': 'scatter', 'x': cycles, 'y': df_metrics['youden_j'].tolist(), 'name': "Youden's J-Statistic (J)", 'line': {'color': '#2ca02c', 'width': 3.5}, 'marker': {'size': 9}},
        {'type': 'scatter', 'x': cycles, 'y': df_metrics['mcc'].tolist(), 'name': 'Matthews Correlation (MCC)', 'line': {'color': '#e377c2', 'width': 2.5, 'dash': 'dot'}, 'marker': {'size': 8}}
    ],
    'layout': {
        'title': "<b>Binary Winner Classification Metrics: Youden's J, Cohen's Kappa & MCC</b>",
        'xaxis': {'title': 'Election Cycle', 'tickvals': cycles},
        'yaxis': {'title': 'Metric Value (Scale: -1.0 to 1.0)', 'range': [0.75, 1.05], 'zeroline': False},
        'legend': {'x': 0.02, 'y': 0.1, 'bgcolor': 'rgba(255,255,255,0.9)'},
        'height': 450,
        'template': 'plotly_white',
        'hovermode': 'x unified'
    }
}

# 4. Interactive CI Coverage Gap Bar Chart (Figure 4)
interactive_figs['ci_coverage_gap'] = {
    'data': [
        {'type': 'bar', 'x': [str(c) for c in cycles], 'y': df_metrics['margin_95ci_coverage_pct'].tolist(), 'name': 'Margin Difference MoE 95% Coverage', 'marker': {'color': '#2ca02c'}},
        {'type': 'bar', 'x': [str(c) for c in cycles], 'y': df_metrics['cand_95ci_coverage_pct'].tolist(), 'name': 'Candidate MoE Coverage (Headline MoE)', 'marker': {'color': '#dc2626'}},
        {'type': 'scatter', 'x': [str(c) for c in cycles], 'y': [95.0]*len(cycles), 'mode': 'lines', 'name': 'Nominal 95% Confidence Line', 'line': {'color': '#333333', 'dash': 'dash', 'width': 2}}
    ],
    'layout': {
        'title': '<b>Nominal 95% Confidence vs. Empirical Coverage (2000–2024)</b>',
        'xaxis': {'title': 'Election Cycle'},
        'yaxis': {'title': 'Empirical Coverage Rate (%)', 'range': [50, 105]},
        'barmode': 'group',
        'legend': {'x': 0.02, 'y': 0.15, 'bgcolor': 'rgba(255,255,255,0.9)'},
        'height': 450,
        'template': 'plotly_white',
        'hovermode': 'x unified'
    }
}

# 5. Interactive Regional Bias & MAE (Figure 5)
regional_traces = []
for reg, col in palette_reg.items():
    r_df = df_reg[df_reg['region'] == reg]
    regional_traces.append({
        'type': 'scatter',
        'x': r_df['cycle'].tolist(),
        'y': r_df['directional_bias'].tolist(),
        'name': f"{reg} Bias",
        'line': {'color': col, 'width': 2.5},
        'marker': {'size': 8},
        'hovertext': [f"<b>{reg} ({c})</b><br>Bias: {b:+.2f} pts<br>MAE: {m:.2f} pts" for c, b, m in zip(r_df['cycle'], r_df['directional_bias'], r_df['mae'])],
        'hoverinfo': 'text'
    })

interactive_figs['regional_disparities'] = {
    'data': regional_traces,
    'layout': {
        'title': '<b>Regional Directional Polling Bias Across Cycles (2000–2024)</b><br><sup>Positive = Democratic Overstatement / Republican Undercount</sup>',
        'xaxis': {'title': 'Election Cycle', 'tickvals': cycles},
        'yaxis': {'title': 'Directional Bias (Percentage Points)', 'zeroline': True, 'zerolinecolor': '#cccccc'},
        'legend': {'x': 0.02, 'y': 0.95, 'bgcolor': 'rgba(255,255,255,0.9)'},
        'height': 450,
        'template': 'plotly_white',
        'hovermode': 'closest'
    }
}

# 6. Interactive Signed Bias Distribution with FL and TX Anomalies (Figure 7)
bias_vals_2024 = df_2024['signed_error'].tolist()
state_names_2024 = df_2024['state'].tolist()
codes_2024 = df_2024['code'].tolist()
z_scores_2024 = [round((val - mean_2024)/std_2024, 2) for val in bias_vals_2024]

hover_bias_2024 = [
    f"<b>{s} ({c})</b><br>Signed Bias: {b:+.2f} pts<br>Z-score: {z:+.2f}σ<br>Poll Margin: {r['poll_margin']:+.1f}%<br>Actual Margin: {r['actual_margin']:+.1f}%"
    for s, c, b, z, (_, r) in zip(state_names_2024, codes_2024, bias_vals_2024, z_scores_2024, df_2024.iterrows())
]

interactive_figs['bias_distribution_anomalies'] = {
    'data': [
        {
            'type': 'histogram',
            'x': bias_vals_2024,
            'name': '2024 State Observations',
            'nbinsx': 18,
            'marker': {'color': '#3b82f6', 'line': {'color': '#1d4ed8', 'width': 1}},
            'opacity': 0.75
        },
        {
            'type': 'scatter',
            'x': [fl_row['signed_error']],
            'y': [1],
            'mode': 'markers+text',
            'name': 'Florida (+4.56σ)',
            'text': ['<b>FL (+6.6 pts, +4.56σ)</b>'],
            'textposition': 'top center',
            'marker': {'size': 14, 'color': '#dc2626', 'symbol': 'diamond', 'line': {'color': '#000', 'width': 1.5}},
            'hovertext': [f"<b>Florida (FL)</b><br>Signed Bias: +6.60 pts<br>Z-Score: +4.56σ (p &lt; 0.00001)<br>Poll Margin: -6.5%<br>Actual Margin: -13.1% (Trump +13.1%)"],
            'hoverinfo': 'text'
        },
        {
            'type': 'scatter',
            'x': [tx_row['signed_error']],
            'y': [1],
            'mode': 'markers+text',
            'name': 'Texas (+3.23σ)',
            'text': ['<b>TX (+5.3 pts, +3.23σ)</b>'],
            'textposition': 'top center',
            'marker': {'size': 14, 'color': '#ea580c', 'symbol': 'diamond', 'line': {'color': '#000', 'width': 1.5}},
            'hovertext': [f"<b>Texas (TX)</b><br>Signed Bias: +5.30 pts<br>Z-Score: +3.23σ (p = 0.0012)<br>Poll Margin: -8.5%<br>Actual Margin: -13.8% (Trump +13.8%)"],
            'hoverinfo': 'text'
        }
    ],
    'layout': {
        'title': '<b>Distribution of 2024 Signed Polling Bias & Anomaly Detection</b><br><sup>Mean = +2.13 pts, Std Dev = 0.98 pts | Highlighting Florida (+4.56σ) and Texas (+3.23σ)</sup>',
        'xaxis': {'title': 'Signed Polling Error: Dem Overcount / Rep Undercount (Percentage Points)', 'range': [-1, 8]},
        'yaxis': {'title': 'Number of States'},
        'shapes': [
            {'type': 'line', 'x0': mean_2024, 'x1': mean_2024, 'y0': 0, 'y1': 14, 'line': {'color': '#111827', 'width': 2}},
            {'type': 'line', 'x0': mean_2024 + 2*std_2024, 'x1': mean_2024 + 2*std_2024, 'y0': 0, 'y1': 14, 'line': {'color': '#f59e0b', 'width': 1.5, 'dash': 'dash'}},
            {'type': 'line', 'x0': mean_2024 + 3*std_2024, 'x1': mean_2024 + 3*std_2024, 'y0': 0, 'y1': 14, 'line': {'color': '#ef4444', 'width': 1.5, 'dash': 'dash'}}
        ],
        'annotations': [
            {'x': mean_2024, 'y': 13, 'text': f'Mean: +{mean_2024:.2f}', 'showarrow': False, 'font': {'size': 10}},
            {'x': mean_2024 + 2*std_2024, 'y': 11, 'text': '+2σ', 'showarrow': False, 'font': {'size': 10, 'color': '#d97706'}},
            {'x': mean_2024 + 3*std_2024, 'y': 9, 'text': '+3σ', 'showarrow': False, 'font': {'size': 10, 'color': '#dc2626'}}
        ],
        'legend': {'x': 0.7, 'y': 0.95, 'bgcolor': 'rgba(255,255,255,0.9)'},
        'height': 480,
        'template': 'plotly_white',
        'hovermode': 'closest'
    }
}

# 7. Interactive Confidence Interval Quality & Anomaly Rates (Figure 8)
interactive_figs['ci_anomaly_quality'] = {
    'data': [
        {'type': 'bar', 'x': [str(c) for c in cycles], 'y': margin_anom_rates, 'name': 'Margin MoE Anomalies (|Z| > 1.96)', 'marker': {'color': '#10b981'}},
        {'type': 'bar', 'x': [str(c) for c in cycles], 'y': cand_anom_rates, 'name': 'Candidate MoE Anomalies (Headline MoE)', 'marker': {'color': '#ef4444'}},
        {'type': 'scatter', 'x': [str(c) for c in cycles], 'y': [5.0]*len(cycles), 'mode': 'lines', 'name': 'Expected 5.0% Tail Rate (Gaussian)', 'line': {'color': '#1f2937', 'dash': 'dash', 'width': 2}}
    ],
    'layout': {
        'title': '<b>Empirical Anomaly Rates: Observed vs. 5.0% Expected Under 95% Confidence</b><br><sup>Evaluating Whether State Polling Misses Exceed Pure Sampling Error</sup>',
        'xaxis': {'title': 'Election Cycle'},
        'yaxis': {'title': 'Percentage of States Outside 95% CI (%)', 'range': [0, 36]},
        'barmode': 'group',
        'legend': {'x': 0.02, 'y': 0.95, 'bgcolor': 'rgba(255,255,255,0.9)'},
        'height': 450,
        'template': 'plotly_white',
        'hovermode': 'x unified'
    }
}

# 8. Interactive Demographic Correlates (Figure 6)
demo_traces = []
for c in trump_cycles:
    sub_c = df[df['cycle'] == c]
    demo_traces.append({
        'type': 'scatter',
        'x': [nc*100.0 for nc in sub_c['non_college_white_pct']],
        'y': sub_c['signed_error'].tolist(),
        'mode': 'markers+text',
        'text': sub_c['code'].tolist(),
        'textposition': 'top center',
        'name': f"{c} Cycle",
        'marker': {'size': 9, 'opacity': 0.8},
        'hovertext': [f"<b>{r['state']} ({r['code']})</b><br>{c} Election<br>Non-College White: {r['non_college_white_pct']*100:.1f}%<br>Signed Bias: {r['signed_error']:+.2f} pts<br>Region: {r['region']}" for _, r in sub_c.iterrows()],
        'hoverinfo': 'text'
    })

interactive_figs['demographic_misses'] = {
    'data': demo_traces,
    'layout': {
        'title': '<b>State Polling Bias vs. Non-College White Population Share (2016–2024)</b><br><sup>Quantifying the Educational Non-Response Bias in Modern Presidential Elections</sup>',
        'xaxis': {'title': 'Non-College White Population Share (%)'},
        'yaxis': {'title': 'Signed Polling Bias: Dem Overcount / Rep Undercount (pts)', 'zeroline': True, 'zerolinecolor': '#cccccc'},
        'legend': {'x': 0.02, 'y': 0.95, 'bgcolor': 'rgba(255,255,255,0.9)'},
        'height': 480,
        'template': 'plotly_white',
        'hovermode': 'closest'
    }
}

with open('figures/interactive_polling_charts.json', 'w') as f:
    json.dump(interactive_figs, f, indent=2)

print("All static figures and interactive Plotly JSON specs successfully generated.")
