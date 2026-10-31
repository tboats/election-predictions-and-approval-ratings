"""
build_visualizations.py
Generates publication-quality charts (both PNG for the notebook and interactive Plotly JSON/HTML for the report).
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load datasets
df = pd.read_csv('data/state_approval_election_economic.csv')
df_ts = pd.read_csv('data/presidential_approval_historical_timeseries.csv')
with open('data/demographic_issue_tracker.json', 'r') as f:
    demo_data = json.load(f)

# Ensure figures folder exists
os.makedirs('figures', exist_ok=True)

# Set matplotlib style
sns.set_theme(style='whitegrid', font='DejaVu Sans')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

palette = {
    'Manufacturing & Industrial': '#1f77b4',
    'Energy & Resource Extraction': '#ff7f0e',
    'Agriculture & Farm Belt': '#2ca02c',
    'Tech, Finance & Knowledge': '#9467bd',
    'Sunbelt & Consumer Growth': '#e377c2',
    'Public Sector & Diversified': '#7f7f7f'
}

# -------------------------------------------------------------
# 1. CORE SCATTER PLOT: 2024 Win Margin vs Net Approval (PNG)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

# Guide line (1:1 slope)
x_vals = np.linspace(-45, 50, 100)
ax.plot(x_vals, x_vals, color='#d9534f', linestyle='--', linewidth=2, label='1:1 Parity Line (Net Approval = 2024 Margin)', zorder=2)

# Shaded erosion zone
ax.fill_between(x_vals, -90, x_vals, color='#fee2e2', alpha=0.35, label='Erosion Zone (Underwater relative to 2024)', zorder=1)

# OLS Fit
cov_xy = np.cov(df['trump_win_margin_2024'], df['primary_net_approval_2026'])[0, 1]
var_x = np.var(df['trump_win_margin_2024'], ddof=1)
m = cov_xy / var_x
b = np.mean(df['primary_net_approval_2026']) - m * np.mean(df['trump_win_margin_2024'])
r_val = np.corrcoef(df['trump_win_margin_2024'], df['primary_net_approval_2026'])[0, 1]
ax.plot(x_vals, m*x_vals + b, color='#2b6cb0', linestyle='-', linewidth=2.2,
        label=f'Fitted Trend: y = {m:.2f}x - {abs(b):.1f} (R² = {r_val**2:.2f})', zorder=3)

# Scatter points by Economic Segment
for seg, color in palette.items():
    sub = df[df['economic_segment'] == seg]
    ax.scatter(sub['trump_win_margin_2024'], sub['primary_net_approval_2026'],
               color=color, s=sub['total_votes_2024']/55000 + 45, alpha=0.85, edgecolors='black', linewidth=0.7, label=seg, zorder=4)

# Key state labels
key_states = ['WY', 'WV', 'ID', 'ND', 'TX', 'FL', 'OH', 'PA', 'MI', 'WI', 'GA', 'NC', 'AZ', 'NV', 'CA', 'NY', 'IA', 'IN', 'DC']
for _, row in df.iterrows():
    if row['code'] in key_states:
        offset_y = 1.4 if row['code'] not in ['FL', 'TX', 'MI', 'GA'] else -2.6
        offset_x = 0.5 if row['code'] not in ['NC', 'WI'] else -1.8
        ax.text(row['trump_win_margin_2024'] + offset_x, row['primary_net_approval_2026'] + offset_y,
                row['code'], fontsize=9, fontweight='bold', color='#1a202c', zorder=5)

ax.axhline(0, color='#718096', linestyle=':', linewidth=1)
ax.axvline(0, color='#718096', linestyle=':', linewidth=1)

ax.set_title("President Trump: 2024 Win Margin vs. Net Job Approval by State (Fall 2026)", fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel("Trump 2024 Win Margin (%) [Trump % - Harris %]", fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel("Net Job Approval (%) [Approve % - Disapprove %]", fontsize=12, fontweight='bold', labelpad=10)
ax.set_xlim(-45, 52)
ax.set_ylim(-85, 32)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5, loc='lower right')

plt.tight_layout()
plt.savefig('figures/fig1_margin_vs_approval_scatter.png')
plt.close()
print("Saved fig1_margin_vs_approval_scatter.png")

# -------------------------------------------------------------
# 2. ECONOMIC SEGMENT EROSION ANALYSIS (PNG)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# Segment summary
seg_agg = df.groupby('economic_segment').agg(
    win_margin=('trump_win_margin_2024', 'mean'),
    net_approval=('primary_net_approval_2026', 'mean'),
    gap=('approval_vs_win_margin_gap', 'mean'),
    count=('state', 'count')
).reset_index().sort_values(by='gap', ascending=True)

# Subplot 1: Bar chart of average erosion gap
bars = ax1.barh(seg_agg['economic_segment'], seg_agg['gap'], color=[palette[s] for s in seg_agg['economic_segment']], edgecolor='black', alpha=0.85)
ax1.axvline(df['approval_vs_win_margin_gap'].mean(), color='red', linestyle='--', linewidth=1.5, label=f'National Mean Gap: {df["approval_vs_win_margin_gap"].mean():.1f} pts')
for bar in bars:
    w = bar.get_width()
    ax1.text(w - 1.4, bar.get_y() + bar.get_height()/2, f"{w:.1f} pts", ha='right', va='center', fontsize=10, fontweight='bold', color='#1a202c')

ax1.set_title("Average Approval Erosion vs. 2024 Margin by Economic Segment", fontsize=12.5, fontweight='bold')
ax1.set_xlabel("Net Approval Margin Gap (Net Approval - 2024 Margin, pts)", fontsize=11, fontweight='bold')
ax1.set_xlim(-26, 0)
ax1.legend(loc='lower left', fontsize=10)

# Subplot 2: Boxplot of distribution across segments
sns.boxplot(data=df, y='economic_segment', x='approval_vs_win_margin_gap', order=seg_agg['economic_segment'], palette=palette, ax=ax2, fliersize=3)
ax2.set_title("Distribution of State Approval Erosion by Segment", fontsize=12.5, fontweight='bold')
ax2.set_xlabel("Erosion Gap (pts)", fontsize=11, fontweight='bold')
ax2.set_ylabel("")

plt.tight_layout()
plt.savefig('figures/fig2_economic_segments_erosion.png')
plt.close()
print("Saved fig2_economic_segments_erosion.png")

# -------------------------------------------------------------
# 3. ECONOMIC SIGNALS: GDP GROWTH & UNEMPLOYMENT (PNG)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# GDP growth vs Net Approval
sns.regplot(data=df, x='gdp_growth_rate', y='primary_net_approval_2026', ax=ax1,
            scatter_kws={'alpha': 0.8, 's': 50, 'color': '#2b6cb0', 'edgecolors': 'black'},
            line_kws={'color': '#c53030', 'linewidth': 2})
corr_gdp = df[['gdp_growth_rate', 'primary_net_approval_2026']].corr().iloc[0, 1]
ax1.set_title(f"State Real GDP Growth Rate vs. Net Approval\n(Pearson r = {corr_gdp:.2f})", fontsize=12.5, fontweight='bold')
ax1.set_xlabel("Real GDP Growth Rate (%)", fontsize=11, fontweight='bold')
ax1.set_ylabel("Net Job Approval (%)", fontsize=11, fontweight='bold')

# Unemployment vs Net Approval
sns.regplot(data=df, x='unemployment_rate_2026', y='primary_net_approval_2026', ax=ax2,
            scatter_kws={'alpha': 0.8, 's': 50, 'color': '#d69e2e', 'edgecolors': 'black'},
            line_kws={'color': '#742a2a', 'linewidth': 2})
corr_unemp = df[['unemployment_rate_2026', 'primary_net_approval_2026']].corr().iloc[0, 1]
ax2.set_title(f"BLS State Unemployment Rate vs. Net Approval\n(Pearson r = {corr_unemp:.2f})", fontsize=12.5, fontweight='bold')
ax2.set_xlabel("Unemployment Rate (August 2026, %)", fontsize=11, fontweight='bold')
ax2.set_ylabel("Net Job Approval (%)", fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/fig3_macro_economic_signals.png')
plt.close()
print("Saved fig3_macro_economic_signals.png")

# -------------------------------------------------------------
# 4. SWING STATE BATTLEGROUND DUMBBELL / WATERFALL (PNG)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
swings = df[df['is_swing_state']].sort_values(by='trump_win_margin_2024', ascending=True).copy()

y_pos = np.arange(len(swings))
for i, (_, row) in enumerate(swings.iterrows()):
    ax.plot([row['trump_win_margin_2024'], row['primary_net_approval_2026']], [i, i], color='#cbd5e0', linewidth=3, zorder=1)
    ax.scatter(row['trump_win_margin_2024'], i, color='#e53e3e', s=120, edgecolors='black', label='2024 Win Margin' if i==0 else "", zorder=3)
    ax.scatter(row['primary_net_approval_2026'], i, color='#3182ce', s=120, edgecolors='black', label='Fall 2026 Net Approval' if i==0 else "", zorder=3)
    
    # Text annotation of shift
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
plt.savefig('figures/fig4_battleground_dumbbell.png')
plt.close()
print("Saved fig4_battleground_dumbbell.png")

# -------------------------------------------------------------
# 5. DEMOGRAPHIC AND ISSUE DEEP DIVE (PNG)
# -------------------------------------------------------------
demo_df = pd.DataFrame(demo_data['demographics'])
issues_df = pd.DataFrame(demo_data['issues'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# Demographic shift
demo_sorted = demo_df.sort_values(by='shift', ascending=True)
colors_shift = ['#c53030' if s < -20 else '#e53e3e' if s < -10 else '#dd6b20' for s in demo_sorted['shift']]
bars = ax1.barh(demo_sorted['group'], demo_sorted['shift'], color=colors_shift, edgecolor='black', alpha=0.85)
for bar in bars:
    w = bar.get_width()
    ax1.text(w - 1, bar.get_y() + bar.get_height()/2, f"{w:+d} pts", ha='right', va='center', fontsize=9, fontweight='bold', color='#1a202c')

ax1.set_title("National Net Approval Shift by Demographic Group\n(Jan 2025 Inauguration to Fall 2026)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Net Approval Change (Percentage Points)", fontsize=10.5, fontweight='bold')
ax1.set_xlim(-52, 0)

# Issue net approval
issues_sorted = issues_df.sort_values(by='net', ascending=True)
bars2 = ax2.barh(issues_sorted['issue'], issues_sorted['net'], color='#805ad5', edgecolor='black', alpha=0.85)
for bar in bars2:
    w = bar.get_width()
    ax2.text(w - 1, bar.get_y() + bar.get_height()/2, f"{w:+d}%", ha='right', va='center', fontsize=9.5, fontweight='bold', color='#1a202c')

ax2.set_title("President Trump's Net Approval by Key National Issue (Economist/YouGov)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Net Issue Approval (Approve % - Disapprove %)", fontsize=10.5, fontweight='bold')
ax2.set_xlim(-46, 0)

plt.tight_layout()
plt.savefig('figures/fig5_demographic_and_issues.png')
plt.close()
print("Saved fig5_demographic_and_issues.png")

# -------------------------------------------------------------
# 5b. HISTORICAL PRESIDENTIAL TIME SERIES COMPARISON (0-4 YEARS) (PNG)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

pres_styles = {
    'Donald Trump (Term 2)': {'color': '#dc2626', 'lw': 3.5, 'ls': '-', 'zorder': 10},
    'Donald Trump (Term 1)': {'color': '#ea580c', 'lw': 2.2, 'ls': '--', 'zorder': 9},
    'Joe Biden': {'color': '#2563eb', 'lw': 2.2, 'ls': '-', 'zorder': 8},
    'Barack Obama (Term 1)': {'color': '#0284c7', 'lw': 1.8, 'ls': '-', 'zorder': 7},
    'George W. Bush (Term 1)': {'color': '#16a34a', 'lw': 1.8, 'ls': '-', 'zorder': 6},
    'Bill Clinton (Term 1)': {'color': '#9333ea', 'lw': 1.8, 'ls': '-', 'zorder': 5},
    'George H.W. Bush': {'color': '#475569', 'lw': 1.6, 'ls': ':', 'zorder': 4},
    'Ronald Reagan (Term 1)': {'color': '#d97706', 'lw': 1.8, 'ls': '-', 'zorder': 5},
    'Jimmy Carter': {'color': '#ca8a04', 'lw': 1.6, 'ls': ':', 'zorder': 3},
    'Richard Nixon (Term 1)': {'color': '#64748b', 'lw': 1.6, 'ls': ':', 'zorder': 2}
}

current_x = 1.78 # Late October 2026 / Year 1.78 (Final Pre-Election Stage)

for pres, st in pres_styles.items():
    sub = df_ts[df_ts['president'] == pres]
    ax1.plot(sub['years_into_presidency'], sub['approval'], label=pres,
             color=st['color'], linewidth=st['lw'], linestyle=st['ls'], zorder=st['zorder'])
    ax2.plot(sub['years_into_presidency'], sub['net_approval'], label=pres,
             color=st['color'], linewidth=st['lw'], linestyle=st['ls'], zorder=st['zorder'])

# Reference lines on ax1 (Approval)
ax1.axvline(current_x, color='#b91c1c', linestyle='--', linewidth=1.8, alpha=0.85, label='Current Stage (Year 1.78 / Late Oct 2026)')
ax1.axhline(50, color='#718096', linestyle=':', linewidth=1.2, label='50% Approval Threshold')
ax1.set_title("Presidential Job Approval Trajectory (Years 0–4)", fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel("Years into Presidency (0–4 Years)", fontsize=11, fontweight='bold')
ax1.set_ylabel("Job Approval Rating (%)", fontsize=11, fontweight='bold')
ax1.set_xlim(-0.05, 4.05)
ax1.set_ylim(25, 95)
ax1.legend(loc='upper right', frameon=True, fontsize=8.5, ncol=2)

# Point callout for Trump Term 2 on ax1
t2_sub = df_ts[(df_ts['president'] == 'Donald Trump (Term 2)') & (df_ts['years_into_presidency'] <= current_x)]
if not t2_sub.empty:
    last_pt = t2_sub.iloc[-1]
    ax1.scatter([last_pt['years_into_presidency']], [last_pt['approval']], color='#dc2626', s=90, edgecolors='black', zorder=15)
    ax1.annotate(f"Trump T2: {last_pt['approval']:.1f}%\n(Historic Low at Yr 1.78)",
                 xy=(last_pt['years_into_presidency'], last_pt['approval']),
                 xytext=(last_pt['years_into_presidency'] + 0.15, last_pt['approval'] - 6),
                 arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.5),
                 fontsize=9.5, fontweight='bold', color='#991b1b',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2', edgecolor='#dc2626', alpha=0.9))

# Reference lines on ax2 (Net Approval)
ax2.axvline(current_x, color='#b91c1c', linestyle='--', linewidth=1.8, alpha=0.85, label='Current Stage (Year 1.78)')
ax2.axhline(0, color='#4a5568', linestyle='-', linewidth=1.2, label='Net Neutral (0% Net)')
ax2.set_title("Presidential Net Job Approval Trajectory (Years 0–4)", fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel("Years into Presidency (0–4 Years)", fontsize=11, fontweight='bold')
ax2.set_ylabel("Net Job Approval (%) [Approve % - Disapprove %]", fontsize=11, fontweight='bold')
ax2.set_xlim(-0.05, 4.05)
ax2.set_ylim(-35, 85)
ax2.legend(loc='upper right', frameon=True, fontsize=8.5, ncol=2)

# Point callout for Trump Term 2 on ax2
if not t2_sub.empty:
    ax2.scatter([last_pt['years_into_presidency']], [last_pt['net_approval']], color='#dc2626', s=90, edgecolors='black', zorder=15)
    ax2.annotate(f"Trump T2: {last_pt['net_approval']:+.1f}% Net\n(-13.5 pts below T1 & Biden)",
                 xy=(last_pt['years_into_presidency'], last_pt['net_approval']),
                 xytext=(last_pt['years_into_presidency'] + 0.15, last_pt['net_approval'] - 7),
                 arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.5),
                 fontsize=9.5, fontweight='bold', color='#991b1b',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2', edgecolor='#dc2626', alpha=0.9))

plt.tight_layout()
plt.savefig('figures/fig6_presidential_historical_comparison.png')
plt.close()
print("Saved fig6_presidential_historical_comparison.png")

# -------------------------------------------------------------
# 6. INTERACTIVE PLOTLY FIGURES FOR HTML REPORT
# -------------------------------------------------------------
# Chart 1 Interactive: Core Scatter Plot
fig_scatter = go.Figure()

# Add parity line
fig_scatter.add_trace(go.Scatter(
    x=[-45, 50],
    y=[-45, 50],
    mode='lines',
    line=dict(color='#e53e3e', width=2.5, dash='dash'),
    name='1:1 Parity Line (Net Approval = 2024 Margin)',
    hoverinfo='none'
))

# Add OLS regression line
fig_scatter.add_trace(go.Scatter(
    x=[-45, 50],
    y=[m*(-45) + b, m*(50) + b],
    mode='lines',
    line=dict(color='#3182ce', width=2),
    name=f'Fitted Trend Line (y = {m:.2f}x - {abs(b):.1f})',
    hoverinfo='none'
))

# Add scatter points by Economic Segment with size toggles
pop_sizes = []
uniform_sizes = []
trace_indices = []

for idx, (seg, col) in enumerate(palette.items()):
    sub = df[df['economic_segment'] == seg]
    p_size = (sub['pop_2026'] / 500000 + 8).tolist()
    u_size = [11] * len(sub)
    pop_sizes.append(p_size)
    uniform_sizes.append(u_size)
    trace_indices.append(idx + 2) # Traces 0 & 1 are parity and OLS lines
    
    hover_texts = [
        f"<b>{r['state']} ({r['code']})</b><br>" +
        f"Economic Segment: {r['economic_segment']}<br>" +
        f"2024 Win Margin: {r['trump_win_margin_2024']:+.2f}%<br>" +
        f"Fall 2026 Net Approval: {r['primary_net_approval_2026']:+.1f}%<br>" +
        f"Approval Erosion Gap: {r['approval_vs_win_margin_gap']:+.2f} pts<br>" +
        f"Civiqs Approve / Disapprove: {r['civiqs_approve_2026']}% / {r['civiqs_disapprove_2026']}%<br>" +
        f"Real GDP Growth: {r['gdp_growth_rate']}%<br>" +
        f"Unemployment Rate: {r['unemployment_rate_2026']}%<br>" +
        f"Population (2026): {r['pop_2026']:,}<br>" +
        f"Median Household Income: ${r['median_household_income']:,}"
        for _, r in sub.iterrows()
    ]
    
    fig_scatter.add_trace(go.Scatter(
        x=sub['trump_win_margin_2024'],
        y=sub['primary_net_approval_2026'],
        mode='markers+text',
        marker=dict(
            size=u_size, # Default to uniform point size for clean scatter view
            color=col,
            line=dict(width=1, color='black'),
            opacity=0.85
        ),
        text=sub['code'],
        textposition='top center',
        textfont=dict(size=9.5, color='#2d3748'),
        name=seg,
        hovertext=hover_texts,
        hoverinfo='text'
    ))

updatemenus = [
    dict(
        type='buttons',
        direction='left',
        active=0, # 0 is Uniform Size
        x=0.0,
        y=1.11,
        xanchor='left',
        yanchor='top',
        bgcolor='#f1f5f9',
        bordercolor='#cbd5e1',
        borderwidth=1,
        font=dict(size=11, color='#1e293b'),
        buttons=[
            dict(
                label='● Uniform Size (Scatter Focus)',
                method='restyle',
                args=[{'marker.size': uniform_sizes}, trace_indices]
            ),
            dict(
                label='◯ Size by State Population',
                method='restyle',
                args=[{'marker.size': pop_sizes}, trace_indices]
            )
        ]
    )
]

fig_scatter.update_layout(
    title='<b>Trump 2024 Win Margin vs. Net Job Approval by State (Fall 2026)</b><br><sup>Every state except DC falls in the red erosion zone below the 1:1 parity line. Toggle point size mode below.</sup>',
    xaxis_title='Trump 2024 Certified Win Margin (%) [Trump % - Harris %]',
    yaxis_title='Trump Fall 2026 Net Job Approval (%) [Approve % - Disapprove %]',
    template='plotly_white',
    height=760,
    updatemenus=updatemenus,
    legend=dict(orientation='h', yanchor='bottom', y=-0.22, xanchor='center', x=0.5),
    margin=dict(l=50, r=40, b=120, t=100)
)
fig_scatter.add_hline(y=0, line_dash='dot', line_color='#a0aec0')
fig_scatter.add_vline(x=0, line_dash='dot', line_color='#a0aec0')

fig_scatter_html = fig_scatter.to_html(full_html=False, include_plotlyjs='cdn')

# Chart 2 Interactive: Economic Segment Comparison
fig_segments = px.bar(
    seg_agg,
    x='gap',
    y='economic_segment',
    orientation='h',
    color='economic_segment',
    color_discrete_map=palette,
    text=seg_agg['gap'].apply(lambda x: f"{x:.1f} pts"),
    title='<b>Average Net Approval Erosion by Economic Segment</b><br><sup>Difference between Current Net Approval and 2024 Win Margin</sup>',
    labels={'gap': 'Approval Erosion Gap (pts)', 'economic_segment': 'Economic Segment'}
)
fig_segments.update_layout(template='plotly_white', height=400, showlegend=False, margin=dict(l=40, r=40, t=60, b=40))
fig_segments_html = fig_segments.to_html(full_html=False, include_plotlyjs=False)

# Chart 3 Interactive: Swing States Comparison
fig_swing = go.Figure()
swings_sorted = swings.sort_values(by='trump_win_margin_2024', ascending=True)
fig_swing.add_trace(go.Bar(
    y=swings_sorted['state'],
    x=swings_sorted['trump_win_margin_2024'],
    name='2024 Certified Win Margin (%)',
    orientation='h',
    marker_color='#e53e3e',
    text=swings_sorted['trump_win_margin_2024'].apply(lambda x: f"{x:+.1f}%"),
    textposition='outside'
))
fig_swing.add_trace(go.Bar(
    y=swings_sorted['state'],
    x=swings_sorted['primary_net_approval_2026'],
    name='Fall 2026 Net Approval (%)',
    orientation='h',
    marker_color='#3182ce',
    text=swings_sorted['primary_net_approval_2026'].apply(lambda x: f"{x:+.0f}%"),
    textposition='outside'
))
fig_swing.update_layout(
    barmode='group',
    title='<b>The 7 Core 2024 Battlegrounds: From November Victory to Severe Net Deficit</b>',
    xaxis_title='Percentage Points (%)',
    template='plotly_white',
    height=480,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    margin=dict(l=40, r=40, t=80, b=40)
)
fig_swing_html = fig_swing.to_html(full_html=False, include_plotlyjs=False)

# Chart 4 Interactive: Presidential Historical Time Series (0-4 Years)
fig_historical = go.Figure()

# Add traces for Job Approval (Traces 0 to 9)
for pres, st in pres_styles.items():
    sub = df_ts[df_ts['president'] == pres]
    hover = [
        f"<b>{pres}</b><br>" +
        f"Stage: Year {r['years_into_presidency']:.2f} (Month {r['months_into_presidency']:.1f}, Day {r['days_into_presidency']})<br>" +
        f"Job Approval: <b>{r['approval']:.1f}%</b><br>" +
        f"Disapproval: {r['disapproval']:.1f}%<br>" +
        f"Net Approval: {r['net_approval']:+.1f}%"
        for _, r in sub.iterrows()
    ]
    dash_style = 'dash' if 'Term 1' in pres and 'Trump' in pres else ('dot' if st['ls'] == ':' else 'solid')
    fig_historical.add_trace(go.Scatter(
        x=sub['years_into_presidency'],
        y=sub['approval'],
        mode='lines',
        name=pres,
        line=dict(color=st['color'], width=st['lw']*1.5, dash=dash_style),
        hovertext=hover,
        hoverinfo='text',
        visible=True
    ))

# Add traces for Net Approval (Traces 10 to 19)
for pres, st in pres_styles.items():
    sub = df_ts[df_ts['president'] == pres]
    hover = [
        f"<b>{pres}</b><br>" +
        f"Stage: Year {r['years_into_presidency']:.2f} (Month {r['months_into_presidency']:.1f}, Day {r['days_into_presidency']})<br>" +
        f"Net Approval: <b>{r['net_approval']:+.1f}%</b><br>" +
        f"Job Approval: {r['approval']:.1f}%<br>" +
        f"Disapproval: {r['disapproval']:.1f}%"
        for _, r in sub.iterrows()
    ]
    dash_style = 'dash' if 'Term 1' in pres and 'Trump' in pres else ('dot' if st['ls'] == ':' else 'solid')
    fig_historical.add_trace(go.Scatter(
        x=sub['years_into_presidency'],
        y=sub['net_approval'],
        mode='lines',
        name=pres,
        line=dict(color=st['color'], width=st['lw']*1.5, dash=dash_style),
        hovertext=hover,
        hoverinfo='text',
        visible=False
    ))

# Add vertical marker at Year 1.78
fig_historical.add_vline(
    x=1.78,
    line_width=2,
    line_dash='dash',
    line_color='#b91c1c',
    annotation_text='<b>Current Stage (Year 1.78 / Late Oct 2026)</b>',
    annotation_position='top right',
    annotation_font=dict(size=11, color='#b91c1c')
)

# Menus to toggle between Job Approval % and Net Approval %
n_pres = len(pres_styles)
vis_app = [True]*n_pres + [False]*n_pres
vis_net = [False]*n_pres + [True]*n_pres

hist_menus = [
    dict(
        type='buttons',
        direction='left',
        active=0,
        x=0.0,
        y=1.12,
        xanchor='left',
        yanchor='top',
        bgcolor='#f1f5f9',
        bordercolor='#cbd5e1',
        borderwidth=1,
        font=dict(size=11, color='#1e293b'),
        buttons=[
            dict(
                label='📊 Job Approval Rating (%)',
                method='update',
                args=[
                    {'visible': vis_app},
                    {'yaxis': {'title': 'Job Approval Rating (%)', 'range': [20, 95]}}
                ]
            ),
            dict(
                label='⚖️ Net Job Approval (%) [Approve - Disapprove]',
                method='update',
                args=[
                    {'visible': vis_net},
                    {'yaxis': {'title': 'Net Job Approval (%) [Approve % - Disapprove %]', 'range': [-40, 85]}}
                ]
            )
        ]
    )
]

fig_historical.update_layout(
    title='<b>Presidential Job Approval Trajectory Across First 4 Years (1969–2026)</b><br><sup>Comparing Donald Trump (Term 2) at Year 1.72 against modern predecessors across their first 4-year term.</sup>',
    xaxis_title='Years into Presidency (0 to 4 Years)',
    yaxis_title='Job Approval Rating (%)',
    template='plotly_white',
    height=620,
    updatemenus=hist_menus,
    legend=dict(orientation='h', yanchor='bottom', y=-0.32, xanchor='center', x=0.5),
    margin=dict(l=50, r=40, b=140, t=90)
)
fig_historical.add_hline(y=50, line_dash='dot', line_color='#94a3b8')

fig_historical_html = fig_historical.to_html(full_html=False, include_plotlyjs=False)

# Save plotly fragments
with open('figures/interactive_plots.json', 'w') as f:
    json.dump({
        'scatter_html': fig_scatter_html,
        'segments_html': fig_segments_html,
        'swing_html': fig_swing_html,
        'historical_html': fig_historical_html
    }, f)

print("Visualizations successfully built!")

