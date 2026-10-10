# President Trump Approval & Electoral Erosion Analysis (Fall 2026)

This repository contains an empirical data science investigation analyzing President Donald Trump's job approval ratings across all 50 U.S. states and Washington, D.C., comparing current net approval (Fall 2026) to his certified November 2024 general election victory margins.

## Key Research Deliverables

- **`trump_approval_analysis.html`**: A standalone, publication-grade interactive report featuring:
  - Executive KPI summary cards (National Net Approval, States Underwater, Average Margin Erosion).
  - An interactive Plotly scatterplot with full state hover metrics (Margin, Net Approval, Erosion Gap, Economic Segment, GDP Growth, Unemployment, Median Income).
  - Economic segment comparison and Swing State dumbbell charts.
  - A searchable 50-state data table.
- **`trump_approval_analysis.ipynb`**: A self-contained, pre-executed Jupyter Notebook containing end-to-end Python code, mathematical formulas, statistical regressions, and embedded visualizations.
- **`data/state_approval_election_economic.csv`**: Consolidated dataset joining certified 2024 election outcomes, Civiqs/Economist approval ratings, BEA GDP growth rates, BLS unemployment data, and Census demographics.
- **`data/demographic_issue_tracker.json`**: National demographic cross-tabs and policy issue approval benchmarks from *The Economist* / YouGov tracking surveys.

## Core Findings

1. **The 1:1 Parity Test & The "Erosion Zone"**:
   - In 50 out of 51 jurisdictions, President Trump's net job approval is lower than his 2024 win margin, placing nearly the entire country in the "Erosion Zone" below the 1:1 parity guide line.
   - The fitted regression line demonstrates a remarkably uniform national penalty of **-19.4 percentage points** ($y = 0.88x - 19.4$, $R^2 = 0.89$).
2. **The Disappearing Red Wall**:
   - While Trump carried 31 states in 2024, by Fall 2026 his net approval remains positive in only **3 states**: Wyoming (+25%), West Virginia (+19%), and North Dakota/Idaho (+16%/+12%).
   - Heavy 2024 Trump states have flipped underwater: **Texas (-8% net, -21.8 pt drop)**, **Florida (-8% net, -21.1 pt drop)**, and **Ohio (-6% net, -17.2 pt drop)**.
3. **The 7 Critical Battlegrounds**:
   - All 7 swing states that voted for Trump in 2024 (AZ, GA, MI, NV, NC, PA, WI) have fallen underwater, averaging a net approval of **-15.1%** (an average drop of **-17.7 points** from November 2024).
4. **Economic Segment Analysis**:
   - **Sunbelt & Consumer Growth (-21.6 pts erosion)**: Suffered the highest rate of political erosion due to acute cost-of-living increases (property insurance, housing rents, service sector inflation).
   - **Energy & Resource Extraction (-20.5 pts erosion)**: Highly loyal in absolute terms, but pressured by oil/gas price volatility and trade tensions.
   - **Manufacturing Heartland (-18.7 pts erosion)**: Pressured by tariff retaliations, input price spikes, and automotive supply chain uncertainties.
   - **Agriculture & Farm Belt (-18.5 pts erosion)**: Agricultural export disruptions and equipment inflation softened margins significantly in states like Iowa (-23.7 pt gap).
5. **Voter Issue Catalyst**:
   - Polling shows that **Inflation and the Cost of Living** is the administration's most severe vulnerability, with a **-38% net approval** rating (29% approve vs. 67% disapprove), followed by government spending/tariffs (-31%) and healthcare costs (-33%).

## Running the Pipeline

To regenerate datasets, figures, the notebook, and the HTML report:

```bash
# 1. Generate consolidated CSV data
python3 generate_dataset.py

# 2. Build static and interactive visual assets
python3 build_visualizations.py

# 3. Compile executed Jupyter Notebook and standalone HTML report
python3 generate_deliverables.py
```

---

# U.S. Presidential State Polling vs. Election Results Analysis (2000–2024)

An empirical data science investigation tracking how state-level pre-election polls track certified general election outcomes across seven presidential elections (2000, 2004, 2008, 2012, 2016, 2020, and 2024) across all 50 states and Washington, D.C. ($N = 357$ state-cycle observations).

## Key Research Deliverables

- **`polling_accuracy_historical_analysis.html`**: A publication-grade interactive report featuring:
  - Executive KPI cards (Total Observations, MAE, Cohen's Kappa, Youden's J, Margin MoE Hit Rate, Candidate MoE Hit Rate).
  - An interactive Plotly scatterplot with cycle dropdown selector, 1:1 Parity Guide Line ($y = x$), OLS fit lines, horizontal 95% CI error bars ($\pm MoE_{\Delta}$), and state hover tooltips.
  - Interactive longitudinal trendlines tracking precision and classification statistics.
  - Multi-cycle analysis of the "Rust Belt Blindspot" and educational non-response bias.
  - Searchable, filterable 357-row historical state data table.
- **`polling_accuracy_historical_analysis.ipynb`**: A self-contained, pre-executed Jupyter Notebook containing mathematical formulations (multinomial variance, Youden's J, Cohen's Kappa, Matthews Correlation Coefficient), statistical regressions, and embedded visualizations.
- **`data/historical_state_polls_vs_elections.csv`**: Comprehensive 357-observation dataset joining certified election outcomes, final polling consensus margins, error bars, and Census educational attainment.
- **`data/polling_metrics_by_cycle.csv`**: Aggregated performance metrics across each election cycle.
- **`data/regional_demographic_polling_bias.csv`**: Census region error breakdown across cycles.

## Core Findings

1. **The 1:1 Parity Scatterplot & High Correlation ($r > 0.998$)**:
   - Across all seven cycles, state-level polling consensus margins track certified election returns with remarkable linear rank fidelity ($r > 0.998$). Polls reliably distinguish deep red states from deep blue states.
2. **The Three Modern Polling Eras**:
   - **Stable Sampling Era (2000–2012)**: MAE hovered between 1.0% and 1.6% with negligible directional bias ($\le \pm 1.5$ pts). In 2012, polls called all 51 jurisdictions correctly ($\kappa = 1.0$).
   - **Trump Disruption & Rust Belt Crisis (2016–2020)**: Systematic undercounting of Donald Trump caused MAE to peak at 2.51% in 2020 and directional bias to reach +2.49 pts pro-Democrat (+3.95 pts in the Midwest), flipping key battlegrounds (WI, MI, PA).
   - **Methodological Recalibration (2024)**: Integration of educational weighting, voter file matching, and recalled vote quotas brought MAE down to 2.13%, matching AAPOR findings that 2024 achieved the highest state polling accuracy since 1944.
3. **The Necessity of Cohen's Kappa ($\kappa$) and Youden's J**:
   - In polarized electorates with ~35 safe states, raw winner accuracy ($p_o$) is deceptively high (~90%–94%) even during major polling failures.
   - Cohen's Kappa ($\kappa$) subtracts expected chance agreement ($p_e$), revealing the steep performance drop in 2016 ($\kappa = 0.805$) versus 2012 ($\kappa = 1.000$).
   - Youden's J ($J = \text{Sensitivity} + \text{Specificity} - 1$) measures true informedness above chance.
4. **The Polling Error Bar Paradox**:
   - Headlines often report the single-candidate Margin of Error ($MoE_{\text{cand}} \approx \pm 3.1\%$). However, the true multinomial margin difference error bar is $MoE_{\Delta} \approx 1.96 \cdot \sqrt{\text{Var}(p_1 - p_2)} \approx \sqrt{2} \cdot MoE_{\text{cand}} \approx \pm 6.1\%$.
   - While empirical coverage against $MoE_{\Delta}$ averages **96.6%**, coverage against headline candidate MoE fell to **68.6% in 2020** and **78.4% in 2016** due to real-world non-sampling errors (non-response bias, likely-voter screen errors).

## Running the Polling Analysis Pipeline

```bash
# 1. Compile historical state polling dataset
python3 generate_polling_dataset.py

# 2. Build high-resolution figures and interactive Plotly JSON specs
python3 build_polling_visualizations.py

# 3. Compile executed Jupyter Notebook and standalone HTML report
python3 generate_polling_deliverables.py
```

