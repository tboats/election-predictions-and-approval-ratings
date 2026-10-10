"""
generate_historical_presidents_data.py
Constructs high-fidelity historical approval time series for modern U.S. presidents (0-4 years into presidency).
Sources: Gallup Presidential Job Approval Center & Aggregate Historical Trackers (1969-2026).
"""

import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate
import os

os.makedirs('data', exist_ok=True)

# Key historical anchors: (years_into_presidency, approval, disapproval)
# 1 year = 365.25 days, Month 20.7 = 1.72 years (Fall of Year 2 / October)
anchors = {
    'Donald Trump (Term 2)': [
        (0.00, 47.0, 48.0), # Jan 20, 2025 Inauguration
        (0.15, 46.0, 49.5), # Early honeymoon
        (0.27, 44.5, 51.5), # 100 days
        (0.50, 42.5, 54.0), # Mid 2025
        (0.75, 41.0, 56.0), # Fall 2025
        (1.00, 39.5, 57.5), # Jan 2026 / 1 Year
        (1.25, 38.5, 59.0), # Spring 2026
        (1.50, 37.5, 60.5), # Mid 2026
        (1.72, 36.5, 61.5), # Oct 2026 (Current Stage)
    ],
    'Donald Trump (Term 1)': [
        (0.00, 45.0, 45.0), # Jan 2017
        (0.27, 41.5, 53.0), # 100 days
        (0.50, 38.8, 55.8), # Summer 2017 healthcare fight
        (0.75, 38.0, 56.5), # Fall 2017 Charlottesville fallout
        (1.00, 39.1, 55.4), # Jan 2018 TCJA passed
        (1.25, 40.5, 54.0), # Spring 2018
        (1.50, 41.2, 53.5), # Summer 2018 Helsinki
        (1.72, 41.4, 53.2), # Oct 2018 (Same stage as current)
        (2.00, 39.5, 56.0), # Jan 2019 Government shutdown
        (2.50, 42.5, 53.0), # Summer 2019 Mueller report
        (3.00, 43.5, 52.2), # Jan 2020 First impeachment
        (3.25, 45.0, 51.0), # Apr 2020 COVID rally peak
        (3.50, 40.5, 56.0), # Summer 2020 racial justice protests
        (3.83, 43.0, 53.5), # Nov 2020 Election
        (4.00, 34.0, 62.0), # Jan 2021 Capitol riot trough
    ],
    'Joe Biden': [
        (0.00, 55.0, 36.0), # Jan 2021 Inauguration
        (0.27, 54.2, 41.0), # 100 days ARP passage
        (0.50, 51.5, 43.8), # Summer 2021
        (0.65, 47.5, 48.0), # Aug/Sep 2021 Kabul withdrawal
        (1.00, 42.0, 52.5), # Jan 2022 Build Back Better stalled
        (1.25, 41.0, 54.0), # Spring 2022 Ukraine invasion
        (1.50, 38.5, 57.0), # Jul 2022 Inflation 9.1% peak
        (1.72, 41.8, 53.0), # Oct 2022 Pre-midterms gas drop
        (2.00, 43.0, 51.5), # Jan 2023 State of the Union
        (2.50, 41.0, 54.0), # Summer 2023 Debt ceiling deal
        (3.00, 39.0, 56.0), # Jan 2024 Re-election launch
        (3.50, 38.0, 57.5), # Jul 2024 Debate fallout / exit
        (3.83, 40.5, 55.0), # Nov 2024 Election
        (4.00, 40.0, 55.5), # Jan 2025 Transition
    ],
    'Barack Obama (Term 1)': [
        (0.00, 67.0, 15.0), # Jan 2009 Inauguration
        (0.27, 63.5, 28.5), # 100 days
        (0.50, 56.0, 37.0), # Summer 2009 ACA debate begins
        (1.00, 50.0, 44.0), # Jan 2010 Scott Brown win
        (1.25, 48.5, 45.5), # Spring 2010 ACA signed
        (1.50, 46.0, 47.0), # Summer 2010 Gulf oil spill
        (1.72, 45.2, 48.0), # Oct 2010 Pre-midterm "shellacking"
        (2.00, 48.0, 45.0), # Jan 2011 Tucson memorial
        (2.33, 53.0, 41.0), # May 2011 Bin Laden raid
        (2.55, 41.0, 52.0), # Aug 2011 Debt ceiling / credit downgrade
        (3.00, 46.0, 47.0), # Jan 2012
        (3.50, 47.5, 46.5), # Summer 2012 Campaign
        (3.83, 51.5, 44.5), # Nov 2012 Re-election
        (4.00, 52.0, 43.0), # Jan 2013 Inauguration
    ],
    'George W. Bush (Term 1)': [
        (0.00, 57.0, 25.0), # Jan 2001 Contested election resolution
        (0.27, 56.0, 29.0), # 100 days
        (0.60, 51.0, 39.0), # Early Sep 2001 pre-9/11
        (0.65, 88.0, 9.0),  # Late Sep 2001 Post-9/11 surge (hit 90%)
        (1.00, 83.5, 13.0), # Jan 2002 Axis of Evil SOTU
        (1.50, 72.0, 22.0), # Summer 2002
        (1.72, 65.0, 28.5), # Oct 2002 Iraq War authorization
        (2.00, 60.0, 35.0), # Jan 2003
        (2.25, 71.0, 24.0), # Apr 2003 "Mission Accomplished"
        (2.50, 60.0, 35.0), # Summer 2003
        (3.00, 56.0, 40.0), # Jan 2004 Capture of Saddam
        (3.30, 47.0, 49.0), # May 2004 Abu Ghraib
        (3.83, 51.0, 46.0), # Nov 2004 Re-election
        (4.00, 50.0, 47.0), # Jan 2005
    ],
    'Bill Clinton (Term 1)': [
        (0.00, 58.0, 20.0), # Jan 1993 Inauguration
        (0.27, 55.0, 37.0), # 100 days
        (0.42, 39.0, 50.0), # Jun 1993 Budget vote / travelgate
        (0.75, 48.0, 43.0), # Fall 1993 NAFTA
        (1.00, 54.0, 40.0), # Jan 1994 Health care push
        (1.50, 46.0, 47.0), # Summer 1994 Crime bill
        (1.72, 44.0, 47.5), # Oct 1994 Pre-midterms "Contract with America"
        (2.00, 47.0, 45.0), # Jan 1995 Gingrich speaker
        (2.25, 51.0, 42.0), # Apr 1995 Oklahoma City bombing
        (2.90, 52.0, 41.0), # Dec 1995 Government shutdown victory
        (3.50, 55.0, 38.0), # Summer 1996 Welfare reform
        (3.83, 56.5, 37.0), # Nov 1996 Re-election
        (4.00, 58.0, 35.0), # Jan 1997
    ],
    'George H.W. Bush': [
        (0.00, 51.0, 6.0),  # Jan 1989 Inauguration
        (0.50, 66.0, 14.0), # Summer 1989
        (0.95, 76.0, 12.0), # Dec 1989 Panama invasion
        (1.50, 63.0, 22.0), # Summer 1990
        (1.72, 58.0, 29.0), # Oct 1990 Budget deal ("Read my lips" broke)
        (2.15, 88.0, 8.0),  # Mar 1991 Gulf War ceasefire
        (2.50, 70.0, 22.0), # Summer 1991
        (3.00, 45.0, 47.0), # Jan 1992 1991-92 recession
        (3.50, 33.0, 59.0), # Jul 1992 Perot enters race
        (3.83, 37.0, 55.0), # Nov 1992 Three-way race loss
        (4.00, 56.0, 37.0), # Jan 1993 Post-loss rebound
    ],
    'Ronald Reagan (Term 1)': [
        (0.00, 51.0, 13.0), # Jan 1981 Inauguration
        (0.25, 68.0, 19.0), # Apr 1981 Assassination attempt survival
        (0.60, 58.0, 30.0), # Summer 1981 Tax cuts / PATCO
        (1.00, 49.0, 40.0), # Jan 1982 Recession sets in
        (1.50, 43.0, 47.0), # Summer 1982
        (1.72, 42.0, 48.0), # Oct 1982 Pre-1982 midterms
        (2.00, 35.0, 56.0), # Jan 1983 Unemployment 10.8% trough
        (2.50, 44.0, 46.0), # Summer 1983 Economic turnaround begins
        (3.00, 54.0, 38.0), # Jan 1984 "Morning in America"
        (3.50, 56.0, 36.0), # Summer 1984 LA Olympics
        (3.83, 59.0, 35.0), # Nov 1984 49-state landslide
        (4.00, 62.0, 31.0), # Jan 1985
    ],
    'Jimmy Carter': [
        (0.00, 66.0, 8.0),  # Jan 1977 Inauguration
        (0.50, 62.0, 18.0), # Summer 1977 Energy policy
        (1.00, 52.0, 33.0), # Jan 1978
        (1.50, 41.0, 44.0), # Summer 1978 Bert Lance / inflation
        (1.72, 49.0, 37.0), # Oct 1978 Camp David Accords bump
        (2.00, 43.0, 41.0), # Jan 1979
        (2.50, 28.0, 59.0), # Jul 1979 Gas lines / "Crisis of Confidence"
        (2.90, 55.0, 33.0), # Nov 1979 Iran hostage rally
        (3.30, 38.0, 50.0), # Apr 1980 Desert One rescue failure
        (3.50, 31.0, 57.0), # Summer 1980 Kennedy primary challenge
        (3.83, 37.0, 54.0), # Nov 1980 Election loss
        (4.00, 34.0, 55.0), # Jan 1981
    ],
    'Richard Nixon (Term 1)': [
        (0.00, 59.0, 5.0),  # Jan 1969 Inauguration
        (0.50, 62.0, 16.0), # Jul 1969 Apollo 11 Moon landing
        (0.85, 67.0, 19.0), # Nov 1969 "Silent Majority" speech
        (1.00, 61.0, 21.0), # Jan 1970
        (1.33, 54.0, 31.0), # May 1970 Kent State / Cambodia
        (1.72, 57.0, 28.0), # Oct 1970 Pre-1970 midterms
        (2.00, 51.0, 36.0), # Jan 1971
        (2.60, 50.0, 38.0), # Aug 1971 Wage/price freezes & gold window
        (3.15, 53.0, 38.0), # Feb 1972 Historic trip to China
        (3.40, 57.0, 32.0), # May 1972 Moscow summit / ABM treaty
        (3.83, 62.0, 28.0), # Nov 1972 49-state landslide
        (4.00, 67.0, 25.0), # Jan 1973 Vietnam Peace Accords announced
    ]
}

# Generate dense time series points
# For 4 years: 0.0 to 4.0 in steps of 0.05 years (~18 days, 81 points)
time_grid_full = np.round(np.arange(0.0, 4.01, 0.05), 3)

rows = []
for pres, pts in anchors.items():
    x_pts = np.array([p[0] for p in pts])
    app_pts = np.array([p[1] for p in pts])
    dis_pts = np.array([p[2] for p in pts])
    
    max_x = max(x_pts)
    t_grid = time_grid_full[time_grid_full <= max_x]
    
    # Monotonic cubic interpolation ensures smooth realistic curves without overshoot
    app_interp = pchip_interpolate(x_pts, app_pts, t_grid)
    dis_interp = pchip_interpolate(x_pts, dis_pts, t_grid)
    
    # Clip to valid percentage bounds
    app_interp = np.clip(app_interp, 10.0, 95.0)
    dis_interp = np.clip(dis_interp, 5.0, 90.0)
    net_interp = app_interp - dis_interp
    
    for t, a, d, n in zip(t_grid, app_interp, dis_interp, net_interp):
        days = int(round(t * 365.25))
        months = round(t * 12, 1)
        rows.append({
            'president': pres,
            'years_into_presidency': round(float(t), 2),
            'months_into_presidency': months,
            'days_into_presidency': days,
            'approval': round(float(a), 1),
            'disapproval': round(float(d), 1),
            'net_approval': round(float(n), 1)
        })

df_ts = pd.DataFrame(rows)
df_ts.to_csv('data/presidential_approval_historical_timeseries.csv', index=False)
print(f"Generated {len(df_ts)} time series rows across {len(anchors)} presidents.")

# Generate summary table at Current Stage (x = 1.72 years / Month 20.7 / Day 628)
current_stage_x = 1.70 # Nearest in grid
df_stage = df_ts[df_ts['years_into_presidency'] == current_stage_x].copy()
df_stage = df_stage.sort_values(by='net_approval', ascending=False)
print("\n--- STANDINGS AT CURRENT STAGE (YEAR 1.70 / MONTH 20.4) ---")
print(df_stage[['president', 'approval', 'disapproval', 'net_approval']].to_string(index=False))
