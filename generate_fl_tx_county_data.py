"""
generate_fl_tx_county_data.py
Merges 2016, 2020, and 2024 county election results for Florida and Texas with:
1. U.S. Census Bureau Demographics (CORGIS dataset: Hispanic %, White non-Hispanic %, Bachelor's %, Income, Total Pop)
2. Florida Department of State Division of Elections Official 2024 General Election Active Registered Voters by Race/Ethnicity
"""

import os
import openpyxl
import pandas as pd
import numpy as np

def build_dataset():
    print("Loading raw election files...")
    df24 = pd.read_csv('data/2024_county_raw.csv')
    df20 = pd.read_csv('data/2020_county_raw.csv')
    df16 = pd.read_csv('data/2016_county_raw.csv')
    df_demo = pd.read_csv('data/county_demographics_corgis.csv')

    # Filter FL and TX
    df24_s = df24[df24['state_name'].isin(['Florida', 'Texas'])][
        ['county_fips', 'state_name', 'county_name', 'votes_gop', 'votes_dem', 'total_votes', 'per_gop', 'per_dem']
    ].rename(
        columns={
            'per_gop': 'gop_pct_24', 'per_dem': 'dem_pct_24',
            'total_votes': 'votes_24', 'votes_gop': 'gop_votes_24', 'votes_dem': 'dem_votes_24'
        }
    )

    df20_s = df20[df20['state_name'].isin(['Florida', 'Texas'])][
        ['county_fips', 'votes_gop', 'votes_dem', 'total_votes', 'per_gop', 'per_dem']
    ].rename(
        columns={
            'per_gop': 'gop_pct_20', 'per_dem': 'dem_pct_20',
            'total_votes': 'votes_20', 'votes_gop': 'gop_votes_20', 'votes_dem': 'dem_votes_20'
        }
    )

    df16_s = df16[df16['combined_fips'].isin(df24_s['county_fips'])][
        ['combined_fips', 'votes_gop', 'votes_dem', 'total_votes', 'per_gop', 'per_dem']
    ].rename(
        columns={
            'combined_fips': 'county_fips', 'per_gop': 'gop_pct_16', 'per_dem': 'dem_pct_16',
            'total_votes': 'votes_16', 'votes_gop': 'gop_votes_16', 'votes_dem': 'dem_votes_16'
        }
    )

    m = df24_s.merge(df20_s, on='county_fips').merge(df16_s, on='county_fips')
    m['state_abbr'] = m['state_name'].map({'Florida': 'FL', 'Texas': 'TX'})
    df_demo['state_abbr'] = df_demo['State']
    df_demo['county_name'] = df_demo['County']

    demo_cols = [
        'state_abbr', 'county_name', 'Ethnicities.Hispanic or Latino',
        'Population.2020 Population', 'Ethnicities.White Alone\t not Hispanic or Latino',
        'Education.Bachelor\'s Degree or Higher', 'Income.Median Houseold Income'
    ]
    m = m.merge(df_demo[demo_cols], on=['state_abbr', 'county_name'])
    m.rename(columns={
        'Ethnicities.Hispanic or Latino': 'hispanic_pct',
        'Population.2020 Population': 'pop_2020',
        'Ethnicities.White Alone\t not Hispanic or Latino': 'white_non_hispanic_pct',
        'Education.Bachelor\'s Degree or Higher': 'bachelors_pct',
        'Income.Median Houseold Income': 'median_income'
    }, inplace=True)

    # 5-digit zero-padded FIPS code for choropleth matching
    m['fips_str'] = m['county_fips'].astype(str).str.zfill(5)

    # Margins (Percentage points): Trump Margin = GOP% - Dem% (+ = Trump win)
    m['trump_margin_24'] = (m['gop_pct_24'] - m['dem_pct_24']) * 100.0
    m['trump_margin_20'] = (m['gop_pct_20'] - m['dem_pct_20']) * 100.0
    m['trump_margin_16'] = (m['gop_pct_16'] - m['dem_pct_16']) * 100.0

    # Dem Margins (+ = Dem win)
    m['dem_margin_24'] = (m['dem_pct_24'] - m['gop_pct_24']) * 100.0
    m['dem_margin_20'] = (m['dem_pct_20'] - m['gop_pct_20']) * 100.0
    m['dem_margin_16'] = (m['dem_pct_16'] - m['gop_pct_16']) * 100.0

    # Swings (+ = swing toward Trump / GOP)
    m['swing_trump_20_24'] = m['trump_margin_24'] - m['trump_margin_20']
    m['swing_trump_16_24'] = m['trump_margin_24'] - m['trump_margin_16']
    m['swing_trump_16_20'] = m['trump_margin_20'] - m['trump_margin_16']

    # Turnout growth
    m['turnout_growth_20_24'] = (m['votes_24'] - m['votes_20']) / m['votes_20'] * 100.0
    m['turnout_growth_16_24'] = (m['votes_24'] - m['votes_16']) / m['votes_16'] * 100.0

    # Load FL DOS official voter registration by race for 2024 General Election
    print("Loading FL DOS 2024 General Election active voter registration by race...")
    wb = openpyxl.load_workbook('data/fl_2024_gen_race_by_county.xlsx')
    sheet = wb.active
    rows = list(sheet.iter_rows(values_only=True))

    fl_reg_dict = {}
    for r in rows[8:]:
        if r[0] and str(r[0]).strip().lower() != 'total':
            c_name = str(r[0]).strip() + ' County'
            hisp_reg = r[4]
            white_reg = r[5]
            black_reg = r[3]
            tot_reg = r[9]
            if isinstance(tot_reg, (int, float)) and tot_reg > 0:
                fl_reg_dict[c_name] = {
                    'reg_total_fl': tot_reg,
                    'reg_hispanic_fl': hisp_reg,
                    'reg_hispanic_pct_fl': (hisp_reg / tot_reg) * 100.0,
                    'reg_white_fl': white_reg,
                    'reg_white_pct_fl': (white_reg / tot_reg) * 100.0,
                    'reg_black_fl': black_reg,
                    'reg_black_pct_fl': (black_reg / tot_reg) * 100.0
                }

    reg_df = pd.DataFrame.from_dict(fl_reg_dict, orient='index').reset_index().rename(columns={'index': 'county_name'})
    m = m.merge(reg_df, on='county_name', how='left')

    out_path = 'data/fl_tx_county_analysis.csv'
    m.to_csv(out_path, index=False)
    print(f"Successfully generated {out_path} with {len(m)} counties (FL: {len(m[m['state_abbr']=='FL'])}, TX: {len(m[m['state_abbr']=='TX'])}).")

if __name__ == '__main__':
    build_dataset()
