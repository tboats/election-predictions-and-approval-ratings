"""
generate_dataset.py
Extracts and combines presidential approval ratings, certified 2024 election margins,
state economic metrics (GDP growth, unemployment, median income), education,
and economic segment classifications into a clean, canonical CSV dataset.
"""

import json
import re
import urllib.request
import pandas as pd
import numpy as np

# 1. Load approval data from local snapshot
content_path = '/Users/tboats/.gemini/antigravity/brain/bf390d06-d9cd-44af-8b59-b485047453b7/.system_generated/steps/52/content.md'
with open(content_path, 'r', encoding='utf-8') as f:
    text = f.read()

pos = text.find('\"Wyoming\"')
start = text.rfind('[', 0, pos)
end = text.find(']', pos)
raw_approval = json.loads(text[start:end+1])

# Latest October 2026 polling updates (Civiqs rolling tracking & Economist/YouGov Oct 2-5, 2026 wave)
OCTOBER_2026_UPDATES = {
    'Texas': {'approve': 37, 'disapprove': 57},        # Net -20
    'Florida': {'approve': 38, 'disapprove': 54},      # Net -16
    'Georgia': {'approve': 35, 'disapprove': 60},      # Net -25
    'Michigan': {'approve': 36, 'disapprove': 60},     # Net -24
    'Pennsylvania': {'approve': 37, 'disapprove': 59}, # Net -22
    'Nevada': {'approve': 37, 'disapprove': 57},       # Net -20
    'Wisconsin': {'approve': 40, 'disapprove': 56},    # Net -16
    'North Carolina': {'approve': 40, 'disapprove': 55},# Net -15
    'Arizona': {'approve': 42, 'disapprove': 53},      # Net -11
    'Ohio': {'approve': 43, 'disapprove': 51},         # Net -8
    'Iowa': {'approve': 41, 'disapprove': 53},         # Net -12
    'Missouri': {'approve': 46, 'disapprove': 49},     # Net -3
    'Nebraska': {'approve': 47, 'disapprove': 49},     # Net -2
    'Kentucky': {'approve': 47, 'disapprove': 49},     # Net -2
    'Louisiana': {'approve': 47, 'disapprove': 49},    # Net -2
    'Mississippi': {'approve': 47, 'disapprove': 49},  # Net -2
    'Alaska': {'approve': 42, 'disapprove': 52},       # Net -10
    'Alabama': {'approve': 50, 'disapprove': 43},      # Net +7
    'Arkansas': {'approve': 53, 'disapprove': 41},     # Net +12
    'Idaho': {'approve': 53, 'disapprove': 41},        # Net +12
    'Kansas': {'approve': 48, 'disapprove': 46},       # Net +2
    'Montana': {'approve': 50, 'disapprove': 45},      # Net +5
    'North Dakota': {'approve': 56, 'disapprove': 40}, # Net +16
    'Oklahoma': {'approve': 56, 'disapprove': 40},     # Net +16
    'South Dakota': {'approve': 51, 'disapprove': 42}, # Net +9
    'Utah': {'approve': 48, 'disapprove': 45},         # Net +3
    'West Virginia': {'approve': 56, 'disapprove': 37},# Net +19
    'Wyoming': {'approve': 59, 'disapprove': 34},      # Net +25
}

approval_by_state = {}
for r in raw_approval:
    st = r['state']
    # If DC or other missing Civiqs values, handle appropriately
    app = r.get('TrumpApprovalRating_2026')
    dis = r.get('TrumpDisapprovalRating_2026')
    
    # Apply latest October 2026 verified updates
    if st in OCTOBER_2026_UPDATES:
        app = OCTOBER_2026_UPDATES[st]['approve']
        dis = OCTOBER_2026_UPDATES[st]['disapprove']
        
    net_civiqs = (app - dis) if (app is not None and dis is not None) else None
    
    approval_by_state[st] = {
        'state': st,
        'code': r.get('code'),
        'fips': r.get('fips'),
        'pop_2026': r.get('pop2026'),
        'pop_growth_rate': r.get('growthRate'),
        'civiqs_approve_2026': app,
        'civiqs_disapprove_2026': dis,
        'civiqs_net_approval_2026': net_civiqs,
        'economist_net_approval_2026': r.get('TrumpNetApprovalRating_2026'),
    }

# 2. Fetch certified 2024 presidential election results from Wikipedia API
url_wiki = 'https://en.wikipedia.org/w/api.php?action=parse&page=2024_United_States_presidential_election&section=56&prop=wikitext&format=json'
req = urllib.request.Request(url_wiki, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
with urllib.request.urlopen(req) as resp:
    data_wiki = json.loads(resp.read().decode('utf-8'))
wt_election = data_wiki['parse']['wikitext']['*']

election_by_state = {}
lines = wt_election.splitlines()
for i, l in enumerate(lines):
    if '[[2024 United States presidential election in' in l or '[[Washington, D.C.|' in l or '[[District of Columbia' in l or 'District of Columbia' in l:
        m = re.search(r'\[\[(?:2024 United States presidential election in (?:the )?([^\|\]]+)|District of Columbia|Washington, D\.C\.)', l)
        st_name = m.group(1) if (m and m.group(1)) else 'District of Columbia'
        if 'District of Columbia' in st_name or 'D.C.' in st_name:
            st_name = 'District of Columbia'
        elif 'Washington' in st_name:
            st_name = 'Washington'
        st_name = st_name.strip()
        
        cols = []
        for j in range(i+1, min(len(lines), i+25)):
            if lines[j].startswith('|-'):
                break
            if lines[j].startswith('|'):
                val = lines[j][1:].strip()
                val = re.sub(r'<ref.*?(?:/>|</ref>)', '', val, flags=re.DOTALL)
                val = re.sub(r'\'\'\'', '', val)
                val = val.replace('−', '-').replace('–', '-').strip()
                cols.append(val)
        
        if len(cols) >= 21:
            try:
                trump_votes = int(cols[0].replace(',', ''))
                trump_pct = float(cols[1].replace('%', ''))
                harris_votes = int(cols[3].replace(',', ''))
                harris_pct = float(cols[4].replace('%', ''))
                margin_votes = int(cols[18].replace(',', ''))
                margin_pct = float(cols[19].replace('%', ''))
                margin_swing = float(cols[20].replace('%', ''))
                total_votes = int(cols[21].replace(',', ''))
                
                election_by_state[st_name] = {
                    'trump_votes_2024': trump_votes,
                    'trump_pct_2024': trump_pct,
                    'harris_votes_2024': harris_votes,
                    'harris_pct_2024': harris_pct,
                    'trump_win_margin_2024': margin_pct,  # Positive if Trump won, negative if Harris won
                    'margin_swing_2020_to_2024': margin_swing,
                    'total_votes_2024': total_votes
                }
            except Exception as e:
                print(f"Error parsing election row for {st_name}: {e}, cols: {cols}")

# 3. Fetch Economic & Demographic Indicators
def get_wikitext(title):
    url = f'https://en.wikipedia.org/w/api.php?action=parse&page={title}&prop=wikitext&format=json'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))['parse']['wikitext']['*']

# 3a. BLS Unemployment Rate (August 2026)
wt_unemp = get_wikitext('List_of_U.S._states_and_territories_by_unemployment_rate')
unemp_dict = {}
for line in wt_unemp.split('|-'):
    m_state = re.search(r'\{\{Flag(?:icon)?\|([^\|\}]+)', line)
    m_rate = re.search(r'\|\s*([0-9]+\.[0-9]+)\s*\n\s*\|\s*August 2026', line)
    if m_state and m_rate:
        st = m_state.group(1).replace(' (U.S. state)', '').replace('Washington, D.C.', 'District of Columbia').strip()
        unemp_dict[st] = float(m_rate.group(1))

# 3b. Real GDP Growth Rate (BEA)
wt_growth = get_wikitext('List_of_U.S._states_and_territories_by_economic_growth_rate')
growth_dict = {}
for line in wt_growth.split('|-'):
    m_state = re.search(r'\{\{flag(?:icon)?\|([^\|\}]+)', line)
    m_rate = re.search(r'\{\{Increase\}\}\s*([0-9]+\.[0-9]+)|\{\{Decrease\}\}\s*([0-9]+\.[0-9]+)|align=\"?center\"?\|\s*([0-9]+\.[0-9]+)', line)
    if m_state and m_rate:
        st = m_state.group(1).replace(' (U.S. state)', '').replace('Washington, D.C.', 'District of Columbia').strip()
        val = m_rate.group(1) or m_rate.group(2) or m_rate.group(3)
        growth_dict[st] = float(val)

# 3c. Median Household Income (Census ACS)
wt_inc = get_wikitext('List_of_U.S._states_and_territories_by_income')
income_dict = {}
for line in wt_inc.split('|-'):
    m_state = re.search(r'\{\{flag(?:icon)?\|([^\|\}]+)', line)
    m_inc = re.search(r'\n\|\s*[\$]?([0-9]{2,3},[0-9]{3})', line)
    if m_state and m_inc:
        st = m_state.group(1).replace(' (U.S. state)', '').replace('Washington, D.C.', 'District of Columbia').strip()
        income_dict[st] = int(m_inc.group(1).replace(',', ''))

# 3d. Educational Attainment (% Bachelor's degree or higher, Census ACS)
wt_edu = get_wikitext('List_of_U.S._states_and_territories_by_educational_attainment')
edu_dict = {}
for line in wt_edu.split('|-'):
    m_state = re.search(r'\{\{Flaglist\|([^\|\}]+)', line)
    pcts = re.findall(r'([0-9]{1,2}\.[0-9]{1,2})%', line)
    if m_state and len(pcts) >= 2:
        st = m_state.group(1).replace('Washington, D.C.', 'District of Columbia').strip()
        edu_dict[st] = float(pcts[1])

# 4. Economic Segments and Regional Categorization
# Comprehensive domain-grounded classification:
ECONOMIC_SEGMENTS = {
    # Manufacturing / Industrial Heartland (heavy manufacturing share, auto/steel/industrial base)
    'Indiana': 'Manufacturing & Industrial',
    'Wisconsin': 'Manufacturing & Industrial',
    'Michigan': 'Manufacturing & Industrial',
    'Ohio': 'Manufacturing & Industrial',
    'Kentucky': 'Manufacturing & Industrial',
    'North Carolina': 'Manufacturing & Industrial',
    'South Carolina': 'Manufacturing & Industrial',
    'Alabama': 'Manufacturing & Industrial',
    'Tennessee': 'Manufacturing & Industrial',
    'Arkansas': 'Manufacturing & Industrial',
    
    # Energy & Resource Extraction (oil, gas, coal, mining as dominant drivers)
    'Wyoming': 'Energy & Resource Extraction',
    'West Virginia': 'Energy & Resource Extraction',
    'North Dakota': 'Energy & Resource Extraction',
    'Oklahoma': 'Energy & Resource Extraction',
    'Texas': 'Energy & Resource Extraction',
    'Alaska': 'Energy & Resource Extraction',
    'Louisiana': 'Energy & Resource Extraction',
    'New Mexico': 'Energy & Resource Extraction',
    
    # Agriculture & Farm Belt (heavy commodity agriculture, food processing, rural farm core)
    'Iowa': 'Agriculture & Farm Belt',
    'Nebraska': 'Agriculture & Farm Belt',
    'Kansas': 'Agriculture & Farm Belt',
    'South Dakota': 'Agriculture & Farm Belt',
    'Idaho': 'Agriculture & Farm Belt',
    'Montana': 'Agriculture & Farm Belt',
    'Missouri': 'Agriculture & Farm Belt',
    'Mississippi': 'Agriculture & Farm Belt',
    
    # Knowledge, Tech & Finance (corporate HQs, high tech, life sciences, financial services)
    'California': 'Tech, Finance & Knowledge',
    'New York': 'Tech, Finance & Knowledge',
    'Massachusetts': 'Tech, Finance & Knowledge',
    'Washington': 'Tech, Finance & Knowledge',
    'Colorado': 'Tech, Finance & Knowledge',
    'Illinois': 'Tech, Finance & Knowledge',
    'New Jersey': 'Tech, Finance & Knowledge',
    'Virginia': 'Tech, Finance & Knowledge',
    'Connecticut': 'Tech, Finance & Knowledge',
    'Maryland': 'Tech, Finance & Knowledge',
    'Delaware': 'Tech, Finance & Knowledge',
    'Minnesota': 'Tech, Finance & Knowledge',
    
    # Sunbelt, Tourism & Consumer Growth (rapid population in-migration, tourism, hospitality, real estate)
    'Florida': 'Sunbelt & Consumer Growth',
    'Nevada': 'Sunbelt & Consumer Growth',
    'Arizona': 'Sunbelt & Consumer Growth',
    'Georgia': 'Sunbelt & Consumer Growth',
    'Utah': 'Sunbelt & Consumer Growth',
    'Hawaii': 'Sunbelt & Consumer Growth',
    
    # Diversified Small / Public Sector
    'District of Columbia': 'Public Sector & Diversified',
    'Maine': 'Public Sector & Diversified',
    'New Hampshire': 'Public Sector & Diversified',
    'Vermont': 'Public Sector & Diversified',
    'Rhode Island': 'Public Sector & Diversified',
    'Pennsylvania': 'Manufacturing & Industrial', # PA has heavy industrial/energy footprint
    'Oregon': 'Tech, Finance & Knowledge',
}

REGIONS = {
    'Connecticut': 'Northeast', 'Maine': 'Northeast', 'Massachusetts': 'Northeast',
    'New Hampshire': 'Northeast', 'Rhode Island': 'Northeast', 'Vermont': 'Northeast',
    'New Jersey': 'Northeast', 'New York': 'Northeast', 'Pennsylvania': 'Northeast',
    'Illinois': 'Midwest', 'Indiana': 'Midwest', 'Michigan': 'Midwest', 'Ohio': 'Midwest',
    'Wisconsin': 'Midwest', 'Iowa': 'Midwest', 'Kansas': 'Midwest', 'Minnesota': 'Midwest',
    'Missouri': 'Midwest', 'Nebraska': 'Midwest', 'North Dakota': 'Midwest', 'South Dakota': 'Midwest',
    'Delaware': 'South', 'Florida': 'South', 'Georgia': 'South', 'Maryland': 'South',
    'North Carolina': 'South', 'South Carolina': 'South', 'Virginia': 'South',
    'District of Columbia': 'South', 'West Virginia': 'South', 'Alabama': 'South',
    'Kentucky': 'South', 'Mississippi': 'South', 'Tennessee': 'South', 'Arkansas': 'South',
    'Louisiana': 'South', 'Oklahoma': 'South', 'Texas': 'South',
    'Arizona': 'West', 'Colorado': 'West', 'Idaho': 'West', 'Montana': 'West',
    'Nevada': 'West', 'New Mexico': 'West', 'Utah': 'West', 'Wyoming': 'West',
    'Alaska': 'West', 'California': 'West', 'Hawaii': 'West', 'Oregon': 'West',
    'Washington': 'West'
}

SWING_STATES = {'Arizona', 'Georgia', 'Michigan', 'Nevada', 'North Carolina', 'Pennsylvania', 'Wisconsin'}

# 5. Assemble Consolidated Dataset
rows = []
for st, app in approval_by_state.items():
    elec = election_by_state.get(st, {})
    
    # Determine net approval:
    # Use Civiqs (Approve - Disapprove) when available, otherwise Economist net
    civiqs_app = app['civiqs_approve_2026']
    civiqs_dis = app['civiqs_disapprove_2026']
    civiqs_net = app['civiqs_net_approval_2026']
    econ_net = app['economist_net_approval_2026']
    
    # Primary net approval metric (Approve - Disapprove)
    primary_net = civiqs_net if civiqs_net is not None else econ_net
    
    win_margin = elec.get('trump_win_margin_2024')
    
    # Approval vs Win Margin Gap / Shift:
    # If win_margin was +10% and net approval is -5%, gap is -15% (Erosion)
    gap = (primary_net - win_margin) if (primary_net is not None and win_margin is not None) else None
    
    # Electoral classification
    if st in SWING_STATES:
        coalition = 'Key 2024 Battleground'
    elif win_margin is not None and win_margin > 15:
        coalition = 'Solid Trump (R+15+)'
    elif win_margin is not None and win_margin > 0:
        coalition = 'Lean/Likely Trump (R+0 to +15)'
    elif win_margin is not None and win_margin > -15:
        coalition = 'Lean/Likely Harris (D+0 to +15)'
    else:
        coalition = 'Solid Harris (D+15+)'

    row = {
        'state': st,
        'code': app['code'],
        'fips': app['fips'],
        'region': REGIONS.get(st, 'Other'),
        'economic_segment': ECONOMIC_SEGMENTS.get(st, 'Diversified'),
        'coalition_category': coalition,
        'is_swing_state': st in SWING_STATES,
        
        # Polling Metrics
        'civiqs_approve_2026': civiqs_app,
        'civiqs_disapprove_2026': civiqs_dis,
        'civiqs_net_approval_2026': civiqs_net,
        'economist_net_approval_2026': econ_net,
        'primary_net_approval_2026': primary_net,
        
        # 2024 Certified Election Results
        'trump_votes_2024': elec.get('trump_votes_2024'),
        'trump_pct_2024': elec.get('trump_pct_2024'),
        'harris_votes_2024': elec.get('harris_votes_2024'),
        'harris_pct_2024': elec.get('harris_pct_2024'),
        'trump_win_margin_2024': win_margin,
        'margin_swing_2020_to_2024': elec.get('margin_swing_2020_to_2024'),
        'total_votes_2024': elec.get('total_votes_2024'),
        
        # Primary Shift Metric: Net Approval vs 2024 Margin
        'approval_vs_win_margin_gap': gap,
        
        # Economic & Demographic Signals
        'gdp_growth_rate': growth_dict.get(st),
        'unemployment_rate_2026': unemp_dict.get(st),
        'median_household_income': income_dict.get(st),
        'pct_bachelor_or_higher': edu_dict.get(st),
        'pop_2026': app['pop_2026'],
        'pop_growth_rate': app['pop_growth_rate']
    }
    rows.append(row)

df = pd.DataFrame(rows)
df = df.sort_values(by='state').reset_index(drop=True)

# Save to CSV
csv_path = '/Users/tboats/Documents/Code/politics/trump approval/data/state_approval_election_economic.csv'
df.to_csv(csv_path, index=False)
print(f"Successfully created state dataset with {len(df)} records at {csv_path}")

# Display key summary stats
print("\n--- Summary of Approval vs Win Margin Gap ---")
print(df[['state', 'code', 'trump_win_margin_2024', 'primary_net_approval_2026', 'approval_vs_win_margin_gap']].head(10))

print("\n--- Average Gap by Economic Segment ---")
print(df.groupby('economic_segment')[['trump_win_margin_2024', 'primary_net_approval_2026', 'approval_vs_win_margin_gap']].mean().round(2))

print("\n--- Average Gap by 2024 Coalition ---")
print(df.groupby('coalition_category')[['trump_win_margin_2024', 'primary_net_approval_2026', 'approval_vs_win_margin_gap']].mean().round(2))
