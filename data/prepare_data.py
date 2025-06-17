import pandas as pd
import json
from datetime import datetime

# Load the raw CSV
df = pd.read_csv('crime_dataset_india_raw.csv')

# Clean data: Example steps (customize as needed)
# 1. Drop rows with missing essential fields
df = df.dropna(subset=['Report Number', 'Date Reported', 'City', 'Crime Description'])

# 2. Fill missing optional fields with defaults
df['Victim Age'] = df['Victim Age'].fillna(-1)
df['Victim Gender'] = df['Victim Gender'].fillna('Unknown')
df['Weapon Used'] = df['Weapon Used'].fillna('Unknown')
df['Crime Domain'] = df['Crime Domain'].fillna('Unknown')
df['Police Deployed'] = df['Police Deployed'].fillna('Unknown')
df['Case Closed'] = df['Case Closed'].fillna('No')
df['Date Case Closed'] = df['Date Case Closed'].fillna('')

# 3. Parse dates to ISO format
def parse_date(date_str):
    try:
        return pd.to_datetime(date_str).isoformat()
    except:
        return None

df['Date Reported'] = df['Date Reported'].apply(parse_date)
df['Date of Occurrence'] = df['Date of Occurrence'].apply(parse_date)
df['Date Case Closed'] = df['Date Case Closed'].apply(parse_date)

# 4. Convert DataFrame to list of dicts
records = df.to_dict(orient='records')

# 5. Save as JSON
with open('crime_dataset_india_cleaned.json', 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("Data cleaned and saved to crime_dataset_india_cleaned.json")
