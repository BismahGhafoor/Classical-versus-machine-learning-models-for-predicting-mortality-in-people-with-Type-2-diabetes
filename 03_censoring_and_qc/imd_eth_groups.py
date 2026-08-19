import pandas as pd
import numpy as np

# Load data
file_path = '/scratch/alice/b/bg205/16_02_26/CLEANED_DATA/Combined_GOLD_Aurum.txt'
df = pd.read_csv(file_path, sep='\t')  # assuming tab-delimited based on .txt extension

# 1. Collapse IMD deciles to quintiles
df['imd_quintile'] = np.ceil(df['e2019_imd_10'] / 2).astype('Int64')

# 2. Remap ethnicity
eth_mapping = {
    'Indian': 'South Asian',
    'Pakistani': 'South Asian',
    'Bangladesi': 'South Asian',
    'Bl_Carib': 'Black',
    'Bl_Afric': 'Black',
    'Bl_Other': 'Black',
    'Other': 'Mixed/Other',
    'Mixed': 'Mixed/Other',
    'Chinese': 'Mixed/Other',
    'Oth_Asian': 'Mixed/Other',
    'Other Mixed': 'Mixed/Other'
}
df['gen_ethnicity'] = df['gen_ethnicity'].replace(eth_mapping)

# Check results
# ============== CHECKING STATEMENTS ==============
print("=" * 50)
print("VERIFICATION CHECK")
print("=" * 50)

print("\nUnique IMD quintile values:")
print(sorted(df['imd_quintile'].dropna().unique()))

print("\nUnique ethnicity values:")
print(df['gen_ethnicity'].unique())

print("\n" + "=" * 50)
print("VALUE COUNTS")
print("=" * 50)

print("\nIMD Quintile distribution:")
print(df['imd_quintile'].value_counts().sort_index())

print("\nEthnicity distribution:")
print(df['gen_ethnicity'].value_counts())

# Save output (optional - uncomment if needed)
output_path = '/scratch/alice/b/bg205/16_02_26/CLEANED_DATA/Combined_GOLD_Aurum_recoded.txt'
df.to_csv(output_path, sep='\t', index=False)
print(f"\nSaved to {output_path}")
