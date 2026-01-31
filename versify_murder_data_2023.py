# -*- coding: utf-8 -*-
"""
FIXED CATHOLIC JOURNAL CODE - COMPLETE VERSION

Adjustments:
1. Forces numeric conversion for Race/Gender to prevent data loss.
2. Uses "Adaptive Sampling" (takes full population if n < 500).
3. Handles string vs integer mismatch for Offense Codes.
"""

import pandas as pd
import numpy as np
import warnings
from scipy import stats
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - UPDATE YOUR PATH HERE
# ============================================================================
DATA_PATH = r'C:\college\UALR\research\dissertation_data\parquet_files\opafy23nid.parquet'
OUTPUT_FILE = 'NUMBERS_FOR_PAPER.txt'
RANDOM_SEED = 42
SAMPLE_SIZE = 500  # Will auto-adjust if actual count is lower

# Column Search Lists (Priority Order)
TARGET_CHOICES = ["SENTTOT", "TOTSENTN", "SENTENCE_MONTHS", "SENTMON"]
OFFENSE_CHOICES = ["OFFGUIDE", "OFFENSE_CODE", "CRIMECODE"]
RACE_CHOICES = ['MONRACE', 'RACE']
GENDER_CHOICES = ['MONSEX', 'GENDER', 'SEX']
HISPANIC_CHOICES = ['HISPORIG', 'HISPANIC']

# Set Seed
np.random.seed(RANDOM_SEED)

def log(msg):
    print(f"[LOG] {msg}")

def first_existing(df, candidates):
    """Find first column that exists in dataframe."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ============================================================================
# PART 1: ROBUST DATA LOADING
# ============================================================================
def load_and_clean_data(file_path):
    log(f"Loading data from: {file_path}")
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')
    except Exception as e:
        log(f"Error loading file: {e}")
        return None

    # 1. Identify Columns
    sent_col = first_existing(df, TARGET_CHOICES)
    off_col = first_existing(df, OFFENSE_CHOICES)
    race_col = first_existing(df, RACE_CHOICES)
    gender_col = first_existing(df, GENDER_CHOICES)

    if not (sent_col and off_col and race_col and gender_col):
        log("CRITICAL ERROR: Missing one or more required columns.")
        return None

    log(f"Using columns: Sent={sent_col}, Off={off_col}, Race={race_col}, Sex={gender_col}")

    # 2. Filter for Murders (Handle String/Int Mismatch)
    # Convert offense column to string to catch both 22 and '22'
    df['temp_offense'] = df[off_col].astype(str).str.strip()
    
    # Standard USSC Murder codes
    murder_codes = ['22', '2201', '2A1.1'] 
    df_murders = df[df['temp_offense'].isin(murder_codes)].copy()
    
    log(f"Total Federal Murder Cases Found: {len(df_murders)}")
    
    if len(df_murders) == 0:
        log("No murders found with standard codes.")
        return None

    # 3. Fix Data Types (THE CRITICAL FIX)
    # Force Race and Gender to numeric. Coerce errors to NaN.
    df_murders[race_col] = pd.to_numeric(df_murders[race_col], errors='coerce')
    df_murders[gender_col] = pd.to_numeric(df_murders[gender_col], errors='coerce')

    # 4. Map Demographics
    # Gender Map (0=Male, 1=Female usually in USSC)
    df_murders['gender'] = df_murders[gender_col].map({0: 'Male', 1: 'Female'})
    
    # Race Map (Standard USSC)
    race_labels = {
        1: 'White', 2: 'Black', 3: 'Native American',
        4: 'Asian/Pacific', 5: 'Multi-racial', 7: 'Other',
        8: 'Not Available', 9: 'Non-US', 10: 'Unknown'
    }
    df_murders['race'] = df_murders[race_col].map(race_labels)

    # Hispanic Override
    hisp_col = first_existing(df_murders, HISPANIC_CHOICES)
    if hisp_col:
        df_murders[hisp_col] = pd.to_numeric(df_murders[hisp_col], errors='coerce')
        # If HISPORIG is 1, overwrite race to Hispanic
        df_murders.loc[df_murders[hisp_col] == 1, 'race'] = 'Hispanic'

    # 5. Final Cleaning
    # Rename sentence column
    df_murders = df_murders.rename(columns={sent_col: 'sentence_months'})
    df_murders['sentence_months'] = pd.to_numeric(df_murders['sentence_months'], errors='coerce')

    # Drop missing core data
    df_clean = df_murders.dropna(subset=['gender', 'race', 'sentence_months'])
    
    # Filter for relevant groups
    target_races = ['White', 'Black', 'Hispanic']
    df_final = df_clean[df_clean['race'].isin(target_races)]
    
    # Cap sentences at 600 months (50 years) to handle outliers/life sentences
    df_final = df_final[df_final['sentence_months'] <= 600]
    
    log(f"Final usable dataset after cleaning: {len(df_final)}")
    return df_final

# ============================================================================
# PART 2: ADAPTIVE SAMPLING
# ============================================================================
def get_analysis_sample(df):
    """
    If data < 500, return all of it.
    If data > 500, return a stratified sample of 500.
    """
    n_total = len(df)
    
    if n_total < SAMPLE_SIZE:
        log(f"NOTE: Total cases ({n_total}) < Target ({SAMPLE_SIZE}).")
        log("Using ENTIRE POPULATION (No sampling needed).")
        return df.copy()
    
    log(f"Sampling {SAMPLE_SIZE} cases from {n_total} total...")
    
    # Stratified Sample by Race/Gender
    proportions = df.groupby(['race', 'gender']).size() / n_total
    samples = []
    
    for (race, gender), prop in proportions.items():
        group_data = df[(df['race'] == race) & (df['gender'] == gender)]
        n_target = max(1, int(SAMPLE_SIZE * prop))
        
        if len(group_data) >= n_target:
            sample = group_data.sample(n=n_target, random_state=RANDOM_SEED, replace=False)
        else:
            sample = group_data.sample(n=n_target, random_state=RANDOM_SEED, replace=True)
        samples.append(sample)
        
    df_sample = pd.concat(samples, ignore_index=True)
    # Shuffle
    df_sample = df_sample.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df_sample

# ============================================================================
# PART 3: GENERATE REPORT
# ============================================================================
def generate_stats(df, filename):
    results = []
    results.append("=" * 60)
    results.append("CATHOLIC LAW JOURNAL - FINAL DATA EXTRACT")
    results.append("=" * 60)
    results.append(f"Date: {datetime.now()}")
    results.append(f"Total N: {len(df)}")
    results.append("-" * 60)
    
    # 1. Demographics
    results.append("\n>>> FIGURE 2 DATA: Basic Demographics")
    group_stats = df.groupby(['race', 'gender'])['sentence_months'].agg(['mean', 'count', 'median', 'std'])
    results.append(group_stats.to_string())
    
    # 2. Intersectional Analysis
    results.append("\n\n>>> FIGURE 3 DATA: Intersectional Effects")
    overall_mean = df['sentence_months'].mean()
    results.append(f"Overall Mean Sentence: {overall_mean:.2f}")
    
    # Calculate Effects
    race_means = df.groupby('race')['sentence_months'].mean()
    gender_means = df.groupby('gender')['sentence_months'].mean()
    
    results.append("\n--- Intersectionality Check ---")
    for (race, gender), row in group_stats.iterrows():
        actual = row['mean']
        
        # Additive Prediction = Overall + (Race_Diff) + (Gender_Diff)
        race_diff = race_means[race] - overall_mean
        gender_diff = gender_means[gender] - overall_mean
        predicted = overall_mean + race_diff + gender_diff
        
        intersectional_effect = actual - predicted
        
        results.append(f"\nGroup: {race} {gender}")
        results.append(f"  Actual Mean:    {actual:.2f}")
        results.append(f"  Predicted Mean: {predicted:.2f}")
        results.append(f"  Intersectional: {intersectional_effect:+.2f}")
    
    # 3. ANOVA
    results.append("\n\n>>> STATISTICAL SIGNIFICANCE (ANOVA)")
    groups = [df[df['race'] == r]['sentence_months'] for r in ['White', 'Black', 'Hispanic']]
    f_stat, p_val = stats.f_oneway(*groups)
    results.append(f"One-way ANOVA (Race): F={f_stat:.2f}, p={p_val:.4f}")
    
    # Write to file
    with open(filename, 'w') as f:
        f.write("\n".join(results))
    
    print("\n" + "\n".join(results))
    log(f"Report saved to {filename}")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # 1. Load
    df_clean = load_and_clean_data(DATA_PATH)
    
    if df_clean is not None:
        # 2. Sample (or take all)
        df_final = get_analysis_sample(df_clean)
        
        # 3. Analyze
        generate_stats(df_final, OUTPUT_FILE)
        
        # 4. Save CSV for safety
        df_final.to_csv("final_murder_data_used.csv", index=False)
        log("Data CSV saved as 'final_murder_data_used.csv'")
