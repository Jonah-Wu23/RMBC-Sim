"""
Observation Operator Audit & Decontamination (Op-L2-v1.1)
Flag "Ghost Jams" (non-propagating stalls/layovers) in off-peak ETA data.

Logic:
    Mark link records as GHOST_JAM if:
    1. Speed < 5 km/h
    2. Travel Time > threshold (default 300s for Rule M)
    3. Distance < 1500m (Guardrail: don't filter long-distance genuine jams)

Outputs:
    - data2/processed/ghost_jams_flags.csv (Audit log)
    - data2/processed/link_stats_offpeak_clean.csv (Cleaned truth)
"""

import pandas as pd
import argparse
import os

INPUT_FILE = "data2/processed/link_stats_offpeak.csv"
OUTPUT_DIR = "data2/processed"

def flag_ghost_jams(input_file, threshold_s=300, rule_name="Rule M"):
    print(f"\n=== Ghost Jam Audit ({rule_name}: Time > {threshold_s}s) ===")
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    df = pd.read_csv(input_file)
    total = len(df)
    
    # Define Filter Logic
    # 1. Speed condition
    cond_speed = df['speed_median'] < 5.0
    
    # 2. Time condition
    cond_time = df['tt_median'] > threshold_s
    
    # 3. Distance Guardrail (only filter short/medium links)
    cond_dist = df['dist_m'] < 1500
    
    # Combined Flag
    df['is_ghost'] = cond_speed & cond_time & cond_dist
    df['flag_reason'] = df['is_ghost'].apply(lambda x: 'GHOST_JAM' if x else 'VALID')
    
    # Stats
    ghosts = df[df['is_ghost']]
    ghost_rate = len(ghosts) / total
    
    print(f"Total Records: {total}")
    print(f"Ghost Jams Flagged: {len(ghosts)} ({ghost_rate:.1%})")
    
    if not ghosts.empty:
        print("\n--- Ghost Jam Examples ---")
        print(ghosts[['route', 'from_seq', 'to_seq', 'dist_m', 'tt_median', 'speed_median']].head(5).to_string(index=False))

    # Save Audit Log
    audit_file = os.path.join(OUTPUT_DIR, f"ghost_jams_flags_{rule_name.replace(' ', '_')}.csv")
    df.to_csv(audit_file, index=False)
    print(f"\nSaved audit log: {audit_file}")
    
    # Save Cleaned Data (only for Main Rule usually, but here we can save all)
    clean_df = df[~df['is_ghost']].copy()
    clean_file = os.path.join(OUTPUT_DIR, f"link_stats_{rule_name.replace(' ', '_')}_clean.csv")
    clean_df.to_csv(clean_file, index=False)
    print(f"Saved cleaned data: {clean_file}")
    print(f"Clean Median Speed: {clean_df['speed_median'].median():.2f} km/h (Original: {df['speed_median'].median():.2f} km/h)")

    return clean_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--threshold", type=int, default=300, help="Time threshold in seconds")
    parser.add_argument("--rule", default="Rule_M", help="Rule name (e.g. Rule_M)")
    args = parser.parse_args()
    
    flag_ghost_jams(args.input, args.threshold, args.rule)
