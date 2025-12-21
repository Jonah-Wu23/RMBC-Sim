# -*- coding: utf-8 -*-
"""
L1 Calibration Results Analysis Script
"""

import pandas as pd
import numpy as np
import sys

def analyze_calibration_log(csv_path):
    # Read data
    df = pd.read_csv(csv_path)
    
    print("=" * 65)
    print("L1 CALIBRATION RESULTS ANALYSIS REPORT")
    print("=" * 65)
    
    print(f"\nTotal Iterations: {len(df)}")
    print(f"  - Initial Samples (LHS): {len(df[df['type']=='initial'])}")
    print(f"  - Bayesian Optimization (BO): {len(df[df['type']=='bo'])}")
    
    print("\n" + "-" * 65)
    print("[Combined Loss (rmse) Analysis]")
    print("-" * 65)
    print(f"Minimum Combined Loss: {df['rmse'].min():.2f} (iteration {df['rmse'].idxmin()})")
    print(f"Maximum Combined Loss: {df['rmse'].max():.2f}")
    print(f"Average Combined Loss: {df['rmse'].mean():.2f}")
    
    # Find best iteration
    best_idx = df['rmse'].idxmin()
    best_row = df.loc[best_idx]
    
    print("\n" + "-" * 65)
    print("[BEST PARAMETER SET] (Minimum Combined Loss)")
    print("-" * 65)
    print(f"Iteration: {int(best_row['iter'])} (type: {best_row['type']})")
    print(f"t_board  = {best_row['t_board']:.4f}")
    print(f"t_fixed  = {best_row['t_fixed']:.4f}")
    print(f"tau      = {best_row['tau']:.4f}")
    print(f"sigma    = {best_row['sigma']:.4f}")
    print(f"minGap   = {best_row['minGap']:.4f}")
    print(f"accel    = {best_row['accel']:.4f}")
    print(f"decel    = {best_row['decel']:.4f}")
    
    print("\n" + "-" * 65)
    print("[Per-Route RMSE Analysis]")
    print("-" * 65)
    print(f"Route 68X (Optimization Target):")
    print(f"  Min RMSE: {df['rmse_68x'].min():.2f}")
    print(f"  Max RMSE: {df['rmse_68x'].max():.2f}")
    print(f"  Avg RMSE: {df['rmse_68x'].mean():.2f}")
    
    print(f"\nRoute 960 (Constraint Anchor, threshold=350):")
    print(f"  Min RMSE: {df['rmse_960'].min():.2f}")
    print(f"  Max RMSE: {df['rmse_960'].max():.2f}")
    print(f"  Avg RMSE: {df['rmse_960'].mean():.2f}")
    
    print("\n" + "-" * 65)
    print("[Best Iteration Per-Route Performance]")
    print("-" * 65)
    print(f"Combined Loss: {best_row['rmse']:.2f}")
    print(f"68X RMSE: {best_row['rmse_68x']:.2f}")
    print(f"960 RMSE: {best_row['rmse_960']:.2f}")
    
    # Check constraint violations
    print("\n" + "-" * 65)
    print("[Constraint Trigger Analysis (960 threshold = 350)]")
    print("-" * 65)
    constraint_violations = df[df['rmse_960'] > 350]
    print(f"Constraint triggered: {len(constraint_violations)} / {len(df)} times")
    if len(constraint_violations) > 0:
        print(f"Triggered at iterations: {list(constraint_violations['iter'].astype(int))}")
    
    # Find 68X best (ignoring constraint)
    print("\n" + "-" * 65)
    print("[68X Best (Ignoring Constraints)]")
    print("-" * 65)
    best_68x_idx = df['rmse_68x'].idxmin()
    best_68x_row = df.loc[best_68x_idx]
    print(f"68X Minimum RMSE: {best_68x_row['rmse_68x']:.2f} (iteration {int(best_68x_row['iter'])})")
    print(f"  Corresponding 960 RMSE: {best_68x_row['rmse_960']:.2f}")
    if best_68x_row['rmse_960'] > 350:
        print(f"  *** WARNING: This iteration triggered 960 constraint penalty! ***")
    
    # BO phase analysis
    bo_df = df[df['type'] == 'bo']
    if len(bo_df) > 0:
        print("\n" + "-" * 65)
        print("[Bayesian Optimization (BO) Phase Analysis]")
        print("-" * 65)
        print(f"BO Phase Best Combined Loss: {bo_df['rmse'].min():.2f}")
        print(f"BO Phase Avg Combined Loss: {bo_df['rmse'].mean():.2f}")
        
        # BO best result
        bo_best_idx = bo_df['rmse'].idxmin()
        bo_best = bo_df.loc[bo_best_idx]
        print(f"\nBO Best Iteration: {int(bo_best['iter'])}")
        print(f"  68X RMSE: {bo_best['rmse_68x']:.2f}")
        print(f"  960 RMSE: {bo_best['rmse_960']:.2f}")
    
    # Convergence trend
    print("\n" + "-" * 65)
    print("[Convergence Trend (First 10 vs Last 10)]")
    print("-" * 65)
    if len(df) >= 10:
        first_10_avg = df.head(10)['rmse'].mean()
        last_10_avg = df.tail(10)['rmse'].mean()
        print(f"First 10 iterations Avg Combined Loss: {first_10_avg:.2f}")
        print(f"Last 10 iterations Avg Combined Loss: {last_10_avg:.2f}")
        improvement = ((first_10_avg - last_10_avg) / first_10_avg) * 100
        if improvement > 0:
            print(f"Improvement: {improvement:.1f}%")
        else:
            print(f"Change: {improvement:.1f}% (no significant improvement)")
    
    # Summary
    print("\n" + "=" * 65)
    print("[SUMMARY AND RECOMMENDATIONS]")
    print("=" * 65)
    
    # Check constraint strategy effectiveness
    valid_runs = df[df['rmse_960'] <= 350]
    invalid_runs = df[df['rmse_960'] > 350]
    
    print(f"\n1. Constraint Strategy Effectiveness:")
    print(f"   - Iterations satisfying constraint: {len(valid_runs)} / {len(df)} ({100*len(valid_runs)/len(df):.1f}%)")
    print(f"   - Iterations violating constraint: {len(invalid_runs)} / {len(df)} ({100*len(invalid_runs)/len(df):.1f}%)")
    
    if len(valid_runs) > 0:
        best_valid_idx = valid_runs['rmse'].idxmin()
        best_valid = valid_runs.loc[best_valid_idx]
        print(f"\n2. Best Result Satisfying Constraints:")
        print(f"   Iteration: {int(best_valid['iter'])}")
        print(f"   68X RMSE: {best_valid['rmse_68x']:.2f}")
        print(f"   960 RMSE: {best_valid['rmse_960']:.2f}")
        
        # Compare with baseline
        print(f"\n3. Performance Assessment:")
        if best_valid['rmse_68x'] < 200:
            print(f"   68X: EXCELLENT (RMSE < 200)")
        elif best_valid['rmse_68x'] < 300:
            print(f"   68X: GOOD (RMSE < 300)")
        else:
            print(f"   68X: NEEDS IMPROVEMENT (RMSE >= 300)")
            
        if best_valid['rmse_960'] < 300:
            print(f"   960: EXCELLENT (RMSE < 300)")
        elif best_valid['rmse_960'] < 350:
            print(f"   960: GOOD (Within constraint)")
        else:
            print(f"   960: CONSTRAINT VIOLATED")
    
    print("\n" + "=" * 65)
    print("ANALYSIS COMPLETE")
    print("=" * 65)
    
    return df

if __name__ == "__main__":
    csv_path = r"d:\Documents\Bus Project\Sorce code\data\calibration\l1_calibration_log_20251221_005412.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    analyze_calibration_log(csv_path)
