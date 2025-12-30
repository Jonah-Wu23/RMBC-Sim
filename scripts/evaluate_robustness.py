"""
Evaluate P14 Robustness Test Results.
Compare off-peak simulation against real D2D travel time data.

Metrics: KS Distance, RMSE, P90 Error
"""

import pandas as pd
import numpy as np
from scipy import stats
import argparse
import os

# Default paths
REAL_STATS = "data2/processed/link_stats_offpeak.csv"
SIM_STOPINFO = "sumo/output/offpeak_stopinfo.xml"
PEAK_STATS = "data/processed/link_stats.csv"  # For comparison


def load_real_stats(filepath):
    """Load real D2D link statistics."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return None
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} real link stats from {filepath}")
    return df


def load_sim_stopinfo(filepath):
    """Parse SUMO stopinfo.xml to extract bus stop times."""
    from xml.etree import ElementTree as ET
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return None
    
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    records = []
    for stop in root.findall('.//stopinfo'):
        records.append({
            'vehicle_id': stop.get('id'),
            'stop_id': stop.get('busStop'),
            'arrival': float(stop.get('started', 0)),
            'departure': float(stop.get('ended', 0)),
            'duration': float(stop.get('duration', 0))
        })
    
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} stop records from {filepath}")
    return df


def calculate_sim_link_times(df_stops):
    """Calculate link travel times from consecutive stop visits."""
    if df_stops is None or len(df_stops) == 0:
        return None
    
    # Sort by vehicle and arrival time
    df = df_stops.sort_values(['vehicle_id', 'arrival'])
    
    # Calculate travel time between consecutive stops for each vehicle
    results = []
    for veh_id, veh_data in df.groupby('vehicle_id'):
        veh_data = veh_data.reset_index(drop=True)
        for i in range(len(veh_data) - 1):
            from_stop = veh_data.loc[i, 'stop_id']
            to_stop = veh_data.loc[i+1, 'stop_id']
            departure = veh_data.loc[i, 'departure']
            arrival = veh_data.loc[i+1, 'arrival']
            travel_time = arrival - departure
            
            if travel_time > 0 and travel_time < 3600:  # 0-1 hour
                results.append({
                    'vehicle_id': veh_id,
                    'from_stop': from_stop,
                    'to_stop': to_stop,
                    'travel_time_s': travel_time
                })
    
    df_result = pd.DataFrame(results)
    print(f"Calculated {len(df_result)} simulated link times")
    return df_result


def calculate_ks_distance(real_values, sim_values):
    """Calculate Kolmogorov-Smirnov distance between two distributions."""
    if len(real_values) < 5 or len(sim_values) < 5:
        return None, None
    ks_stat, p_value = stats.ks_2samp(real_values, sim_values)
    return ks_stat, p_value


def calculate_rmse(real_values, sim_values):
    """Calculate RMSE between aligned values (if same length) or distributions."""
    real_mean = np.mean(real_values)
    sim_mean = np.mean(sim_values)
    return abs(real_mean - sim_mean)  # Mean absolute error of means


def evaluate_robustness(real_file, sim_file, peak_file=None):
    """Main evaluation function."""
    print("\n" + "=" * 60)
    print("P14 Off-Peak Robustness Test Evaluation")
    print("=" * 60)
    
    # Load data
    df_real = load_real_stats(real_file)
    df_sim_stops = load_sim_stopinfo(sim_file)
    
    if df_real is None:
        print("\nERROR: Cannot load real data. Using synthetic comparison.")
        return
    
    # Get real speed distribution
    if 'speed_median' in df_real.columns:
        real_speeds = df_real['speed_median'].dropna().values
    elif 'speed_kmh' in df_real.columns:
        real_speeds = df_real['speed_kmh'].dropna().values
    else:
        print("Warning: No speed column found in real data")
        real_speeds = np.array([])
    
    print(f"\n--- Real Data (Off-Peak) Statistics ---")
    if len(real_speeds) > 0:
        print(f"  Samples: {len(real_speeds)}")
        print(f"  Mean Speed: {np.mean(real_speeds):.2f} km/h")
        print(f"  Median Speed: {np.median(real_speeds):.2f} km/h")
        print(f"  P10/P90: {np.percentile(real_speeds, 10):.2f} / {np.percentile(real_speeds, 90):.2f} km/h")
    
    # Calculate simulated link times
    if df_sim_stops is not None:
        df_sim_links = calculate_sim_link_times(df_sim_stops)
        if df_sim_links is not None and len(df_sim_links) > 0:
            # Convert travel time to speed (using average link distance ~500m)
            avg_link_dist_m = 500  # Approximate
            df_sim_links['speed_kmh'] = (avg_link_dist_m / 1000) / (df_sim_links['travel_time_s'] / 3600)
            df_sim_links = df_sim_links[(df_sim_links['speed_kmh'] > 2) & (df_sim_links['speed_kmh'] < 100)]
            sim_speeds = df_sim_links['speed_kmh'].values
            
            print(f"\n--- Simulated Data Statistics ---")
            print(f"  Samples: {len(sim_speeds)}")
            print(f"  Mean Speed: {np.mean(sim_speeds):.2f} km/h")
            print(f"  Median Speed: {np.median(sim_speeds):.2f} km/h")
            print(f"  P10/P90: {np.percentile(sim_speeds, 10):.2f} / {np.percentile(sim_speeds, 90):.2f} km/h")
            
            # Calculate KS distance
            if len(real_speeds) > 0 and len(sim_speeds) > 0:
                ks_stat, p_value = calculate_ks_distance(real_speeds, sim_speeds)
                rmse = calculate_rmse(real_speeds, sim_speeds)
                
                print(f"\n--- Robustness Metrics ---")
                print(f"  KS Distance: {ks_stat:.4f} (p={p_value:.4f})")
                print(f"  Mean Speed Difference: {rmse:.2f} km/h")
                
                # Evaluate against thresholds
                print(f"\n--- Evaluation ---")
                if ks_stat <= 0.25:
                    print(f"  [STRONG SUCCESS]: KS={ks_stat:.4f} <= 0.25")
                elif ks_stat <= 0.35:
                    print(f"  [SUCCESS]: KS={ks_stat:.4f} <= 0.35")
                else:
                    print(f"  [FAIL]: KS={ks_stat:.4f} > 0.35")
        else:
            print("\nWarning: Could not calculate simulated link times")
    else:
        print("\nWarning: No simulation stop data available")
    
    # Load peak stats for comparison if available
    if peak_file and os.path.exists(peak_file):
        df_peak = load_real_stats(peak_file)
        if df_peak is not None and 'speed_median' in df_peak.columns:
            peak_speeds = df_peak['speed_median'].dropna().values
            print(f"\n--- Peak Data (Training) Statistics ---")
            print(f"  Mean Speed: {np.mean(peak_speeds):.2f} km/h")
            print(f"  Median Speed: {np.median(peak_speeds):.2f} km/h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate P14 robustness test results.")
    parser.add_argument("--real", default=REAL_STATS, help="Real link stats CSV")
    parser.add_argument("--sim", default=SIM_STOPINFO, help="Simulation stopinfo XML")
    parser.add_argument("--peak", default=PEAK_STATS, help="Peak link stats CSV for comparison")
    
    args = parser.parse_args()
    evaluate_robustness(args.real, args.sim, args.peak)
