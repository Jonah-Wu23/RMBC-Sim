
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_data import load_sim_data, load_route_stop_dist, build_sim_trajectory, load_real_link_speeds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_links', required=True) # Used to get timestamps derived if needed, or we load station_eta directly
    parser.add_argument('--real_dist', required=True, help="Path to kmb_route_stop_dist.csv")
    parser.add_argument('--real_eta', required=True, help="Path to station_eta.csv")
    parser.add_argument('--sim', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--route', default='68X')
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="talk")
    
    # 1. Load Real Data
    # We need arrival times at each stop to calc headway
    # station_eta.csv has capture_ts. We need to infer arrival.
    # Simpler approach: Use the 'link_speeds.csv' which has 'arrival_ts' at 'to_seq'.
    # This gives us arrival times at every stop except the first one.
    
    # Load Distance Data (Needed for Coverage Check)
    dist_df = load_route_stop_dist(args.real_dist) # Note: Argument needs to be added to parser if not present?
    # args.real_dist is NOT currently in plot_headway parser. Check main args.
    # It seems we need to add real_dist argument first.
    dist_df = dist_df[dist_df['route'] == args.route]

    sim_raw = load_sim_data(args.sim)
    
    # Identify Sim Range (Research Area)
    try:
        from common_data import get_sim_coverage
        min_sim_seq, max_sim_seq = get_sim_coverage(sim_raw, dist_df)
    except ImportError:
         min_sim_seq = None

    real_links = load_real_link_speeds(args.real_links)
    real_links = real_links[real_links['route'] == args.route]
    
    # Filter using Coverage and Distance Limit
    if min_sim_seq is not None:
        dist_map = dist_df.set_index('seq')['cum_dist_m'].to_dict()
        start_dist = dist_map.get(min_sim_seq, 0)
        sim_max_abs = dist_map.get(max_sim_seq, start_dist)
        effective_end_dist = min(sim_max_abs, start_dist + 5000)
        
        print(f"Aligning Headway to Research Area: Seq {min_sim_seq}-{max_sim_seq} (Dist <= {effective_end_dist:.1f})")
        
        # real_links uses 'to_seq' for stop arrival.
        real_links['dist_to'] = real_links['to_seq'].map(dist_map)
        real_links = real_links[
            (real_links['to_seq'] >= min_sim_seq) & 
            (real_links['to_seq'] <= max_sim_seq) &
            (real_links['dist_to'] <= effective_end_dist)
        ]

    # Timezone conversion (UTC -> HK)
    if real_links['arrival_ts'].dt.tz is None:
         pass 
    else:
         real_links['arrival_ts'] = real_links['arrival_ts'].dt.tz_convert('Asia/Hong_Kong')
         
    real_links = real_links[(real_links['arrival_ts'].dt.hour == 17)]
    
    real_headways = []
    # Group by 'to_seq' (Stop)
    for stop_seq, group in real_links.groupby('to_seq'):
        group = group.sort_values('arrival_ts')
        hw = group['arrival_ts'].diff().dt.total_seconds()
        real_headways.extend(hw.dropna().tolist())
        
    # 2. Load Sim Data
    sim_raw = load_sim_data(args.sim)
    # Filter for route 68X
    sim_raw = sim_raw[sim_raw['vehicle_id'].str.contains(args.route)]
    
    sim_headways = []
    # Group by 'bus_stop_id'
    for stop_id, group in sim_raw.groupby('bus_stop_id'):
        group = group.sort_values('started')
        hw = group['started'].diff() # already in seconds
        sim_headways.extend(hw.dropna().tolist())

    # 3. Combine
    df_real = pd.DataFrame({'Headway [s]': real_headways, 'Source': 'Real World'})
    df_sim = pd.DataFrame({'Headway [s]': sim_headways, 'Source': 'Simulation'})
    
    # Filter outliers > 1800s (30 mins)
    df_real = df_real[df_real['Headway [s]'] < 1800]
    df_sim = df_sim[df_sim['Headway [s]'] < 1800]
    
    full_df = pd.concat([df_real, df_sim])
    
    if full_df.empty:
        print("No headway data found.")
        return

    # 4. Plot
    plt.figure(figsize=(10, 6))
    
    # Unified colors: Real=#1f77b4, Sim=#ff7f0e
    custom_palette = {'Real World': '#1f77b4', 'Simulation': '#ff7f0e'}
    
    sns.histplot(data=full_df, x="Headway [s]", hue="Source", 
                 kde=True, element="step", stat="density", common_norm=False,
                 palette=custom_palette, alpha=0.4)
    
    plt.title(f"Headway Distribution Comparison ({args.route})")
    plt.xlim(0, 1200) # Limit to 20 mins
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(args.out, dpi=400)
    print(f"Saved Headway Plot to {args.out}")

if __name__ == "__main__":
    main()
