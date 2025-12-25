
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
    parser.add_argument('--real_links', required=True)
    parser.add_argument('--real_dist', required=True)
    parser.add_argument('--sim', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--route', default='68X')
    args = parser.parse_args()

    # sns.set_theme(style="ticks", context="talk") # Disabled for IEEE style
    
    # IEEE Style Configuration (8pt fonts with small figure size)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7
    
    # Load Data
    dist_df = load_route_stop_dist(args.real_dist)
    dist_df = dist_df[dist_df['route'] == args.route]
    # Map Seq to Name/ID for X-axis
    # Maybe use 'stop_name_en' but it's long. Use ID or Seq.
    
    real_links = load_real_link_speeds(args.real_links)
    real_links = real_links[real_links['route'] == args.route]
    
    # Timezone conversion (UTC -> HK)
    if real_links['arrival_ts'].dt.tz is None:
         # Assume UTC if naive, or check data source
         pass 
    else:
         real_links['arrival_ts'] = real_links['arrival_ts'].dt.tz_convert('Asia/Hong_Kong')

    # Filter 17:00-18:00 HK Time
    real_links = real_links[real_links['arrival_ts'].dt.hour == 17]

    if real_links.empty:
        print("Warning: No Real Link data found for Hour 17. Checking full dataset...")
        # Fallback or debug print
        print(load_real_link_speeds(args.real_links).head())
        # return # Option: exit or try plotting what we have
    
    # SIM DATA
    sim_raw = load_sim_data(args.sim)
    
    # Identify Sim Range (Research Area) using common utility
    min_sim_seq, max_sim_seq = None, None
    try:
        from common_data import get_sim_coverage
        min_sim_seq, max_sim_seq = get_sim_coverage(sim_raw, dist_df)
    except ImportError:
         pass 

    if min_sim_seq is not None: 
        # Calculate Effective Bounds
        dist_map = dist_df.set_index('seq')['cum_dist_m'].to_dict()
        start_dist = dist_map.get(min_sim_seq, 0)
        sim_max_abs = dist_map.get(max_sim_seq, start_dist)
        effective_end_dist = min(sim_max_abs, start_dist + 5000)
        
        print(f"Aligning Boxplot to Research Area: Seq {min_sim_seq}-{max_sim_seq} (Dist <= {effective_end_dist:.1f})")
        
        # Filter Real Links
        real_links['dist_from'] = real_links['from_seq'].map(dist_map)
        real_links = real_links[
            (real_links['from_seq'] >= min_sim_seq) & 
            (real_links['from_seq'] <= max_sim_seq) &
            (real_links['dist_from'] <= effective_end_dist)
        ]
    
    sim_traj = build_sim_trajectory(sim_raw, dist_df)
    
    # Calculate Sim Link Speeds
    sim_link_data = []
    dist_map = dist_df.set_index('seq')['cum_dist_m'].to_dict()
    
    for vid, group in sim_traj.groupby('vehicle_id'):
        group = group.sort_values('seq')
        group['next_seq'] = group['seq'].shift(-1)
        group['next_arr'] = group['arrival_time'].shift(-1)
        group['curr_dep'] = group['departure_time']
        
        for _, row in group.dropna(subset=['next_seq']).iterrows():
            s1, s2 = int(row['seq']), int(row['next_seq'])
            # Relaxed condition: just need s2 > s1 (moving forward) and both in map
            if s2 > s1 and s1 in dist_map and s2 in dist_map: 
                dist = dist_map[s2] - dist_map[s1]
                t = row['next_arr'] - row['curr_dep']
                if t > 0 and dist > 0:
                    speed = (dist/t) * 3.6
                    # Assign this speed to the starting link (s1)
                    # Note: If s2-s1 > 1, this speed covers multiple links, but for boxplot we assign to s1
                    sim_link_data.append({'seq': s1, 'speed_kmh': speed, 'Source': 'Simulation'})

    # Prepare Real Data
    real_link_data = []
    for _, row in real_links.iterrows():
        real_link_data.append({'seq': row['from_seq'], 'speed_kmh': row['speed_kmh'], 'Source': 'Real World'})
        
    df_real = pd.DataFrame(real_link_data)
    df_sim = pd.DataFrame(sim_link_data)
    
    if df_real.empty:
        print("Error: df_real is empty. Cannot determine top variance links.")
        full_df = df_sim # Plot only sim?
        top_vars = []
    else:
        full_df = pd.concat([df_real, df_sim])
        # Filter for Key Links
        top_vars = df_real.groupby('seq')['speed_kmh'].std().sort_values(ascending=False).head(8).index.tolist()
        top_vars.sort()

    if not top_vars:
        if not df_sim.empty:
             print("Plotting all available Sim/Real links (limited)")
             top_vars = df_sim['seq'].unique().tolist()[:10]
        else:
             print("No data to plot.")
             return

    
    filtered_df = full_df[full_df['seq'].isin(top_vars)]
    
    plt.figure(figsize=(7.16, 3.0))  # IEEE double column width
    
    # Unified colors
    custom_palette = {'Real World': '#1f77b4', 'Simulation': '#ff7f0e'}
    
    # Boxplot
    sns.boxplot(data=filtered_df, x="seq", y="speed_kmh", hue="Source", 
                palette=custom_palette, width=0.6, fliersize=3)
    
    plt.xlabel("Link Sequence ID")
    plt.ylabel("Speed (km/h)")
    
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Saved Boxplot to {args.out}")

if __name__ == "__main__":
    main()
