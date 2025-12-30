
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_data import load_sim_data, load_route_stop_dist, build_sim_trajectory, load_real_link_speeds, get_dist_map

def interpolate_points(row, num_points=10):
    d1, d2 = row['dist_start'], row['dist_end']
    t1, t2 = row['time_start'], row['time_end']
    
    if pd.isna(t1) or pd.isna(t2) or t1 == t2:
        return pd.DataFrame()
    
    # Check if timestamps
    is_ts = isinstance(t1, pd.Timestamp)
    
    if is_ts:
        t1_h = t1.hour + t1.minute/60 + t1.second/3600
        t2_h = t2.hour + t2.minute/60 + t2.second/3600
    else:
        t1_h, t2_h = t1, t2

    if t1_h > t2_h: return pd.DataFrame()

    d_vals = np.linspace(d1, d2, num_points)
    t_vals = np.linspace(t1_h, t2_h, num_points)
    
    return pd.DataFrame({
        'Distance (m)': d_vals,
        'Time of Day (HH:MM)': t_vals,
        'Speed (km/h)': row['speed_kmh'],
        'TravelTime': row.get('travel_time_s', 0)
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_links', required=True)
    parser.add_argument('--real_dist', required=True)
    parser.add_argument('--sim', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--route', default='68X')
    parser.add_argument('--hour', type=int, default=17, help='Start hour of analysis (e.g. 17 for 17:00-18:00)')
    parser.add_argument('--ghost', action='store_true', help='Enable 3-panel Ghost/Clean/Sim plot')
    parser.add_argument('--t_critical', type=float, default=325.0, help='Critical time threshold for ghost filtering')
    args = parser.parse_args()

    # sns.set_theme(style="whitegrid", context="talk") # Removed to enforce own style
    
    # IEEE Style Configuration
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    
    dist_df = load_route_stop_dist(args.real_dist)
    dist_df = dist_df[dist_df['route'] == args.route]
    # 使用 get_dist_map 处理 cum_dist_m_dir 的 NaN
    seq_dist_map = get_dist_map(dist_df, 'seq')
    
    real_links = load_real_link_speeds(args.real_links)
    real_links = real_links[real_links['route'] == args.route]
    # Timezone conversion (UTC -> HK)
    if real_links['departure_ts'].dt.tz is None:
         pass
    else:
         real_links['departure_ts'] = real_links['departure_ts'].dt.tz_convert('Asia/Hong_Kong')
         real_links['arrival_ts'] = real_links['arrival_ts'].dt.tz_convert('Asia/Hong_Kong')

    real_links['h'] = real_links['departure_ts'].dt.hour
    real_links = real_links[real_links['h'] == args.hour] # Filter Hour
    
    if real_links.empty:
        print(f"Warning: Real data empty after filtering for Hour {args.hour} HK.")  
    sim_raw = load_sim_data(args.sim)
    sim_traj = build_sim_trajectory(sim_raw, dist_df)
    sim_start_offset_sec = 0 # Assume sim outputs absolute time or aligned relative
    # Actually, logic requires verifying SIM start time. 
    # For now, we align SIM 0 to 17:00 (17.0 hours)
    SIM_START_HOUR = float(args.hour)

    plot_data_list = []
    
    # Real Processing
    # Real Processing
    # STRICT FILTERING: Only show Real World segments that exist in Simulation (The "Research Area")
    # 1. Identify valid Sim segments
    valid_sim_segments = set()
    for vid, group in sim_traj.groupby('vehicle_id'):
        group = group.sort_values('seq')
        group['next_seq'] = group['seq'].shift(-1)
        for _, row in group.dropna(subset=['next_seq']).iterrows():
            s1, s2 = int(row['seq']), int(row['next_seq'])
            valid_sim_segments.add((s1, s2))
            
    # 2. Process Real Data with Filter
    for _, row in real_links.iterrows():
        s1, s2 = row['from_seq'], row['to_seq']
        
        # KEY CHANGE: Filter out segments not in Sim
        if (s1, s2) not in valid_sim_segments:
            continue
            
        if s1 in seq_dist_map and s2 in seq_dist_map:
            seg = {
                'dist_start': seq_dist_map[s1], 'dist_end': seq_dist_map[s2],
                'time_start': row['departure_ts'], 'time_end': row['arrival_ts'],
                'speed_kmh': row['speed_kmh'],
                'travel_time_s': row.get('travel_time_s', 0)
            }
            pts = interpolate_points(seg, 15)
            if not pts.empty:
                pts['Source'] = 'Real World'
                plot_data_list.append(pts)

    # Sim Processing
    for vid, group in sim_traj.groupby('vehicle_id'):
        group = group.sort_values('seq')
        group['next_seq'] = group['seq'].shift(-1)
        group['next_arr'] = group['arrival_time'].shift(-1)
        group['curr_dep'] = group['departure_time']
        
        for _, row in group.dropna(subset=['next_seq']).iterrows():
            s1, s2 = row['seq'], int(row['next_seq'])
            if s1 in seq_dist_map and s2 in seq_dist_map:
                dist = seq_dist_map[s2] - seq_dist_map[s1]
                t_diff = row['next_arr'] - row['curr_dep']
                if t_diff > 0 and dist > 0:
                    speed = (dist / t_diff) * 3.6
                    # Convert sim sec to hour float
                    t1 = SIM_START_HOUR + (row['curr_dep'] / 3600)
                    t2 = SIM_START_HOUR + (row['next_arr'] / 3600)
                    
                    seg = {'dist_start': seq_dist_map[s1], 'dist_end': seq_dist_map[s2],
                           'time_start': t1, 'time_end': t2, 'speed_kmh': speed}
                    pts = interpolate_points(seg, 15)
                    pts['Source'] = 'Simulation'
                    plot_data_list.append(pts)

    if not plot_data_list:
        print("No data to plot.")
        return

    full_df = pd.concat(plot_data_list)
    
    # Identify Sim Range (Research Area) using common utility
    try:
        from common_data import get_sim_coverage
        min_sim_seq, max_sim_seq = get_sim_coverage(sim_raw, dist_df)
    except ImportError:
        # Fallback if common_data not reloadable or path issue (should not happen in this env)
        sim_filtered = sim_raw[sim_raw['vehicle_id'].str.contains(args.route)]
        sim_traj_temp = build_sim_trajectory(sim_filtered, dist_df)
        sim_seqs = sim_traj_temp['seq'].dropna().astype(int).unique()
        if len(sim_seqs) == 0:
            print("No sim data found.")
            return
        min_sim_seq, max_sim_seq = sim_seqs.min(), sim_seqs.max()
        
    if min_sim_seq is None:
        print(f"No stops found for route {args.route} in simulation coverage.")
        return
    
    print(f"Spacetime Filter: Research Area Seq {min_sim_seq} to {max_sim_seq}")
    
    # Calculate Distances for strict cropping
    # 使用 get_dist_map 处理 NaN
    dist_map = get_dist_map(dist_df, 'seq')
    start_dist = dist_map.get(min_sim_seq, 0)
    
    # Determine bounds: Min(SimMax, Start+5000)
    sim_max_abs = dist_map.get(max_sim_seq, start_dist)
    effective_end_dist = min(sim_max_abs, start_dist + 5000)
    
    print(f"Spacetime Dist Bounds: [{start_dist}, {effective_end_dist}]")
    
    # Filter Real Data
    # Real data has 'Distance [m]' column created earlier? 
    # Let's check where 'Distance [m]' comes from.
    # It comes from `full_df`. `real_plot_rows` uses `dist - start_dist`? No, currently absolute.
    # Let's check `build_real_plot_data` logic (not shown, but I assume it makes absolute).
    # Wait, `full_df` comes from `plot_data_list`.
    # Let's filter `real_links` based on distance (using sequence mapping) BEFORE creating plot data if possible,
    # OR filter `full_df` after creation. Filtering `full_df` is easier if it has "Distance [m]".
    
    # Actually, `full_df` is built from chunks. 
    # Real data is built iterating `real_links`.
    # Let's clean up logic: Filter `real_links` by sequence first (done above), 
    # BUT also we need to filter by distance < effective_end_dist.
    
    # Filter Sim Data
    # Sim data `sim_traj` has `cum_dist_m`. Filtering it is good.
    sim_traj = sim_traj[(sim_traj['cum_dist_m'] >= start_dist) & (sim_traj['cum_dist_m'] <= effective_end_dist)]
    
    # Filter Real Data (Links)
    # We map 'from_seq' to dist.
    real_links['dist_from'] = real_links['from_seq'].map(dist_map)
    real_links = real_links[(real_links['dist_from'] >= start_dist) & (real_links['dist_from'] <= effective_end_dist)]

    # Now rebuild full_df or re-run processing?
    # The existing code builds `plot_data_list` from `real_links` and `sim_traj`.
    # So if we filter them here, the plotting loop below (which iterates them) will be correct.
    # BUT wait, the previous code block for `plot_data_list` preparation might be BEFORE this replacement?
    # I am editing *after* `full_df = pd.concat(...)`.
    # Ah, `full_df` is ALREADY created. I need to filter `full_df`.
    
    full_df = full_df[
        (full_df['Distance (m)'] >= start_dist) & 
        (full_df['Distance (m)'] <= effective_end_dist)
    ]
    
    # Plotting Configuration
    if args.ghost:
        nrows = 3
        fig_height = 5.0
        # Config: (Title, Source_Key, Mode, Colormap)
        sources_cfg = [
            ('Ghost System (Real Raw)', 'Real World', 'all', 'Greys'),
            ('Real (Op-L2-v1.1 / Rule C)', 'Real World', 'clean', 'Blues'),
            ('Simulation', 'Simulation', 'all', 'Oranges')
        ]
    else:
        nrows = 2
        fig_height = 3.5
        sources_cfg = [
            ('Real (Op-L2-v1.1)', 'Real World', 'clean', 'Blues'), 
            ('Simulation', 'Simulation', 'all', 'Oranges')
        ]

    fig, axes = plt.subplots(nrows, 1, figsize=(3.5, fig_height), sharex=True, sharey=True, constrained_layout=True)
    if nrows == 1: axes = [axes]
    
    # Unified Scale but Different Colormaps
    vmin, vmax = 0, 70
    
    def get_truncated_cmap(name, min_val=0.4, max_val=1.0):
        base_cmap = plt.get_cmap(name)
        new_colors = base_cmap(np.linspace(min_val, max_val, 256))
        return mcolors.LinearSegmentedColormap.from_list(f"trunc({name})", new_colors)

    for i, (title, src_key, mode, cmap_name) in enumerate(sources_cfg):
        ax = axes[i]
        
        # Filter Data
        df_src = full_df[full_df['Source'] == src_key].copy()
        
        if mode == 'clean' and src_key == 'Real World':
            # Apply Rule C: Tick out Ghost Jams
            # Ghost: Time > 300 AND Speed < 5.
            # Clean: NOT (Time > 325 AND Speed < 5)
            # Note: args.t_critical default 325.
            mask_ghost = (df_src['TravelTime'] > args.t_critical) & (df_src['Speed (km/h)'] < 5.0)
            df_src = df_src[~mask_ghost]
        
        if df_src.empty: continue

        norm = plt.Normalize(vmin, vmax)
        
        # Use Truncated Colormap to ensure visibility (min alpha/intensity 0.4)
        # This is necessary because 'Blues'/'Oranges' start at white (invisible)
        trunc_cmap = get_truncated_cmap(cmap_name)
        
        sc = ax.scatter(df_src["Distance (m)"], df_src["Time of Day (HH:MM)"], 
                        c=df_src["Speed (km/h)"], cmap=trunc_cmap, norm=norm, 
                        s=12, linewidth=0, marker='o', alpha=1.0)
        
        ax.set_title(title, color='black', fontweight='bold', fontsize=8)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):02d}:{int((x-int(x))*60):02d}"))
        ax.set_ylim(args.hour, args.hour + 1.2)
        ax.set_ylabel("Time (HH:MM)", fontsize=8)
        
        if i == nrows - 1:
            ax.set_xlabel("Distance (m)", fontsize=8)
        
        ax.grid(True, linestyle='--', alpha=0.5)

        cbar = fig.colorbar(sc, ax=ax, pad=0.02, aspect=15)
        cbar.set_label('Speed (km/h)', fontsize=8)
        cbar.ax.tick_params(labelsize=8)

    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
