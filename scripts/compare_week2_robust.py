import pandas as pd
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt

def analyze_route(target_route, target_bound):
    print(f"\n--- Analyzing Route: {target_route} {target_bound} ---")
    
    # Files
    real_link_file = r"data/processed/enriched_link_stats.csv"
    sim_stopinfo = r"sumo/output/stopinfo_exp2_cropped.xml"
    route_dist_file = r"data/processed/kmb_route_stop_dist.csv"
    output_plot = f"sumo/output/week2_cropped_trajectory_robust_{target_route}.png"
    
    if not os.path.exists(sim_stopinfo):
        print(f"Error: {sim_stopinfo} not found.")
        return

    # 1. Reference Data (Route Structure)
    df_dist = pd.read_csv(route_dist_file)
    df_ref = df_dist[(df_dist['route'] == target_route) & (df_dist['bound'] == target_bound)].sort_values('seq').copy()
    
    if df_ref.empty:
        print(f"Warning: No reference distance data for {target_route}")
        return

    # Map stop_id to info
    stop_info_map = df_ref.set_index('stop_id')[['seq', 'cum_dist_m']].to_dict('index')
    
    # 2. Real Travel Times
    df_real = pd.read_csv(real_link_file)
    df_real = df_real[(df_real['route'] == target_route) & (df_real['bound'] == target_bound)]
    
    if df_real.empty:
        print(f"Warning: No real travel time data for {target_route}")
        # Proceeding might fail if real_travel_map is empty, but we'll try
        
    real_travel_map = df_real.set_index(['from_seq', 'to_seq'])['travel_time_s'].to_dict()

    # 3. Simulation Data
    tree = ET.parse(sim_stopinfo)
    root = tree.getroot()
    
    target_stop_ids = set(df_ref['stop_id'].values)
    veh_data = {} 
    
    for stop in root.findall('stopinfo'):
        s_id = stop.get('busStop')
        if s_id in target_stop_ids:
            v_id = stop.get('id')
            arrival = float(stop.get('started'))
            if v_id not in veh_data: veh_data[v_id] = {}
            veh_data[v_id][s_id] = arrival
            
    if not veh_data:
        print("No simulation data found for target route stops.")
        return

    # 4. Alignment
    all_sim_seqs = []
    for stops in veh_data.values():
        for s_id in stops:
            if s_id in stop_info_map:
                all_sim_seqs.append(stop_info_map[s_id]['seq'])
    
    if not all_sim_seqs:
        print("Stops found in sim do not match reference data mapping.")
        return
        
    start_seq = min(all_sim_seqs)
    print(f"Aligning trajectory at Sequence {start_seq} (Start of Cropped Area)")
    
    # 5. Build Trajectories
    # SIMULATION CURVE
    sim_plot_data = {} 
    
    for v_id, stops in veh_data.items():
        start_stop_id = None
        for s, info in stop_info_map.items():
            if info['seq'] == start_seq:
                start_stop_id = s
                break
        
        if start_stop_id and start_stop_id in stops:
            t0 = stops[start_stop_id]
            for s_id, t in stops.items():
                if s_id in stop_info_map:
                    seq = stop_info_map[s_id]['seq']
                    if seq >= start_seq:
                        dt = t - t0
                        if seq not in sim_plot_data: sim_plot_data[seq] = []
                        sim_plot_data[seq].append(dt)
    
    if not sim_plot_data:
        print(f"No vehicles found starting exactly at seq {start_seq}.")
        return

    sim_curve = [] 
    for seq in sorted(sim_plot_data.keys()):
        avg_time = sum(sim_plot_data[seq]) / len(sim_plot_data[seq])
        dist = 0
        for s, info in stop_info_map.items():
            if info['seq'] == seq:
                dist = info['cum_dist_m']
                break
        sim_curve.append((dist, avg_time))
        
    # REAL CURVE
    real_curve = []
    
    start_dist = 0
    for s, info in stop_info_map.items():
        if info['seq'] == start_seq:
            start_dist = info['cum_dist_m']
            break
            
    real_curve.append((start_dist, 0.0))
    
    sorted_ref_seqs = sorted([s for s in df_ref['seq'] if s >= start_seq])
    
    curr_time = 0.0
    for i in range(len(sorted_ref_seqs) - 1):
        s_from = sorted_ref_seqs[i]
        s_to = sorted_ref_seqs[i+1]
        
        if (s_from, s_to) in real_travel_map:
            dt = real_travel_map[(s_from, s_to)]
        else:
            dt = 0 
            
        curr_time += dt
        
        d_next = 0
        for s, info in stop_info_map.items():
            if info['seq'] == s_to:
                d_next = info['cum_dist_m']
                break
        
        real_curve.append((d_next, curr_time))

    # Truncate Real Curve
    if sim_curve:
        max_sim_dist = sim_curve[-1][0]
        real_curve = [pt for pt in real_curve if pt[0] <= max_sim_dist + 10]

    # 6. Plotting
    plt.figure(figsize=(10, 6))
    
    xr, yr = zip(*real_curve) if real_curve else ([], [])
    xs, ys = zip(*sim_curve) if sim_curve else ([], [])
    
    xr = [x - start_dist for x in xr]
    xs = [x - start_dist for x in xs]
    
    plt.plot(xr, yr, 'b-o', label='Real World (Avg)', alpha=0.7)
    plt.plot(xs, ys, 'r-s', label='Simulation (Week 2)', linewidth=2)
    
    plt.xlabel(f'Distance from Entry Stop (Seq {start_seq}) [m]')
    plt.ylabel('Cumulative Travel Time [s]')
    plt.title(f'Trajectory Comparison: {target_route} {target_bound}\n(Cropped Network, Start Seq {start_seq})')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    
    # Report
    if xs and xr:
        final_sim_t = ys[-1]
        max_sim_dist = xs[-1]
        
        final_real_t = 0
        if len(xr) > 0:
             final_real_t = yr[-1]
        
        print("\n--- Summary ---")
        print(f"Traversed Distance: {max_sim_dist:.1f} m")
        print(f"Sim Time: {final_sim_t:.1f} s")
        print(f"Real Time (approx): {final_real_t:.1f} s")
        if final_real_t > 0:
            diff = final_sim_t - final_real_t
            print(f"Difference: {diff:.1f} s ({diff/final_real_t*100:.1f}%)")

def compare_week2_robust():
    analyze_route("68X", "inbound")
    analyze_route("960", "inbound")

if __name__ == "__main__":
    compare_week2_robust()
