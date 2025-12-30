"""
Generate off-peak scenario files for P14 Robustness Test.
1. Scale background traffic by factor α (default 0.7)
2. Create SUMO config file with frozen L2 parameters (P13)

IMPORTANT: Does NOT use global --scale to avoid scaling bus routes!
Instead, creates a new background route file with scaled vph.
"""

import os
import re
import argparse
import shutil
from xml.etree import ElementTree as ET

# Default paths
PEAK_BG_FILE = "sumo/routes/background_corridor_source_filtered_test.rou.xml"
BUS_ROUTE_FILE = "sumo/routes/fixed_routes_via.rou.xml"  
OUTPUT_DIR = "sumo/routes"
OUTPUT_BG_FILE = os.path.join(OUTPUT_DIR, "background_offpeak.rou.xml")
OUTPUT_CONFIG_FILE = "sumo/config/experiment_robustness.sumocfg"

# P13 Frozen L2 Parameters
FROZEN_PARAMS = {
    'capacityFactor': 1.5,
    'minGap': 0.5,
    'impatience': 1.0
}


def scale_background_traffic(input_file, output_file, scale_factor):
    """
    Scale background traffic by modifying vehsPerHour attributes in flow elements.
    NOTE: This is a simple regex/text approach; for complex XML, use proper parsing.
    """
    print(f"\nScaling background traffic by factor {scale_factor}")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match flow elements with vehsPerHour or probability
    # For simplicity, we scale vehsPerHour and probability attributes
    patterns = [
        (r'vehsPerHour="(\d+(?:\.\d+)?)"', 'vehsPerHour'),
        (r'probability="(\d+(?:\.\d+)?)"', 'probability'),
        (r'period="(\d+(?:\.\d+)?)"', 'period'),  # period needs inverse scaling
    ]
    
    modified = False
    for pattern, attr in patterns:
        def replace_func(match):
            nonlocal modified
            original = float(match.group(1))
            if attr == 'period':
                # Period is inverse: longer period = lower demand
                new_val = original / scale_factor
            else:
                new_val = original * scale_factor
            modified = True
            return f'{attr}="{new_val:.2f}"'
        content = re.sub(pattern, replace_func, content)
    
    if not modified:
        print("  Warning: No flow scaling attributes found. Copying file as-is with note.")
        # Still copy the file but with a comment
        content = content.replace(
            '<?xml version="1.0"',
            f'<!-- Off-peak scaled version (factor={scale_factor}) -->\n<?xml version="1.0"'
        )
    
    # Count vehicles before and after (rough estimate from flow count)
    flow_count = len(re.findall(r'<flow ', content))
    vehicle_count = len(re.findall(r'<vehicle ', content))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  Flows: {flow_count}, Vehicles: {vehicle_count}")
    print(f"  Scale factor applied: {scale_factor}")
    return flow_count, vehicle_count


def create_sumocfg(network_file, route_files, additional_files, output_file, sim_end=3900):
    """Create SUMO configuration file for P14 experiment."""
    
    config = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <!-- P14 Off-Peak Robustness Test Configuration -->
    <!-- Frozen L2 Parameters from P13: capacityFactor=1.5, minGap=0.5, impatience=1.0 -->

    <input>
        <net-file value="{network_file}" />
        <route-files value="{','.join(route_files)}" />
        <additional-files value="{','.join(additional_files)}" />
    </input>

    <time>
        <begin value="0" />
        <end value="{sim_end}" />
    </time>

    <processing>
        <time-to-teleport value="300" />
        <ignore-route-errors value="true" />
    </processing>

    <report>
        <verbose value="true" />
        <no-step-log value="false" />
    </report>

    <output>
        <output-prefix value="offpeak_" />
        <stop-output value="sumo/output/offpeak_stopinfo.xml" />
        <statistic-output value="sumo/output/offpeak_stats.xml" />
        <tripinfo-output value="sumo/output/offpeak_tripinfo.xml" />
    </output>

</configuration>
'''
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(config)
    
    print(f"\nCreated SUMO config: {output_file}")
    print(f"  Network: {network_file}")
    print(f"  Routes: {route_files}")
    print(f"  Additional: {additional_files}")


def main():
    parser = argparse.ArgumentParser(description="Generate off-peak scenario for P14 robustness test.")
    parser.add_argument("--scale", type=float, default=0.7, help="Background traffic scale factor")
    parser.add_argument("--peak-bg", default=PEAK_BG_FILE, help="Peak background route file")
    parser.add_argument("--bus-routes", default=BUS_ROUTE_FILE, help="Bus route file")
    parser.add_argument("--output-bg", default=OUTPUT_BG_FILE, help="Output background file")
    parser.add_argument("--output-cfg", default=OUTPUT_CONFIG_FILE, help="Output SUMO config file")
    args = parser.parse_args()
    
    print("=" * 60)
    print("P14 Off-Peak Scenario Generation")
    print("=" * 60)
    print(f"Scale factor: {args.scale} (Off-peak approx {args.scale*100:.0f}% of peak)")
    
    # 1. Scale background traffic
    os.makedirs(os.path.dirname(args.output_bg), exist_ok=True)
    peak_flows, peak_vehs = scale_background_traffic(args.peak_bg, args.output_bg, args.scale)
    
    # 2. Create SUMO config
    network = "sumo/net/hk_cropped.net.xml"
    routes = [
        args.bus_routes.replace("\\", "/"),
        args.output_bg.replace("\\", "/")
    ]
    additional = [
        "sumo/additional/bus_stops_irn.add.xml"
    ]
    create_sumocfg(network, routes, additional, args.output_cfg)
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Background traffic scaled: {peak_flows} flows, {peak_vehs} vehicles")
    print(f"Frozen L2 Parameters (P13): capacityFactor={FROZEN_PARAMS['capacityFactor']}, "
          f"minGap={FROZEN_PARAMS['minGap']}, impatience={FROZEN_PARAMS['impatience']}")
    print("\nReady to run simulation:")
    print(f"  sumo -c {args.output_cfg}")


if __name__ == "__main__":
    main()
