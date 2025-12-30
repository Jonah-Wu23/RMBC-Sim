"""
Scale only background traffic routes, preserving bus routes at 100%.
Deterministic downsampling for reproducibility.

Usage:
    python scripts/tools/scale_background_routes.py --input IN --output OUT --alpha 0.9
"""

import xml.etree.ElementTree as ET
import hashlib
import argparse
import os


def scale_background(in_xml, out_xml, alpha: float, seed_tag="deterministic"):
    """
    Scale background traffic file by factor alpha.
    - Scales flow vehsPerHour/probability/number attributes
    - Deterministically downsamples explicit <vehicle> elements
    """
    tree = ET.parse(in_xml)
    root = tree.getroot()
    
    scaled_flows = 0
    removed_vehicles = 0
    total_vehicles = 0

    # 1) Scale <flow> elements
    for flow in root.findall("flow"):
        if "vehsPerHour" in flow.attrib:
            old_val = float(flow.get("vehsPerHour"))
            flow.set("vehsPerHour", str(round(old_val * alpha, 2)))
            scaled_flows += 1
        if "probability" in flow.attrib:
            p = float(flow.get("probability")) * alpha
            flow.set("probability", str(min(1.0, round(p, 4))))
            scaled_flows += 1
        if "number" in flow.attrib:
            n = int(round(float(flow.get("number")) * alpha))
            flow.set("number", str(max(0, n)))
            scaled_flows += 1

    # 2) Deterministically downsample explicit <vehicle> elements
    vehicles = list(root.findall("vehicle"))
    total_vehicles = len(vehicles)
    for veh in vehicles:
        vid = veh.get("id", "")
        # Create deterministic hash based on vehicle ID
        h = int(hashlib.md5((seed_tag + vid).encode()).hexdigest()[:8], 16)
        u = h / 0xFFFFFFFF  # Normalize to [0, 1]
        if u > alpha:
            root.remove(veh)
            removed_vehicles += 1

    # Write output
    os.makedirs(os.path.dirname(out_xml) if os.path.dirname(out_xml) else ".", exist_ok=True)
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)
    
    print(f"Scaled background traffic by alpha={alpha}")
    print(f"  Input: {in_xml}")
    print(f"  Output: {out_xml}")
    print(f"  Flows scaled: {scaled_flows}")
    print(f"  Vehicles: {total_vehicles} -> {total_vehicles - removed_vehicles} (removed {removed_vehicles})")
    
    return scaled_flows, total_vehicles - removed_vehicles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale background traffic routes.")
    parser.add_argument("--input", "-i", required=True, help="Input route file")
    parser.add_argument("--output", "-o", required=True, help="Output route file")
    parser.add_argument("--alpha", "-a", type=float, default=0.9, help="Scale factor (default: 0.9)")
    parser.add_argument("--seed", default="deterministic", help="Seed tag for reproducibility")
    
    args = parser.parse_args()
    scale_background(args.input, args.output, args.alpha, args.seed)
