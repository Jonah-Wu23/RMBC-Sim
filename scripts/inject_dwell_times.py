
import xml.etree.ElementTree as ET
import numpy as np
import sys
import shutil

INPUT_ROU = "sumo/routes/baseline_irn.rou.xml"
OUTPUT_ROU = "sumo/routes/baseline_irn_calibrated.rou.xml"

# Stats from analyze_dwell_times.py (Peak Hour)
MEAN_DWELL = 117.7
STD_DWELL = 67.2

def inject_dwell_times():
    print(f"Injecting dwell times into {INPUT_ROU}...")
    try:
        tree = ET.parse(INPUT_ROU)
        root = tree.getroot()
        
        # We need to find <vehicle> or <flow> -> <route> -> <stop> ??
        # Or usually <route> is defined separately, and vehicle references it.
        # But commonly stops are defined *within* the vehicle/flow definition or referencing a route.
        # Let's inspect specific structure of baseline_irn.rou.xml first?
        # Assuming standard structure: <vehicle ...> <stop .../> </vehicle> or <flow ...> <stop .../> </flow>
        
        # Iterating all 'stop' elements
        count = 0
        for stop in root.iter('stop'):
            # Check if it's a bus stop (usually has 'busStop' attribute)
            if 'busStop' in stop.attrib:
                # Generate a random duration from LogNormal distribution usually fits dwell times better,
                # but let's use truncated Normal for simplicity as strictly requested "Mean 117s".
                # Actually, dwell times are strictly positive.
                
                # Using Normal distribution
                # val = np.random.normal(MEAN_DWELL, STD_DWELL)
                
                # Using LogNormal to avoid negatives and have a long tail
                # Calculate mu and sigma for LogNormal from Mean and StdDev
                # mean = exp(mu + sigma^2/2)
                # var = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
                
                phi = np.sqrt(STD_DWELL**2 + MEAN_DWELL**2)
                mu = np.log(MEAN_DWELL**2 / phi)
                sigma = np.sqrt(np.log(phi**2 / MEAN_DWELL**2))
                
                val = np.random.lognormal(mu, sigma)
                
                # Clip reasonable bounds (e.g. min 10s, max 300s)
                val = max(10, min(val, 300))
                
                stop.set('duration',f"{val:.2f}")
                count += 1
        
        tree.write(OUTPUT_ROU, encoding='utf-8', xml_declaration=True)
        print(f"Injected dwell times for {count} stops. Saved to {OUTPUT_ROU}")
        
    except Exception as e:
        print(f"Error processing route file: {e}")

if __name__ == "__main__":
    inject_dwell_times()
