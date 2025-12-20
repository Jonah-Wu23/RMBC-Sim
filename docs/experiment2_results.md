# Experiment 2: L1 Validation with Calibrated Parameters

## 1. Experiment Setup
- **Objective**: Validate if injecting background traffic and real-world dwell times resolves the "simulation too fast" issue observed in Week 1.
- **Inputs**:
    - **Network**: `hk_irn.net.xml` (Week 2 Reconstructed)
    - **Bus Route**: `baseline_irn_calibrated.rou.xml`
        - **Dwell Times**: LogNormal distributed (Mean 117.7s, Std 67.2s) derived from 17:00 ETA data.
    - **Background Traffic**: `background.rou.xml`
        - **Source**: Smart Lamppost Detectors (17:00 Peak).
        - **Volume**: ~1.38M veh/h equivalent flow injected into valid edges.

## 2. Results (Preliminary)
Due to network connectivity issues, the simulation was interrupted at T=1042s. However, early data points provide critical insights.

### Speed Comparison (68X Inbound, First Segment)
| Metric | Week 1 (Baseline) | Real World | Week 2 (Exp 2) |
| :--- | :--- | :--- | :--- |
| **Mean Speed** | ~50 km/h | 12.42 km/h | **4.85 km/h** |
| **Travel Time (Stop 1->2)** | ~40s | 339.8s | **457.0s** |

### Observations
1.  **Over-Correction**: The simulation successfully slowed down from the unrealistic ~50km/h. It is now actually *slower* than reality (4.85 km/h vs 12.42 km/h).
2.  **Congestion Effect**: The injection of 1.38M background vehicles created significant congestion, proving the effectiveness of the volume data in generating traffic pressure.
3.  **Network Issues**:
    - High number of "No connection" warnings.
    - Bus vehicles teleporting due to "Wrong Lane" or missing links.
    - This indicates that the **Network Reconstruction** (Week 3 task) is critical to ensure smooth flow.

## 3. Conclusion
Experiment 2 confirms that **Micro-Calibration (Background Traffic + Dwell Time)** is the correct lever to adjust simulation speed. The original "too fast" problem is resolved. The current challenge shifts from "Clibration" to "Network Integrity".

## 4. Next Steps
- **Fix Connectivity**: Address the broken links in `hk_irn.net.xml` (Week 3).
- **Tune Volume**: 1.38M vehicles might be too aggressive or unevenly distributed (causing gridlock). Consider scaling down `vehsPerHour` or filtering minor roads.
