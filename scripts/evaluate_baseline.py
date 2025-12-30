
import os
import subprocess
import pandas as pd
import sys

def run_command(cmd):
    print(f"Running: {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"Error running command: {cmd}")
        sys.exit(ret)

def main():
    # Configuration
    REAL_LINKS = "data/processed/link_speeds.csv"
    REAL_DIST = "data/processed/kmb_route_stop_dist.csv"
    REAL_ETA = "data/processed/station_eta.csv" # For future use if needed
    REAL_ETA = "data/processed/station_eta.csv" # For future use if needed
    # Corrected Sim File: B1 Baseline (Freeflow / Default Params), generated from baseline_b1.sumocfg
    SIM_XML = "sumo/output/stopinfo_b1.xml"
    ROUTE = "68X"
    ROUTE = "68X"
    
    os.makedirs("plots", exist_ok=True)
    
    ROUTES = ['68X', '960']
    
    # Header for report
    report_content = "# 仿真基准评估报告 (Baseline Evaluation Report)\n\n"
    report_content += "## 实验背景\n本报告评估了 **Week 2 Experiment 2.3** 的仿真表现 (Baseline)。\n\n"
    report_content += "## 1. 核心指标统计 (Key Metrics)\n\n"
    
    # Loop over routes
    for ROUTE in ROUTES:
        print(f"=== Processing Route: {ROUTE} ===")
        
        # 1. Calculate Metrics
        print(f"--- {ROUTE}: Calculating Metrics ---")
        run_command(f'python scripts/metrics_calc.py --real_links "{REAL_LINKS}" --real_dist "{REAL_DIST}" --sim "{SIM_XML}" --out "docs/baseline_metrics_{ROUTE}.csv" --route {ROUTE}')
        
        # 2. Plot Space-Time Diagram
        print(f"--- {ROUTE}: Plotting Space-Time Diagram ---")
        run_command(f'python scripts/plot_spacetime.py --real_links "{REAL_LINKS}" --real_dist "{REAL_DIST}" --sim "{SIM_XML}" --out "plots/spacetime_{ROUTE}_baseline.png" --route {ROUTE}')
        
        # 3. Plot Headway
        print(f"--- {ROUTE}: Plotting Headway Distribution ---") # Note: plot_headway uses real_links for arrival times
        run_command(f'python scripts/plot_headway.py --real_links "{REAL_LINKS}" --real_dist "{REAL_DIST}" --real_eta "{REAL_ETA}" --sim "{SIM_XML}" --out "plots/headway_distribution_{ROUTE}_baseline.png" --route {ROUTE}')
        
        # 4. Plot Boxplot
        print(f"--- {ROUTE}: Plotting Link Speed Boxplot ---")
        run_command(f'python scripts/plot_link_boxplot.py --real_links "{REAL_LINKS}" --real_dist "{REAL_DIST}" --sim "{SIM_XML}" --out "plots/link_speed_boxplot_{ROUTE}_baseline.png" --route {ROUTE}')
        
        # 5. Plot Trajectory (NEW)
        print(f"--- {ROUTE}: Plotting Cumulative Trajectory ---")
        run_command(f'python scripts/plot_trajectory.py --real_links "{REAL_LINKS}" --real_dist "{REAL_DIST}" --sim "{SIM_XML}" --out "plots/trajectory_{ROUTE}_baseline.png" --route {ROUTE}')

        # Append to Report
        metrics = pd.read_csv(f"docs/baseline_metrics_{ROUTE}.csv")
        rmse = metrics.loc[metrics['Metric'].str.contains('RMSE'), 'Value'].values[0] if not metrics.empty else 0
        mape = metrics.loc[metrics['Metric'].str.contains('MAPE'), 'Value'].values[0] if not metrics.empty else 0
        
        report_content += f"### Route {ROUTE}\n"
        report_content += f"- **RMSE**: {rmse:.2f} s\n"
        report_content += f"- **MAPE**: {mape:.2f} %\n\n"
        
        report_content += f"#### 可视化 (Visuals - {ROUTE})\n\n"
        report_content += f"**Cumulative Trajectory**\n![Trajectory](../plots/trajectory_{ROUTE}_baseline.png)\n\n"
        report_content += f"**Space-Time Diagram**\n![Space Time](../plots/spacetime_{ROUTE}_baseline.png)\n\n"
        report_content += f"**Headway Distribution**\n![Headway](../plots/headway_distribution_{ROUTE}_baseline.png)\n\n"
        report_content += f"**Link Speed Boxplot**\n![Boxplot](../plots/link_speed_boxplot_{ROUTE}_baseline.png)\n\n"
        report_content += "---\n\n"
        
    with open("docs/baseline_evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print("Report generated: docs/baseline_evaluation_report.md")

if __name__ == "__main__":
    main()
