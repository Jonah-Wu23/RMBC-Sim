#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apply_v3_patches.py
===================
Apply V3 bridge edge patches to the network

1. Export plain files from V1 network
2. Insert V3 patches
3. Rebuild network as V3 version
"""

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = PROJECT_ROOT / "tmp"
NET_DIR = PROJECT_ROOT / "sumo" / "net"

V1_NET = NET_DIR / "hk_irn_v3_patched_v1.net.xml"
V3_NET = NET_DIR / "hk_irn_v3_patched_v3.net.xml"


def insert_before_closing_tag(file_path: Path, closing_tag: str, content: str):
    """Insert content before closing tag"""
    text = file_path.read_text(encoding="utf-8")
    idx = text.rfind(closing_tag)
    if idx == -1:
        raise ValueError(f"Cannot find {closing_tag} in {file_path}")
    new_text = text[:idx] + content + "\n" + text[idx:]
    file_path.write_text(new_text, encoding="utf-8")
    print(f"  [OK] Inserted into {file_path.name}")


def main():
    print("="*80)
    print("[V3] Apply Bridge Edge Patches")
    print("="*80)
    
    # 1. Export plain
    print("\n[Step 1] Export plain files from V1 network")
    cmd = [
        "netconvert",
        "-s", str(V1_NET),
        "--plain-output-prefix", str(TMP_DIR / "hk_v3_plain")
    ]
    print(f"  Running: netconvert ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] {result.stderr[:500]}")
        return 1
    print("  [OK] Plain files exported")
    
    # 2. Read patches
    print("\n[Step 2] Read V3 patches")
    edge_patch = (TMP_DIR / "v3_bridge_edges.xml").read_text(encoding="utf-8")
    conn_patch = (TMP_DIR / "v3_bridge_connections.xml").read_text(encoding="utf-8")
    
    # Extract only the edge/connection lines
    edge_content = "\n".join([
        line for line in edge_patch.split("\n")
        if line.strip() and ("<edge" in line or "<!--" in line)
    ])
    conn_content = "\n".join([
        line for line in conn_patch.split("\n")
        if line.strip() and "<connection" in line
    ])
    
    print(f"  Edge patch: {len(edge_content)} chars")
    print(f"  Conn patch: {len(conn_content)} chars")
    
    # 3. Insert patches
    print("\n[Step 3] Insert patches into plain files")
    insert_before_closing_tag(TMP_DIR / "hk_v3_plain.edg.xml", "</edges>", edge_content)
    insert_before_closing_tag(TMP_DIR / "hk_v3_plain.con.xml", "</connections>", conn_content)
    
    # 4. Rebuild network
    print("\n[Step 4] Rebuild network")
    cmd = [
        "netconvert",
        "--node-files", str(TMP_DIR / "hk_v3_plain.nod.xml"),
        "--edge-files", str(TMP_DIR / "hk_v3_plain.edg.xml"),
        "--connection-files", str(TMP_DIR / "hk_v3_plain.con.xml"),
        "--tllogic-files", str(TMP_DIR / "hk_v3_plain.tll.xml"),
        "-o", str(V3_NET),
    ]
    
    print(f"  Output: {V3_NET.name}")
    log_file = PROJECT_ROOT / "logs" / "netconvert_v3_patch.log"
    
    with open(log_file, "w", encoding="utf-8") as f:
        result = subprocess.run(cmd, capture_output=True, text=True)
        f.write(result.stdout)
        f.write(result.stderr)
    
    if result.returncode == 0 and V3_NET.exists():
        size_mb = V3_NET.stat().st_size / (1024 * 1024)
        print(f"  [OK] Network rebuilt: {size_mb:.1f} MB")
        print(f"  Log: {log_file}")
    else:
        print(f"  [ERROR] netconvert failed")
        print(result.stderr[-1000:] if result.stderr else "No stderr")
        return 1
    
    print("\n" + "="*80)
    print("[DONE] V3 network ready: " + str(V3_NET))
    print("="*80)
    return 0


if __name__ == "__main__":
    exit(main())
