#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fix_background_from_to.py
=========================
修复背景流 from=to 问题：将原地流转换为真正的 OD 流

问题：背景流文件中 from=to=同一边，车辆原地生成原地消失，
     不会行驶也不会与走廊产生交互。

解决：为每条 flow 随机选择一个不同的 to 边，确保车辆真正行驶。
"""

import xml.etree.ElementTree as ET
import random
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description='修复背景流 from=to 问题')
    ap.add_argument("--in", dest="inp", required=True, help='输入 rou.xml')
    ap.add_argument("--out", dest="out", required=True, help='输出 flows.xml（供 duarouter 使用）')
    ap.add_argument("--begin", type=float, default=0.0, help='begin 时间')
    ap.add_argument("--end", type=float, default=3600.0, help='end 时间')
    ap.add_argument("--seed", type=int, default=42, help='随机种子')
    args = ap.parse_args()

    random.seed(args.seed)

    tree = ET.parse(args.inp)
    root = tree.getroot()

    flows = list(root.iter("flow"))
    if not flows:
        raise SystemExit("No <flow> found.")

    # 目的地候选：用所有出现过的 from 边作为 pool
    pool = [f.get("from") for f in flows if f.get("from")]
    pool = list(set([x for x in pool if x is not None]))  # 去重
    
    if len(pool) < 2:
        raise SystemExit("Destination pool too small.")
    
    print(f"[INFO] 发现 {len(flows)} 条 flow")
    print(f"[INFO] 目的地候选池: {len(pool)} 条边")

    changed = 0
    same_edge_count = 0
    
    for f in flows:
        fr = f.get("from")
        to = f.get("to")
        if not fr:
            continue

        # 统计 from=to 的数量
        if to == fr:
            same_edge_count += 1

        # 补齐时间窗口
        if f.get("begin") is None:
            f.set("begin", str(args.begin))
        if f.get("end") is None:
            f.set("end", str(args.end))

        # 修复 from=to
        if (to is None) or (to == fr):
            # 选一个不同的 to
            cand = fr
            for _ in range(50):
                cand = random.choice(pool)
                if cand != fr:
                    break
            f.set("to", cand)
            changed += 1

        # 加出发分散
        if f.get("departPos") is None:
            f.set("departPos", "random")
        if f.get("departLane") is None:
            f.set("departLane", "best")

    # 保存
    tree.write(args.out, encoding="utf-8", xml_declaration=True)
    
    print(f"\n[结果]")
    print(f"  原 from=to 同边: {same_edge_count}/{len(flows)} ({100*same_edge_count/len(flows):.1f}%)")
    print(f"  已修复: {changed} 条")
    print(f"  输出: {args.out}")
    print(f"\n[下一步] 运行 duarouter 生成路线:")
    print(f"  duarouter -n hk_cropped.net.xml -f {args.out} -o background_fixed.rou.xml --ignore-errors --repair")


if __name__ == "__main__":
    main()
