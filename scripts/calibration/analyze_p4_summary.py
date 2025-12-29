#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分析 P4-1 summary 中的速度趋势"""

import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 分析 summary.xml 中的速度趋势
tree = ET.parse(PROJECT_ROOT / 'sumo/output/p4_isolation/p4_1_insertion/summary.xml')
root = tree.getroot()

steps = root.findall('.//step')
print('time, inserted, waiting, running, halting, meanSpeed')
for step in steps[::60]:  # 每 60 秒采样
    t = float(step.get('time'))
    inserted = step.get('inserted')
    waiting = step.get('waiting')
    running = step.get('running')
    halting = step.get('halting')
    meanSpeed = step.get('meanSpeed')
    print(f'{t:5.0f}, {inserted:>3}, {waiting:>3}, {running:>3}, {halting:>3}, {meanSpeed}')

# 最后一步
last = steps[-1]
print(f"\n最终状态 (t={last.get('time')}s):")
print(f"  inserted={last.get('inserted')}, waiting={last.get('waiting')}, running={last.get('running')}")
print(f"  halting={last.get('halting')}, meanSpeed={last.get('meanSpeed')}")

# 分析：找出什么时候开始拥堵
print("\n拥堵分析:")
for i, step in enumerate(steps):
    t = float(step.get('time'))
    halting = int(step.get('halting'))
    running = int(step.get('running'))
    if running > 0:
        halt_ratio = halting / running
        if halt_ratio > 0.5 and t < 120:  # 前2分钟就有超过50%车辆停止
            print(f"  t={t:.0f}s: {halting}/{running} = {halt_ratio:.0%} halting")
