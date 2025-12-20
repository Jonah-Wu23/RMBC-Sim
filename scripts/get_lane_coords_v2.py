import os
import sys

# Add SUMO tools to path
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.path.append(r"C:\Program Files (x86)\Eclipse\Sumo\tools")

import sumolib

net_file = r"d:\Documents\Bus Project\Sorce code\sumo\net\hk_irn_v3.net.xml"
net = sumolib.net.readNet(net_file)

target_lanes = ["105735_rev_0", "4287_rev_0"]

for lane_id in target_lanes:
    try:
        lane = net.getLane(lane_id)
        shape = lane.getShape()
        print(f"Lane {lane_id} position: {shape[0]}")
    except:
        print(f"Lane {lane_id} not found")
