import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.path.append(r"C:\Program Files (x86)\Eclipse\Sumo\tools")
import sumolib
net = sumolib.net.readNet(r"d:\Documents\Bus Project\Sorce code\sumo\net\hk_irn_v3.net.xml")
lane = net.getLane("8663_rev_0")
print(f"Mei Foo position: {lane.getShape()[0]}")
