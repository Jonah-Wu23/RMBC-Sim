import os
import sys

# Ensure SUMO tools are on path (redundant if PYTHONPATH is set, but safe)
sumo_home = os.environ.get("SUMO_HOME")
if not sumo_home:
    raise RuntimeError("SUMO_HOME is not set. Please set it to your SUMO installation directory.")
tools_path = os.path.join(sumo_home, "tools")
if tools_path not in sys.path:
    sys.path.append(tools_path)

from sumolib import checkBinary  # type: ignore
import traci  # type: ignore


def main() -> None:
    # Use a built-in example network as a smoke test (no routes needed)
    net_path = os.path.join(
        sumo_home,
        "doc",
        "examples",
        "duarouter",
        "flows2routes",
        "input_net.net.xml",
    )
    if not os.path.exists(net_path):
        raise FileNotFoundError(f"Network not found: {net_path}")

    sumo_binary = checkBinary("sumo")
    traci.start([sumo_binary, "-n", net_path, "--no-step-log", "true"])
    try:
        for _ in range(5):
            traci.simulationStep()
        print("sumo+traci smoke test: OK")
    finally:
        traci.close(False)


if __name__ == "__main__":
    main()
