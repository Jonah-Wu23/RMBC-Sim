
import xml.etree.ElementTree as ET
import pandas as pd
import sys
import glob
import os

def inspect_raw_xml(xml_file):
    print(f"Inspecting {xml_file}...")
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Namespace handling? Usually data.gov.hk xmls have namespaces
        # Let's check root tag
        print(f"Root tag: {root.tag}")
        
        # Traverse
        # The structure is likely: root -> periods -> period -> detectors -> detector
        
        detectors_found = 0
        
        # Find all detectors. The tags might be qualified.
        # Let's try flexible search
        for detector in root.iter():
            if 'detector_id' in detector.tag:
                 # This is likely inside a detector block
                 # Need to go up one level or iterating 'detector' tags directly
                 pass

        # Let's iterating 'detector' nodes
        # Assuming tag ends with 'detector'
        for detector in root.iter():
            if detector.tag.endswith('detector') and 'detectors' not in detector.tag:
                det_id_node = None
                for child in detector:
                    if child.tag.endswith('detector_id'):
                        det_id_node = child
                        break
                
                if det_id_node is not None:
                    did = det_id_node.text
                    print(f"\nDetector: {did}")
                    
                    # Look for lanes
                    lanes_node = None
                    for child in detector:
                        if child.tag.endswith('lanes'):
                            lanes_node = child
                            break
                    
                    if lanes_node:
                        for lane in lanes_node:
                            # lane details
                            lid = speed = vol = occ = None
                            for prop in lane:
                                if prop.tag.endswith('lane_id'): lid = prop.text
                                if prop.tag.endswith('speed'): speed = prop.text
                                if prop.tag.endswith('volume'): vol = prop.text
                                if prop.tag.endswith('occupancy'): occ = prop.text
                            
                            print(f"  Lane {lid}: Vol={vol}, Speed={speed}, Occ={occ}")
                            
                    detectors_found += 1
                    if detectors_found >= 5:
                        break
        
        if detectors_found == 0:
            print("No detectors found via iteration. Printing root children tags:")
            for child in root:
                print(child.tag)

    except Exception as e:
        print(f"Error parsing XML: {e}")

if __name__ == "__main__":
    # Find the latest xml
    files = sorted(glob.glob("data/raw/detector_locations/rawSpeedVol-all-*.xml"))
    if not files:
        print("No XML files found.")
    else:
        inspect_raw_xml(files[-1])
