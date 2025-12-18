import json
import xml.etree.ElementTree as ET
import pandas as pd
import os

def peek_json(path):
    print(f"\n--- JSON: {path} ---")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Print keys and a snippet of the data
        print("Keys:", data.keys())
        if 'rainfall' in data:
            print("Rainfall data snippet:", data['rainfall']['data'][:2])
        if 'temperature' in data:
            print("Temperature data snippet:", data['temperature']['data'][:2])
        if 'warningMessage' in data:
            print("Warning Message:", data['warningMessage'])

def peek_xml(path, tag_limit=5):
    print(f"\n--- XML: {path} ---")
    tree = ET.parse(path)
    root = tree.getroot()
    print("Root Tag:", root.tag)
    count = 0
    for child in root:
        print(f"Child Tag: {child.tag}, Attrib: {child.attrib}")
        for grand in child:
            print(f"  Grandchild Tag: {grand.tag}, Text: {grand.text[:50] if grand.text else 'None'}")
        count += 1
        if count >= tag_limit:
            break

def peek_csv(path):
    print(f"\n--- CSV: {path} ---")
    try:
        df = pd.read_csv(path)
        print(df.head())
    except Exception as e:
        print(f"Error reading CSV: {e}")

if __name__ == "__main__":
    peek_json('data/raw/hko/hko-rhrread-20251217-173422.json')
    peek_xml('data/raw/jti/Journeytimev2-20251217-173422.xml')
    peek_xml('data/raw/stn/trafficnews-20251217-173422.xml')
    peek_csv('data/raw/tsm_notification/tsm-notification-20251217-173422.csv')
