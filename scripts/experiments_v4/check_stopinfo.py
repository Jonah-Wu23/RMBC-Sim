import xml.etree.ElementTree as ET

tree = ET.parse('data/experiments_v4/scale_sweep/off_peak/1h/scale0.00/A3/seed0/stopinfo.xml')
root = tree.getroot()
stops = root.findall('.//stopinfo')

print(f'Total stops: {len(stops)}')

route_68x = [s for s in stops if '68X' in s.get('id', '')]
print(f'68X stops: {len(route_68x)}')

if route_68x:
    print(f'\nSample 68X IDs:')
    for s in route_68x[:5]:
        print(f'  - {s.get("id")}')

all_ids = [s.get('id') for s in stops[:20]]
print(f'\nFirst 20 vehicle IDs:')
for vid in all_ids:
    print(f'  - {vid}')
