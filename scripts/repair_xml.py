import sys

def repair_xml(file_path):
    print(f"Repairing {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    
    # 1. Keep Header (approx lines 0-35)
    # 2. Detect and remove intermediate </stops>
    # 3. Filter corrupted lines (lines not starting with < and not empty)
    
    header_found = False
    stops_open = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Pass through header/comments/config
        if not stops_open:
            new_lines.append(line)
            if stripped.startswith('<stops'):
                stops_open = True
            continue
            
        # Inside <stops>...
        
        # If we see </stops> and it's NOT the last line (allowing for some trailing newlines), skip it
        # Actually simplest heuristic: Only keep the LAST </stops>
        if stripped == '</stops>':
             # We will append closing tag manually at the end
             continue
             
        # Check for malformed tags
        if not stripped.startswith('<') and not stripped == '':
            print(f"Dropping corrupted line {i+1}: {stripped[:50]}...")
            continue
            
        new_lines.append(line)
        
    # Ensure final closure
    if new_lines[-1].strip() != '</stops>':
        new_lines.append('</stops>\n')
        
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Repair complete.")

if __name__ == "__main__":
    repair_xml("sumo/output/stopinfo_exp2.xml")
