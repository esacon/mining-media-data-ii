#!/usr/bin/env python3

import json
from pathlib import Path

def debug_json_parsing():
    """Debug the JSON parsing issue with dataset 3 files."""
    
    # Read a larger chunk from the bulkplayers.json file
    file_path = Path("src/data/dataset_game3/bulkplayers.json")
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read(50000)  # Read first 50KB
    
    print("Analyzing bulkplayers.json structure...")
    print("=" * 50)
    
    # Find the first array and extract the first complete element
    try:
        # Find the start of the first array
        first_colon = content.find(':[')
        if first_colon != -1:
            array_start = first_colon + 2  # Skip ':[' 
            
            # Find the first complete object in the array
            brace_count = 0
            in_string = False
            escape_next = False
            first_object_end = None
            
            for i in range(array_start, len(content)):
                char = content[i]
                
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == "\\":
                    escape_next = True
                    continue
                    
                if char == '"':
                    in_string = not in_string
                    
                if not in_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            first_object_end = i + 1
                            break
            
            if first_object_end:
                first_object_json = content[array_start:first_object_end]
                print(f"First player object JSON:")
                print(first_object_json)
                print()
                
                try:
                    player_obj = json.loads(first_object_json)
                    print("Parsed player object structure:")
                    for key, value in player_obj.items():
                        value_type = type(value).__name__
                        if isinstance(value, str) and len(value) > 50:
                            sample_value = value[:50] + "..."
                        else:
                            sample_value = str(value)
                        print(f"  • {key}: {value_type} = {sample_value}")
                    
                    print(f"\nTotal fields in player object: {len(player_obj)}")
                    
                except json.JSONDecodeError as e:
                    print(f"Failed to parse first object: {e}")
            else:
                print("Could not find complete first object")
        else:
            print("Could not find first array")
            
    except Exception as e:
        print(f"Error analyzing structure: {e}")
    
    print("\n" + "=" * 50)
    print("Analyzing bulkmatches.json structure...")
    
    # Also analyze bulkmatches.json for comparison
    matches_file = Path("src/data/dataset_3_game3/bulkmatches.json")
    with open(matches_file, "r", encoding="utf-8") as f:
        matches_content = f.read(10000)  # Read first 10KB
    
    try:
        # Find the first object value
        first_colon = matches_content.find(':{')
        if first_colon != -1:
            object_start = first_colon + 1  # Skip ':'
            
            # Find the complete object
            brace_count = 0
            in_string = False
            escape_next = False
            first_object_end = None
            
            for i in range(object_start, len(matches_content)):
                char = matches_content[i]
                
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == "\\":
                    escape_next = True
                    continue
                    
                if char == '"':
                    in_string = not in_string
                    
                if not in_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            first_object_end = i + 1
                            break
            
            if first_object_end:
                first_match_json = matches_content[object_start:first_object_end]
                print(f"First match object JSON:")
                print(first_match_json[:500] + "..." if len(first_match_json) > 500 else first_match_json)
                print()
                
                try:
                    match_obj = json.loads(first_match_json)
                    print("Parsed match object structure:")
                    for key, value in match_obj.items():
                        value_type = type(value).__name__
                        if isinstance(value, (list, dict)):
                            if isinstance(value, list):
                                sample_value = f"[{len(value)} items]"
                            else:
                                sample_value = f"{{{len(value)} keys}}"
                        elif isinstance(value, str) and len(value) > 30:
                            sample_value = value[:30] + "..."
                        else:
                            sample_value = str(value)
                        print(f"  • {key}: {value_type} = {sample_value}")
                    
                    print(f"\nTotal fields in match object: {len(match_obj)}")
                    
                except json.JSONDecodeError as e:
                    print(f"Failed to parse first match object: {e}")
            else:
                print("Could not find complete first match object")
        else:
            print("Could not find first match object")
            
    except Exception as e:
        print(f"Error analyzing match structure: {e}")

if __name__ == "__main__":
    debug_json_parsing() 