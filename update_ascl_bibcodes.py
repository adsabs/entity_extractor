#!/usr/bin/env python3
import json
import csv
from pathlib import Path

# File paths
ASCL_CODE_PATH = Path("ontologies/ASCL/ascl_code_with_citations.json")
ASCL_UPDATED_PATH = Path("ontologies/ASCL/ascl_software_list_with_citations_updated.json")
NEGATIVE_CSV_PATH = Path("ontologies/ASCL/negative_ascl_bibcodes.csv")

def load_json(path):
    """Load JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_negative_bibcodes(path):
    """Load negative bibcodes CSV and group by software name"""
    negative_bibcodes = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            software_name = row['software_name'].strip()
            bibcode = row['bibcode'].strip()
            if bibcode and bibcode != "No negative results found":
                if software_name not in negative_bibcodes:
                    negative_bibcodes[software_name] = []
                negative_bibcodes[software_name].append(bibcode)
    return negative_bibcodes

def extract_bibcode_from_url(url):
    """Extract bibcode from ADS URL"""
    if not url or not isinstance(url, str):
        return None
    if "ui.adsabs.harvard.edu/abs/" in url:
        return url.split("/abs/")[-1]
    return None

def get_positive_bibcodes(entry, positive_lookup):
    """Get positive bibcodes for an entry"""
    positive_bibcodes = set()
    
    # From the updated JSON file
    ascl_id = entry.get("ascl_id")
    if ascl_id in positive_lookup:
        positive_bibcodes.update(positive_lookup[ascl_id])
    
    # From used_in, described_in, cited_in fields
    for field in ["used_in", "described_in", "cited_in"]:
        if field in entry and entry[field]:
            if isinstance(entry[field], list):
                for item in entry[field]:
                    bibcode = extract_bibcode_from_url(item)
                    if bibcode:
                        positive_bibcodes.add(bibcode)
            elif isinstance(entry[field], str):
                bibcode = extract_bibcode_from_url(entry[field])
                if bibcode:
                    positive_bibcodes.add(bibcode)
    
    return list(positive_bibcodes)

def main():
    print("Loading files...")
    
    # Load main JSON file
    ascl_code_data = load_json(ASCL_CODE_PATH)
    
    # Load updated JSON file with positive bibcodes
    ascl_updated_data = load_json(ASCL_UPDATED_PATH)
    
    # Create lookup for positive bibcodes by ascl_id
    positive_lookup = {}
    for entry in ascl_updated_data:
        ascl_id = entry.get("ascl_id")
        if ascl_id and "positive_bibcodes" in entry:
            positive_lookup[ascl_id] = entry["positive_bibcodes"]
    
    # Load negative bibcodes CSV
    negative_bibcodes = load_negative_bibcodes(NEGATIVE_CSV_PATH)
    
    print(f"Processing {len(ascl_code_data)} entries...")
    
    # Update each entry
    for key, entry in ascl_code_data.items():
        # Get positive bibcodes
        positive_bibcodes = get_positive_bibcodes(entry, positive_lookup)
        
        # Get negative bibcodes (by software title)
        software_title = entry.get("title", "").split(":")[0].strip()
        negative_bibcodes_list = negative_bibcodes.get(software_title, [])
        
        # Remove any negative bibcodes that are in positive bibcodes
        filtered_negative = [b for b in negative_bibcodes_list if b not in positive_bibcodes]
        
        # Add fields to entry
        entry["positive_bibcodes"] = positive_bibcodes
        entry["negative_bibcodes"] = filtered_negative
    
    # Save updated file
    output_path = ASCL_CODE_PATH
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ascl_code_data, f, indent=2, ensure_ascii=False)
    
    print(f"Updated {output_path}")
    print("Sample entries processed:")
    for i, (key, entry) in enumerate(ascl_code_data.items()):
        if i >= 3:
            break
        print(f"  {entry['title']}: {len(entry['positive_bibcodes'])} positive, {len(entry['negative_bibcodes'])} negative")

if __name__ == "__main__":
    main()
