#!/usr/bin/env python3
"""
Transform ASCL JSON file to extract bibcodes from URLs in used_in, described_in, and cited_in fields.
"""

import json

def extract_bibcode_from_url(url):
    """Extract bibcode from ADS URL by taking text after last '/'."""
    if isinstance(url, str) and 'ui.adsabs.harvard.edu/abs/' in url:
        return url.split('/')[-1]
    return url

def transform_entry(entry):
    """Transform a single ASCL entry to extract bibcodes from URLs."""
    transformed = entry.copy()
    
    # Fields to transform
    fields_to_transform = ['used_in', 'described_in', 'cited_in']
    
    for field in fields_to_transform:
        if field in transformed and isinstance(transformed[field], list):
            transformed[field] = [extract_bibcode_from_url(url) for url in transformed[field]]
    
    return transformed

def main():
    input_file = 'ontologies/ASCL/ascl_v2.json'
    output_file = 'ontologies/ASCL/ascl_bibcodelabels.json'
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("Transforming URLs to bibcodes...")
    if isinstance(data, list):
        transformed_data = [transform_entry(entry) for entry in data]
    elif isinstance(data, dict):
        # If it's a dict with entries as values, transform each entry
        transformed_data = {}
        for key, entry in data.items():
            transformed_data[key] = transform_entry(entry)
    else:
        raise ValueError("Unexpected data structure in JSON file")
    
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, indent=2, ensure_ascii=False)
    
    print("Transformation complete!")

if __name__ == '__main__':
    main()
