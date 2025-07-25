# extract_positive_bibcodes.py

import json
import csv

# Load JSON
with open('ontologies/ASCL/ascl_v2.json') as f:
    ontology_data = json.load(f)

# Write to CSV
with open('positive_bibcodes.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Term', 'Bibcode'])

    for entry in ontology_data.values():
        term = entry["title"].split(":")[0].strip()
        for bib in entry.get("positive_bibcodes", []):
            writer.writerow([term, bib])

print("✅ Positive bibcodes written to positive_bibcodes.csv")
