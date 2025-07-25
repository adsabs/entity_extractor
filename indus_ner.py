# run_ner_indus.py

import csv
from transformers import pipeline
from tqdm import tqdm

# Set up INDUS NER model
ner = pipeline("ner", model="adsabs/nasa-smd-ibm-v0.1_NER_DEAL", aggregation_strategy="simple")

# Simulated function – replace this with real fulltext fetching logic
def get_bibcode_context(bibcode):
    return f"Simulated text mentioning {bibcode} with software like GALFIT or ZEUS."

# Open input/output
with open('positive_bibcodes.csv', newline='') as f_in, open('ner_results_indus.csv', 'w', newline='') as f_out:
    reader = csv.DictReader(f_in)
    writer = csv.DictWriter(f_out, fieldnames=[
        'Term', 'Bibcode', 'Model', 'Success', 'Term Type', 'Context', 'Notes'
    ])
    writer.writeheader()

    for row in tqdm(reader):
        term = row['Term']
        bibcode = row['Bibcode']
        context = get_bibcode_context(bibcode)
        model = "INDUS"
        found = False
        success = False
        entity_type = "none"
        notes = ""

        try:
            results = ner(context)
            for r in results:
                if r['word'].lower() == term.lower():
                    found = True
                    entity_type = r['entity_group']
                    success = (entity_type.lower() == 'software')
                    break
            if not found:
                notes = "Term not found in context"
        except Exception as e:
            notes = f"NER error: {e}"

        writer.writerow({
            'Term': term,
            'Bibcode': bibcode,
            'Model': model,
            'Success': success,
            'Term Type': entity_type,
            'Context': context,
            'Notes': notes
        })

print("✅ NER results written to ner_results_indus.csv")
