import os
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# === Configuration ===
CSV_FOLDER = "term_csv_files"
OUTPUT_FOLDER = "annotated_csvs_indus"
MODEL_NAME = "adsabs/nasa-smd-ibm-v0.1_NER_DEAL"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Function to run NER on a text field
def extract_entities(text):
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return []
    try:
        return [(e["word"], e["entity_group"], round(e["score"], 3)) for e in ner_pipeline(text)]
    except Exception as e:
        print(f"Error processing text: {e}")
        return []

# Process each CSV in the folder
for filename in os.listdir(CSV_FOLDER):
    if filename.endswith(".csv"):
        input_path = os.path.join(CSV_FOLDER, filename)
        print(f"🔍 Processing {input_path}")

        try:
            df = pd.read_csv(input_path)
            if "term_context" not in df.columns:
                print(f"⚠️ Skipping {filename}: 'term_context' column not found")
                continue

            df["ner_entities"] = df["term_context"].apply(extract_entities)
            output_path = os.path.join(OUTPUT_FOLDER, filename)
            df.to_csv(output_path, index=False)
            print(f"✅ Saved: {output_path}")
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

print("🎉 All files processed.")
