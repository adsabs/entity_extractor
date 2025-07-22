import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load CSV
df = pd.read_csv("zeus_negative_bibcodes.csv")  # Change this path as needed

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("oeg/software_benchmark_multidomain")
model = AutoModelForTokenClassification.from_pretrained("oeg/software_benchmark_multidomain")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Chunking helper: splits text into smaller segments
def chunk_text(text, max_tokens=512, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_tokens
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks

# Apply NER to each row with chunking
def extract_entities(row):
    text = row["term_context"]
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return []
    
    entities = []
    chunks = chunk_text(text)
    for chunk in chunks:
        try:
            chunk_entities = ner_pipeline(chunk)
            entities.extend([(e["word"], e["entity_group"], round(e["score"], 4)) for e in chunk_entities])
        except Exception as e:
            print(f"NER error on chunk: {e}")
            continue
    return entities

# Run NER
df["entities"] = df.apply(extract_entities, axis=1)

# Save output
df.to_csv("output_zeus_neg_ner_oeg.csv", index=False)
print("✅ Saved to output_with_ner.csv")

