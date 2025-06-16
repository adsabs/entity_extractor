# software_mention_pipeline/test_indus_ner_tags.py

from transformers import pipeline
from pprint import pprint
import torch

# Initialize the INDUS model
model_id = "adsabs/nasa-smd-ibm-v0.1_NER_DEAL"

ner = pipeline(
    "ner",
    model=model_id,
    tokenizer=model_id,
    aggregation_strategy="simple",  # get grouped entities
    device=0 if torch.cuda.is_available() else -1
)

# Example sentences with likely tags
test_sentences = [
    "We processed the data using SPICE and plotted it with Matplotlib.",
    "The Gaia DR3 dataset was downloaded from MAST and analyzed in Python.",
    "We simulated the surface properties using the GIPL permafrost model.",
    "Observations were made with the JWST NIRCam instrument.",
    "The data were stored in the NASA ADS archive and analyzed with NumPy."
]

print(f"\n🔬 Running INDUS NER model ({model_id}) on {len(test_sentences)} test sentences...\n")

for i, sent in enumerate(test_sentences):
    print(f"📝 [{i}] {sent}")
    results = ner(sent)
    pprint(results)
    print("-" * 60)
