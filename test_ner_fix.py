#!/usr/bin/env python3

import torch
from transformers import pipeline

RECOGNIZED_SOFTWARE_TAGS = {"SOFTWARE", "TOOL", "Application_Mention"}

def run_ner_on_span(text, label, ner_model):
    """
    Check if a specific term span in text is classified as software by NER model.
    Uses span-aware classification to preserve original matched terms.
    """
    try:
        # Find the term in the text to get its character offsets
        label_lower = label.lower()
        text_lower = text.lower()
        
        # Find all occurrences of the term
        start_idx = 0
        while True:
            span_start = text_lower.find(label_lower, start_idx)
            if span_start == -1:
                break
            span_end = span_start + len(label)
            
            # Check if this is a word boundary match
            is_word_start = span_start == 0 or not text[span_start - 1].isalnum()
            is_word_end = span_end >= len(text) or not text[span_end].isalnum()
            
            if is_word_start and is_word_end:
                # Run NER on the text and check tokens within our span
                ner_results = ner_model(text[:512])  # Limit text length for model
                
                for token in ner_results:
                    # Check if token overlaps with our target span
                    token_start = token.get("start", 0)
                    token_end = token.get("end", 0)
                    
                    # Token is considered inside span if it overlaps significantly
                    if (token_start >= span_start and token_end <= span_end) or \
                       (token_start < span_end and token_end > span_start):
                        if token.get("entity", "").replace("B-", "").replace("I-", "") in RECOGNIZED_SOFTWARE_TAGS:
                            return True
            
            start_idx = span_start + 1
        
        return False
    except Exception as e:
        print(f"⚠️ NER error: {e}")
        return False

def main():
    print("🧪 Testing span-aware NER fix...")
    
    # Initialize NER model with span-aware configuration
    ner_model = pipeline(
        "token-classification",
        model="oeg/software_benchmark_multidomain",
        tokenizer="oeg/software_benchmark_multidomain",
        aggregation_strategy="none",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Test cases
    test_cases = [
        ("We used Zeus to analyze the data.", "Zeus"),
        ("The team utilized Python for data processing.", "Python"),
        ("MATLAB was employed in our calculations.", "MATLAB"),
        ("We ran the simulation using IDL.", "IDL"),
        ("The hazel tree is beautiful in spring.", "hazel"),  # Should NOT be detected as software
    ]
    
    print("\n📊 Test Results:")
    for text, term in test_cases:
        result = run_ner_on_span(text, term, ner_model)
        status = "✅ SOFTWARE" if result else "❌ NOT SOFTWARE"
        print(f"   {status}: '{term}' in '{text}'")
        
        # Debug: show the actual tokens
        span_start = text.lower().find(term.lower())
        if span_start != -1:
            span_end = span_start + len(term)
            ner_results = ner_model(text[:512])
            relevant_tokens = [
                token for token in ner_results 
                if (token.get("start", 0) >= span_start and token.get("end", 0) <= span_end) or 
                   (token.get("start", 0) < span_end and token.get("end", 0) > span_start)
            ]
            print(f"      Tokens in span: {relevant_tokens}")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    main()
