#!/usr/bin/env python3
# optimized_extractor/search_term.py
"""
Interactive search tool for software mentions extraction results.
Allows searching for specific terms and viewing all associated metadata.
"""

import pandas as pd
import sys
from pathlib import Path

def load_data():
    """Load the extracted software mentions data."""
    # Check for individual contexts version first, then patched, then original
    individual_path = Path("results/exports/software_mentions_all_individual_contexts.csv")
    patched_path = Path("results/exports/software_mentions_all_patched.csv")
    csv_path = Path("results/exports/software_mentions_all.csv")
    
    if individual_path.exists():
        csv_path = individual_path
        print("Using new dataset with individual match contexts")
    elif patched_path.exists():
        csv_path = patched_path
        print("Using patched dataset with fixed title/abstract extractions")
    
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found at {csv_path}")
        print("Please make sure you've run the extraction pipeline first.")
        sys.exit(1)
    
    print("Loading software mentions data...")
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df):,} software mentions")
        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        sys.exit(1)

def search_term(df, search_query):
    """Search for a specific term in the data."""
    # Search by term name (case-insensitive)
    results = df[df['term_name'].str.contains(search_query, case=False, na=False)]
    
    if len(results) == 0:
        # Also search by extracted software name (before colon)
        df_copy = df.copy()
        df_copy['software_name'] = df_copy['term_name'].apply(
            lambda x: x.split(':')[0].strip() if ':' in str(x) else str(x)
        )
        results = df_copy[df_copy['software_name'].str.contains(search_query, case=False, na=False)]
    
    return results

def display_results(results, search_query):
    """Display search results in a formatted way."""
    if len(results) == 0:
        print(f"❌ No results found for '{search_query}'")
        return
    
    print(f"\n🔍 Found {len(results)} mentions for '{search_query}'")
    print("=" * 80)
    
    # Group by term_name to show different software packages
    grouped = results.groupby('term_name')
    
    for term_name, group in grouped:
        extracted_name = term_name.split(':')[0].strip() if ':' in term_name else term_name
        
        print(f"\n📦 SOFTWARE: {term_name}")
        print(f"🔎 Searching for: '{extracted_name}'")
        print(f"📊 Total mentions: {len(group)}")
        
        # Show distribution by location
        location_counts = group['match_location'].value_counts()
        print(f"📍 Mention locations: {dict(location_counts)}")
        
        # Show unique bibcodes
        unique_bibcodes = group['bibcode'].nunique()
        total_matches = group['match_count'].sum()
        print(f"📚 Unique publications: {unique_bibcodes}")
        print(f"🎯 Total word matches: {total_matches}")
        
        print(f"\n📋 DETAILED MENTIONS:")
        print("-" * 60)
        
        # Show each mention
        for idx, (_, row) in enumerate(group.iterrows(), 1):
            print(f"\n{idx}. 📄 {row['bibcode']} ({row['match_location']})")
            
            # Show title if available
            if pd.notna(row['title']) and row['title'].strip():
                title = str(row['title']).strip()[:100]
                print(f"   📝 Title: {title}{'...' if len(str(row['title'])) > 100 else ''}")
            
            # Show match details
            print(f"   🎯 Matches in text: {row['match_count']}")
            
            if row['in_title']:
                print(f"   ✅ Also found in title")
            if row['in_abstract']:
                print(f"   ✅ Also found in abstract")
            
            # Show context
            context = str(row['context']).strip()
            if context and context != 'nan':
                print(f"   📖 Context:")
                # Highlight the search term in context (case-insensitive)
                highlighted_context = context
                if extracted_name.lower() in context.lower():
                    start = context.lower().find(extracted_name.lower())
                    end = start + len(extracted_name)
                    highlighted_context = (
                        context[:start] + 
                        f"**{context[start:end]}**" + 
                        context[end:]
                    )
                
                # Wrap long context
                max_width = 70
                words = highlighted_context.split()
                lines = []
                current_line = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + 1 > max_width and current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word)
                    else:
                        current_line.append(word)
                        current_length += len(word) + 1
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                for line in lines:
                    print(f"      {line}")
            
            print()

def interactive_search(df):
    """Interactive search loop."""
    print("\n🔍 INTERACTIVE SOFTWARE MENTION SEARCH")
    print("=" * 50)
    print("Enter software names to search for mentions.")
    print("Examples: 'LENSKY', 'CMBFAST', 'PyVO', 'astropy'")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            query = input("🔎 Enter search term: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                print("Please enter a search term.")
                continue
            
            results = search_term(df, query)
            display_results(results, query)
            
            print("\n" + "─" * 80)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function."""
    print("🚀 SOFTWARE MENTIONS SEARCH TOOL")
    print("=" * 50)
    
    # Load data
    df = load_data()
    
    # Check if command line argument provided
    if len(sys.argv) > 1:
        search_query = ' '.join(sys.argv[1:])
        print(f"\nSearching for: '{search_query}'")
        results = search_term(df, search_query)
        display_results(results, search_query)
    else:
        # Interactive mode
        interactive_search(df)

if __name__ == "__main__":
    main()
