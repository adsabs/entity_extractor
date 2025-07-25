The data file read in to populate the dashboard, software_mentions_all_with_labels.csv.gzip, has a bibcode_label column that says whether the bibcode associated with the context is positive or negative. The oracle came up with a plan to update the dashboard to separate the context dropdown options into positive contexts, uncurated contexts, and negative contexts. There should also be a way to modify the label of a given context to support the curation of bibcodes based on the evaluated context.

Implementation Plan
Core Concept: Use a "delta file" approach to track label changes without modifying the large 612k-row CSV file. This provides:

Fast operations (append-only)
Audit trail of changes
Immediate UI updates
Key Changes:

Enhanced Data Loading: Add row_id and merge any existing curation changes from a delta file
Tabbed Context Interface: Replace single dropdown with three tabs (Positive ✅, Uncurated 📝, Negative 🚫), any bibcodes with a label of unknown should be in the Negative category
Per-Context Curation: Add label modification widgets for each context
Session State Tracking: Queue changes in memory before committing
Sidebar Commit Tool: Batch save changes with curator attribution
The Oracle's plan maintains the existing search functionality while adding sophisticated curation capabilities that scale well with the large dataset.

