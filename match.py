import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Load dataset
df = pd.read_csv('your_dataset.csv')

# Step 1: Join detected texts into a single string
ocr_texts = [
    'john', 'e', 'fizgerald', 'so', 'o', 'larceny', 'kentucky',
    'straight', 'bourbon', 'whiskery', 'e', '', 'barrel', 'batca',
    'dsaa', 'prodn', 'aas', 'proof', 'alcvola', 'giia'
]
query = ' '.join([word for word in ocr_texts if word.strip()])

# Step 2: Use fuzzy matching to find the best matching 'name'
choices = df['name'].tolist()
top_matches = process.extract(query, choices, scorer=fuzz.token_set_ratio, limit=5)

# Step 3: Display best matches with their corresponding rows
print("\nTop matches:")
for match_text, score in top_matches:
    match_row = df[df['name'] == match_text]
    print(f"\nMatch Score: {score}")
    print(match_row)
