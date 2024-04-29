from fastapi import FastAPI, HTTPException
import pandas as pd
import string
import nltk
# import Levenshtein


# from rapidfuzz import levenshtein
# from Levenshtein import ratio
print(5)

import time

app = FastAPI()

# Load DataFrame
path = '../large_dataset.csv'
df = pd.read_csv(path)

# Rename column
df = df.rename(columns={'NXme': 'Name'})

# Preprocess combined text
df['Combined'] = df['Name'].str.strip() + df['Father Name'].str.strip() + df['Idno'].astype(str)

# Tokenize and preprocess text
nltk.download('punkt')


df= df.head(1000)

def levenshtein_distance(str1, str2):
    n_m = len(str1) + 1
    n_n = len(str2) + 1
    dp = [[0 for _ in range(n_n)] for _ in range(n_m)]

    for i in range(n_m):
        dp[i][0] = i

    for j in range(n_n):
        dp[0][j] = j

    for i in range(1, n_m):
        for j in range(1, n_n):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[n_m - 1][n_n - 1]


def similarity_find(str1, str2):
    # Calculate Levenshtein distance
    distance = levenshtein_distance(str1, str2)
    # Calculate similarity (higher score means more similar)
    max_length = max(len(str1), len(str2))
    return 1 - (distance / max_length)

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    return ' '.join([token.lower() for token in tokens if token.isalnum()])

# Trie implementation
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.data = {'Name': None, 'Father Name': None, 'Idno': None}

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, data):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.data = data

    def search_fuzzy(self, query, threshold=0.8):
        results = []
        self._search_fuzzy_recursive(query, self.root, '', threshold, results)
        return results

    def _search_fuzzy_recursive(self, query, node, prefix, threshold, results):
        if not node:
            return
        for child, child_node in node.children.items():
            new_prefix = prefix + child
            self._search_fuzzy_recursive(query, child_node, new_prefix, threshold, results)
        similarity = similarity_find(query, prefix)
        if similarity >= threshold and node.is_end_of_word:
            results.append((prefix, node.data))

trie = Trie()
for index, row in df.iterrows():
    combined_text = row['Combined']
    preprocessed_word = preprocess_text(combined_text)
    data = {'Name': row['Name'], 'Father Name': row['Father Name'], 'Idno': row['Idno']}
    trie.insert(preprocessed_word, data)

# Endpoint to receive query and return search results
@app.get("/search/")
async def search(query: str):
    start_time = time.time()
    query = preprocess_text(query)
    results = trie.search_fuzzy(query)
    if results:
        search_results = []
        for prefix, data in results:
            search_results.append({
                "prefix": prefix,
                "Name": data['Name'],
                "Father Name": data['Father Name'],
                "Idno": data['Idno']
            })
        end_time = time.time()
        execution_time = end_time - start_time
        return {"execution_time": execution_time, "results": search_results}
    else:
        raise HTTPException(status_code=404, detail=f"No results found for '{query}'.")
