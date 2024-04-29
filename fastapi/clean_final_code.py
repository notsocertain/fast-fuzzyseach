# -*- coding: utf-8 -*-
"""clean-final-code.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1IZs4TkWxu4px4rMUmblObGPZlxQaWw93
"""

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

# from google.colab import drive
# drive.mount('/content/drive')
from fastapi import FastAPI, HTTPException
import time
app = FastAPI()

import pandas as pd
import numpy as np

path ='../large_dataset.csv'

df=pd.read_csv(path)

import time

# !pip install python-levenshtein

import Levenshtein  # Import Levenshtein module
# Import the Levenshtein module for calculating edit distance
# from Levenshtein import ratio

from Levenshtein import *
from warnings import warn

# Rename the 'NXme' column to 'Name'
df = df.rename(columns={'NXme': 'Name'})

# Concatenate data with removing whitespaces
df['Combined'] = df['Name'].str.strip() + df['Father Name'].str.strip() + df['Idno'].astype(str)

# Viewing the first item of the 'Combined' column using iloc
first_combined_item = df['Combined'].iloc[0]
print("First item of 'Combined' column:2", first_combined_item)

# Alternatively, you can use direct indexing
first_combined_item_direct = df['Name'][0]
print("First item of 'Name' column (direct indexing):", first_combined_item_direct)

df.shape

import string
import nltk
nltk.download('punkt')

def preprocess_text(text):
  """
  Preprocesses text using NLTK for tokenization, removing special characters, and converting to lowercase.

  Args:
      text: The text to preprocess.

  Returns:
      The preprocessed text as a single string.
  # """
  # # Download NLTK punkt tokenizer (if not already installed)
  # nltk.download('punkt')

  # Define allowed characters (alphanumeric and underscore)
  allowed_chars = set(string.ascii_letters + string.digits + "_")

  # Remove punctuation (optional, comment out if not needed)
  punctuations = list(string.punctuation)
  for punctuation in punctuations:
    text = text.replace(punctuation, '')

  # Tokenize the text
  tokens = nltk.word_tokenize(text)

  # Filter and lowercase tokens, join into a string
  preprocessed_text = ''.join([token.lower() for token in tokens if all(char in allowed_chars for char in token)])

  return preprocessed_text


start_time = time.time()
# Example usage
text = "This is some text with special characters!@#$. It also has uppercase letters."
text ="BGNGIV@MEVRGBFVKGPSIK?E?RN?LBCW55-01-77-01264faskjjklasdf !@#$%^&oifaskdjls(&&)()"

# Preprocess the text
preprocessed_text = preprocess_text(text)

print(preprocessed_text)  # Output: thisissometextwithspecialcharactersitalsohasuppercaseletters
end_time = time.time()
execution_time = end_time - start_time
print(execution_time)

df= df.head(1000)

df.shape

class TrieNode:
    def __init__(self):
        self.children = {}  # Add this line to `TrieNode`
        self.is_end_of_word = False
        self.data = {'Name': None, 'Father Name': None, 'Idno': None}
        self.has_data = True  # Flag to indicate node stores data (optional)

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, name, father_name, idno):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.data['Name'] = name
        node.data['Father Name'] = father_name
        node.data['Idno'] = idno

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False, None, None
            node = node.children[char]
        return node.is_end_of_word, node.data.get('Name'), node.data.get('Father Name'), node.data.get('Idno')

    def search_fuzzy(self, query, threshold=0.8):
        results = []
        self._search_fuzzy_recursive(query, self.root, '', threshold, results)
        return results

    def _search_fuzzy_recursive(self, query, node, prefix, threshold, results):
        # Check base cases:
        if not node:  # Reached an empty node (no matching prefix so far)
            return

        # Explore child nodes for potential insertions
        for child, child_node in node.children.items():
            new_prefix = prefix + child
            self._search_fuzzy_recursive(query, child_node, new_prefix, threshold, results)

        # Check similarity and add exact matches with data (if applicable)
        similarity = similarity_find(query, prefix)
        if similarity >= threshold and node.is_end_of_word and (hasattr(node, 'has_data') and node.has_data):  # Check for data flag (optional)
            results.append((prefix, node.data))

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# Example usage:
trie = Trie()
# Assuming df is your DataFrame containing columns 'Name', 'Father Name', and 'Idno'
for index, row in df.iterrows():
    combined_text = row['Combined']
    preprocessed_word = preprocess_text(combined_text)  # Preprocess the combined text
    name = row['Name']
    father_name = row['Father Name']
    id_no = row['Idno']

    # Insert into Trie with preprocessed word
    trie.insert(word=preprocessed_word, name=name, father_name=father_name, idno=id_no)

print(f"Successfully inserted {len(df)} elements into the Trie.")

start_time = time.time()
query ="^23gbfViVjXS@?GP?L??PG0LQ0972222"
query = "@V1SPVKWFPCWRFV-JVJGRWFPCWRFV-BGLCWFWFPCWRFVXW?LJ?KGAFF?LCK?E?P0QQ0944546"
# Preprocess the text
query = preprocess_text(query)
print(query)
results = trie.search_fuzzy(query)

# Check if any results were found
if results:
    print("The search word was:", query)
    print("Search results:")
    for prefix, data in results:
        print(f"- {prefix} \n Name: {data.get('Name')} \n Father Name: {data.get('Father Name')}\n Idno: {data.get('Idno')}")  # Fixed parenthesis
else:
    print(f"No results found for '{query}'.")

end_time = time.time()
execution_time = end_time - start_time
print("Execution time ", execution_time)

# Sample data with long names (200 characters) for time complexity analysis
data = [
    ["abcgbfvivjxsgplpg0lq09722","Hari bahadur", "Father Name 1", 1],
    ["This is a very long name used  This string has exactly 200 characters.blahblahblahblahblahblahblahblah","Ram bahadur", "Father 21", 21],
    ["This is a very long name used  This string has exactly 200 characters.blehblehblehblehblehblahblahblah","Ram bahadur", "Father 21", 11],
    ["This is another name with 200 characters for testing fdsjlksfdjklsdflkfsdlkjsdkTrie insertion. The time complexity should ideally be O(n) where n is the length of the name being inserted.","Shyam Kumar", "Father Name 2", 2],
    ]
for word,name, father_name, idno in data:
    trie.insert(word,name, father_name, idno)

start_time = time.time()
query ="^23g bfViVjXS@?GP ?L??PG0LQ0972222"
# query = "@V1SPVKWFPCWRFV-JVJGRWFPCWRFV-BGLCWFWFPCWRFVXW?LJ?KGAFF?LCK?E?P0QQ0944546"
# Preprocess the text
query = preprocess_text(query)
print(query)
results = trie.search_fuzzy(query)

# Check if any results were found
if results:
    print("The search word was:", query)
    print("Search results:")
    for prefix, data in results:
        print(f"- {prefix} \n Name: {data.get('Name')} \n Father Name: {data.get('Father Name')}\n Idno: {data.get('Idno')}")  # Fixed parenthesis
else:
    print(f"No results found for '{query}'.")

end_time = time.time()
execution_time = end_time - start_time
print("Execution time ", execution_time)


# Endpoint to receive query and return search results
@app.post("/search/")
async def search(query: str):
    start_time = time.time()

    # Preprocess the text
    query = preprocess_text(query)
    print(query)
    results = trie.search_fuzzy(query)

    if results:
        search_results = []
        for prefix, data in results:
            search_results.append({
                "prefix": prefix,
                "Name": data.get('Name'),
                "Father Name": data.get('Father Name'),
                "Idno": data.get('Idno')
            })
        end_time = time.time()
        execution_time = end_time - start_time

        print("The search word was:", query)
        print("Search results:")
        for prefix, data in results:
            print(
                f"- {prefix} \n Name: {data.get('Name')} \n Father Name: {data.get('Father Name')}\n Idno: {data.get('Idno')}")  # Fixed parenthesis

        return {"execution_time": execution_time, "results": search_results}
    else:
        print(f"No results found for '{query}'.")
        raise HTTPException(status_code=404, detail=f"No results found for '{query}'.")


