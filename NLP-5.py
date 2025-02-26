# -*- coding: utf-8 -*-
"""Untitled6.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Pzj-TZtoBFCoK1HbRGIL9ZYPkbq_Nbv6
"""

#a
import nltk
from nltk.corpus import wordnet as wn

# Ensure that WordNet is downloaded
nltk.download('wordnet')

# Retrieve the first noun synset for the word 'house'
house_synset = wn.synset('house.n.01')

# Get the hypernyms of this synset
house_hypernyms = house_synset.hypernyms()

# Print the hypernyms
print("Hypernyms of the first noun definition of 'house':")
for hypernym in house_hypernyms:
    print(hypernym.name(), hypernym.definition())

#b
import nltk
from nltk.corpus import wordnet as wn

# Ensure that WordNet is downloaded
nltk.download('wordnet')

# Retrieve the first noun synset for 'mouse' and 'horse'
mouse_synset = wn.synset('mouse.n.01')
horse_synset = wn.synset('horse.n.01')

# Calculate and print path similarity between 'mouse' and 'horse'
mouse_horse_similarity = mouse_synset.path_similarity(horse_synset)
print(f"Path similarity between 'mouse' and 'horse': {mouse_horse_similarity}")

# Retrieve the first noun synset for 'vacation'
vacation_synset = wn.synset('vacation.n.01')

# Calculate and print path similarity between 'horse' and 'vacation'
horse_vacation_similarity = horse_synset.path_similarity(vacation_synset)
print(f"Path similarity between 'horse' and 'vacation': {horse_vacation_similarity}")

# Determine which pair is more similar
if mouse_horse_similarity > horse_vacation_similarity:
    print("The pair 'mouse' and 'horse' is more similar.")
else:
    print("The pair 'horse' and 'vacation' is more similar.")

import numpy as np

def load_glove_embeddings(file_path):
    embeddings_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0  # Prevent division by zero
    similarity = dot_product / (norm_v1 * norm_v2)
    return similarity

# Example usage
glove_file = 'glove.6B.50d.txt'  # Adjust the path to where you have saved the GloVe file
embeddings = load_glove_embeddings(glove_file)

word1 = 'king'
word2 = 'queen'

# Ensure words are in the embeddings dictionary
if word1 in embeddings and word2 in embeddings:
    sim = cosine_similarity(embeddings[word1], embeddings[word2])
    print(f"Cosine similarity between '{word1}' and '{word2}': {sim}")
else:
    print("One or both words not found in the embeddings.")

import numpy as np

def load_glove_embeddings(file_path):
    embeddings_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0  # Prevent division by zero
    similarity = dot_product / (norm_v1 * norm_v2)
    return similarity

# Example usage
glove_file = 'glove.6B.50d.txt'  # Adjust the path to where you have saved the GloVe file
embeddings = load_glove_embeddings(glove_file)

# Define the word pairs
pairs = [
    ('mouse', 'horse'),
    ('horse', 'vacation')
]

# Compute and compare the cosine similarity for each pair
results = {}
for word1, word2 in pairs:
    if word1 in embeddings and word2 in embeddings:
        sim = cosine_similarity(embeddings[word1], embeddings[word2])
        results[(word1, word2)] = sim
        print(f"Cosine similarity between '{word1}' and '{word2}': {sim}")
    else:
        print(f"One or both words not found in the embeddings for pair: {word1}-{word2}")

# Determine which pair has higher similarity
if results:
    more_similar_pair = max(results, key=results.get)
    print(f"The pair with higher cosine similarity is '{more_similar_pair[0]}' and '{more_similar_pair[1]}' with a similarity of {results[more_similar_pair]}")