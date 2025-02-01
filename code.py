import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Ensure the necessary NLTK data is downloaded (this should work in environments like Google Colab)
nltk.download('punkt')

# Path to your text file
file_path = 'data.txt'  # Updated file path to the uploaded text file

# Read the dataset and extract the messages
messages = []
with open(file_path, 'r') as file:
    for line in file:
        # Assuming each line in the file is a separate message
        messages.append(line.strip())

# Join messages into a single text and tokenize into sentences
text = ' '.join(messages)
sentences = sent_tokenize(text)

# (a) Counting sentences, ignoring empty lines and empty sentences
sentence_count = len([sent for sent in sentences if sent.strip()])

# (b) Counting tokens using split()
tokens_split = [word for sent in sentences for word in sent.split()]
token_count_split = len(tokens_split)

# (c) Counting tokens using NLTK's word_tokenize()
tokens_nltk = [word for sent in sentences for word in word_tokenize(sent)]
token_count_nltk = len(tokens_nltk)

# (d) Lowercasing all words and counting unique tokens (types)
lowercase_tokens = [word.lower() for word in tokens_nltk]
unique_tokens = set(lowercase_tokens)
token_count_lower = len(lowercase_tokens)
type_count_lower = len(unique_tokens)

# (e) Comparison of token counts

# (f), (g) Making a dictionary of word type counts and finding the most frequent types
word_freq = nltk.FreqDist(lowercase_tokens)
most_common = word_freq.most_common()

# Most frequent word type
most_frequent_word = most_common[0]

# 5th most frequent word type
fifth_most_frequent_word = most_common[4]

# (h) Zipf's Law Graph
# Extracting ranks and frequencies
ranks = range(1, len(most_common) + 1)
frequencies = [freq for word, freq in most_common]

# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.loglog(ranks, frequencies)
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title('Zipf\'s Law')
plt.show()

# Outputs
print(f"Number of sentences: {sentence_count}")
print(f"Number of tokens using split: {token_count_split}")
print(f"Number of tokens using NLTK's word_tokenize: {token_count_nltk}")
print(f"Number of lowercase tokens: {token_count_lower}, Number of types: {type_count_lower}")
print(f"Most frequent word type: {most_frequent_word}")
print(f"5th most frequent word type: {fifth_most_frequent_word}")
