# from nltk.corpus import brown
# import re
# from collections import Counter

# def preprocess_sentences(sentences):
#     """
#     Lowercase words, remove punctuation-only tokens, add start and end tokens to sentences.
#     """
#     processed_sentences = []
#     for sentence in sentences:
#         # Lowercase and remove punctuation-only tokens
#         processed_sentence = [word.lower() for word in sentence if re.search(r'[A-Za-z0-9]', word)]
#         # Add start and end tokens
#         processed_sentence = ['<s>'] + processed_sentence + ['</s>']
#         processed_sentences.append(processed_sentence)
#     return processed_sentences

# def compute_unigram_bigram_counts(sentences):
#     """
#     Compute unigram and bigram counts for the given sentences.
#     """
#     unigram_counts = Counter()
#     bigram_counts = Counter()
    
#     for sentence in sentences:
#         # Update unigram counts
#         unigram_counts.update(sentence)
#         # Update bigram counts
#         bigram_counts.update(zip(sentence, sentence[1:]))
        
#     return unigram_counts, bigram_counts

# # Load and preprocess the data
# news_sentences = preprocess_sentences(brown.sents(categories='news'))
# romance_sentences = preprocess_sentences(brown.sents(categories='romance'))

# # Compute unigram and bigram counts
# news_unigram_counts, news_bigram_counts = compute_unigram_bigram_counts(news_sentences)
# romance_unigram_counts, romance_bigram_counts = compute_unigram_bigram_counts(romance_sentences)

# # Count non-zero unigrams for each corpus
# non_zero_unigrams_news = len(news_unigram_counts)
# non_zero_unigrams_romance = len(romance_unigram_counts)

# print("Non-zero unigrams in news corpus:", non_zero_unigrams_news)
# print("Non-zero unigrams in romance corpus:", non_zero_unigrams_romance)

# # Count non-zero bigrams for each corpus
# non_zero_bigrams_news = len(news_bigram_counts)
# non_zero_bigrams_romance = len(romance_bigram_counts)

# print("Non-zero bigrams in news corpus:", non_zero_bigrams_news)
# print("Non-zero bigrams in romance corpus:", non_zero_bigrams_romance)

# def compute_unigram_probabilities(unigram_counts, total_tokens):
#     """
#     Compute the Maximum Likelihood Estimation (MLE) probabilities for unigrams.
#     """
#     return {word: count / total_tokens for word, count in unigram_counts.items()}

# # Total number of tokens for each corpus (including <s> and </s>)
# total_tokens_news = sum(news_unigram_counts.values())
# total_tokens_romance = sum(romance_unigram_counts.values())

# # Compute unigram probabilities
# news_unigram_probs = compute_unigram_probabilities(news_unigram_counts, total_tokens_news)
# romance_unigram_probs = compute_unigram_probabilities(romance_unigram_counts, total_tokens_romance)

# # Get the 10 most common unigrams and their probabilities
# most_common_unigrams_news = sorted(news_unigram_probs.items(), key=lambda x: x[1], reverse=True)[:10] 
# most_common_unigrams_romance = sorted(romance_unigram_probs.items(), key=lambda x: x[1], reverse=True)[:10]

# print('most_common_unigrams_news=',most_common_unigrams_news, 'most_common_unigrams_romance=',most_common_unigrams_romance)

# # def compute_bigram_probabilities(bigram_counts, unigram_counts):
# #     """
# #     Compute the MLE probabilities for bigrams.
# #     """
# #     bigram_probs = {}
# #     for bigram, bigram_count in bigram_counts.items():
# #         first_word = bigram[0]
# #         first_word_count = unigram_counts[first_word]
# #         bigram_probs[bigram] = bigram_count / first_word_count
# #     return bigram_probs

# # # Compute bigram probabilities for each corpus
# # news_bigram_probs = compute_bigram_probabilities(news_bigram_counts, news_unigram_counts)
# # romance_bigram_probs = compute_bigram_probabilities(romance_bigram_counts, romance_unigram_counts)

# # # Get the 10 most common bigrams and their probabilities
# # most_common_bigrams_news = sorted(news_bigram_probs.items(), key=lambda x: x[1], reverse=True)[:10]
# # most_common_bigrams_romance = sorted(romance_bigram_probs.items(), key=lambda x: x[1], reverse=True)[:10]

# # def compute_bigram_probabilities(bigram_counts, unigram_counts):
# #     """
# #     Compute the MLE probabilities for bigrams.
# #     """
# #     bigram_probs = {}
# #     for bigram, bigram_count in bigram_counts.items():
# #         first_word = bigram[0]
# #         first_word_count = unigram_counts[first_word]
# #         bigram_probs[bigram] = bigram_count / first_word_count
# #     return bigram_probs

# # # Assuming news_bigram_counts, romance_bigram_counts, news_unigram_counts, romance_unigram_counts are defined
# # # Compute bigram probabilities for each corpus
# # news_bigram_probs = compute_bigram_probabilities(news_bigram_counts, news_unigram_counts)
# # romance_bigram_probs = compute_bigram_probabilities(romance_bigram_counts, romance_unigram_counts)

# # # Get the 10 most common bigrams and their probabilities
# # most_common_bigrams_news = sorted(news_bigram_probs.items(), key=lambda x: x[1], reverse=True)[:10]
# # most_common_bigrams_romance = sorted(romance_bigram_probs.items(), key=lambda x: x[1], reverse=True)[:10]

# # # Print the 10 most common bigrams and their probabilities for news
# # print("News Corpus - 10 Most Common Bigrams and Their Probabilities:")
# # for bigram, probability in most_common_bigrams_news:
# #     print(f"{bigram}: {probability:.4f}")

# # # Print the 10 most common bigrams and their probabilities for romance
# # print("\nRomance Corpus - 10 Most Common Bigrams and Their Probabilities:")
# # for bigram, probability in most_common_bigrams_romance:
# #     print(f"{bigram}: {probability:.4f}")
# # def compute_bigram_probabilities(bigram_counts, unigram_counts):
# #     """
# #     Compute the MLE probabilities for bigrams.
# #     """
# #     bigram_probs = {}
# #     for bigram, bigram_count in bigram_counts.items():
# #         first_word = bigram[0]
# #         first_word_count = unigram_counts[first_word]
# #         bigram_probs[bigram] = bigram_count / first_word_count
# #     return bigram_probs

# # # Assuming news_bigram_counts, romance_bigram_counts, news_unigram_counts, romance_unigram_counts are defined
# # # Compute bigram probabilities for each corpus
# # news_bigram_probs = compute_bigram_probabilities(news_bigram_counts, news_unigram_counts)
# # romance_bigram_probs = compute_bigram_probabilities(romance_bigram_counts, romance_unigram_counts)

# # # Get the 10 most common bigrams and their probabilities
# # most_common_bigrams_news = sorted(news_bigram_probs.items(), key=lambda x: x[1], reverse=True)[:10]
# # most_common_bigrams_romance = sorted(romance_bigram_probs.items(), key=lambda x: x[1], reverse=True)[:10]

# # # Print the 10 most common bigrams and their probabilities for news
# # print("News Corpus - 10 Most Common Bigrams and Their Probabilities:")
# # for bigram, probability in most_common_bigrams_news:
# #     print(f"{bigram}: {probability:.4f}")

# # # Print the 10 most common bigrams and their probabilities for romance
# # print("\nRomance Corpus - 10 Most Common Bigrams and Their Probabilities:")
# # for bigram, probability in most_common_bigrams_romance:
# #     print(f"{bigram}: {probability:.4f}")


#     # Compute the probabilities for each bigram in the news corpus
# news_bigram_probabilities = {bigram: count / news_unigram_counts[bigram[0]] for bigram, count in news_bigram_counts.items()}
# # Get the 10 most common bigrams for the news corpus
# most_common_bigrams_news = news_bigram_counts.most_common(10)

# # Compute the probabilities for each bigram in the romance corpus
# romance_bigram_probabilities = {bigram: count / romance_unigram_counts[bigram[0]] for bigram, count in romance_bigram_counts.items()}
# # Get the 10 most common bigrams for the romance corpus
# most_common_bigrams_romance = romance_bigram_counts.most_common(10)

# # Print the results in a table format
# print("10 Most Common Bigrams and Their Probabilities for News Corpus:")
# print("{:<25} {:<10} {:<15}".format("Bigram", "Count", "Probability"))
# for bigram, count in most_common_bigrams_news:
#     probability = news_bigram_probabilities[bigram]
#     print("{:<25} {:<10} {:<15}".format(bigram, count, probability))

# print("\n10 Most Common Bigrams and Their Probabilities for Romance Corpus:")
# print("{:<25} {:<10} {:<15}".format("Bigram", "Count", "Probability"))
# for bigram, count in most_common_bigrams_romance:
#     probability = romance_bigram_probabilities[bigram]
#     print("{:<25} {:<10} {:<15}".format(bigram, count, probability))

import nltk
from nltk.corpus import brown
from collections import Counter

def preprocess_sentences(sentences):
    preprocessed = []
    for sentence in sentences:
        # Lowercase all words
        sentence = [word.lower() for word in sentence]
        # Remove tokens that consist only of punctuation
        sentence = [word for word in sentence if any(char.isalnum() for char in word)]
        # Add <s> and </s> before and after each sentence
        sentence = ['<s>'] + sentence + ['</s>']
        preprocessed.append(sentence)
    return preprocessed

def compute_unigram_bigram_models(data):
    unigrams = Counter()
    bigrams = Counter()
    
    for sentence in data:
        unigrams.update(sentence)
        bigrams.update(nltk.bigrams(sentence))
        
    return unigrams, bigrams

# Download the Brown corpus
nltk.download('brown')

# Load the Brown corpus data for news and romance categories
news_data = brown.sents(categories='news')
romance_data = brown.sents(categories='romance')

# Preprocess the sentences
preprocessed_news_data = preprocess_sentences(news_data)
preprocessed_romance_data = preprocess_sentences(romance_data)

# Compute unigram and bigram models
news_unigrams, news_bigrams = compute_unigram_bigram_models(preprocessed_news_data)
romance_unigrams, romance_bigrams = compute_unigram_bigram_models(preprocessed_romance_data)

# Count non-zero unigrams
non_zero_unigrams_news = sum(1 for count in news_unigrams.values() if count > 0)
non_zero_unigrams_romance = sum(1 for count in romance_unigrams.values() if count > 0)

print("Number of non-zero unigrams in news corpus:", non_zero_unigrams_news)
print("Number of non-zero unigrams in romance corpus:", non_zero_unigrams_romance)

# Count non-zero bigrams
non_zero_bigrams_news = sum(1 for count in news_bigrams.values() if count > 0)
non_zero_bigrams_romance = sum(1 for count in romance_bigrams.values() if count > 0)

print("Number of non-zero bigrams in news corpus:", non_zero_bigrams_news)
print("Number of non-zero bigrams in romance corpus:", non_zero_bigrams_romance)

def calculate_unigram_probabilities(unigram_counts):
    total_count = sum(unigram_counts.values())
    probabilities = {word: count / total_count for word, count in unigram_counts.items()}
    return probabilities

# Calculate unigram probabilities for news and romance corpora
news_unigram_probabilities = calculate_unigram_probabilities(news_unigrams)
romance_unigram_probabilities = calculate_unigram_probabilities(romance_unigrams)

# Get the 10 most common unigrams for each corpus
top_10_news_unigrams = news_unigrams.most_common(10)
top_10_romance_unigrams = romance_unigrams.most_common(10)

# Print the results in a table
print("Top 10 Most Common Unigrams in News Corpus:")
print("{:<10} {:<15} {:<15}".format("Unigram", "Count", "Probability"))
for unigram, count in top_10_news_unigrams:
    print("{:<10} {:<15} {:.6f}".format(unigram, count, news_unigram_probabilities[unigram]))

print("\nTop 10 Most Common Unigrams in Romance Corpus:")
print("{:<10} {:<15} {:<15}".format("Unigram", "Count", "Probability"))
for unigram, count in top_10_romance_unigrams:
    print("{:<10} {:<15} {:.6f}".format(unigram, count, romance_unigram_probabilities[unigram]))
def calculate_bigram_probabilities_mle(bigram_counts, unigram_counts):
    probabilities = {}
    for bigram, count in bigram_counts.items():
        word1, word2 = bigram
        probability = count / unigram_counts[word1]
        probabilities[bigram] = probability
    return probabilities

# Calculate bigram probabilities using MLE for news and romance corpora
news_bigram_probabilities_mle = calculate_bigram_probabilities_mle(news_bigrams, news_unigrams)
romance_bigram_probabilities_mle = calculate_bigram_probabilities_mle(romance_bigrams, romance_unigrams)

# Get the 10 most common bigrams for each corpus
top_10_news_bigrams = news_bigrams.most_common(10)
top_10_romance_bigrams = romance_bigrams.most_common(10)

# Print the results in a table
print("Top 10 Most Common Bigrams in News Corpus:")
print("{:<20} {:<10} {:<15}".format("Bigram", "Count", "Probability"))
for bigram, count in top_10_news_bigrams:
    probability = news_bigram_probabilities_mle[bigram]
    print("{:<20} {:<10} {:.6f}".format(' '.join(bigram), count, probability))

print("\nTop 10 Most Common Bigrams in Romance Corpus:")
print("{:<20} {:<10} {:<15}".format("Bigram", "Count", "Probability"))
for bigram, count in top_10_romance_bigrams:
    probability = romance_bigram_probabilities_mle[bigram]
    print("{:<20} {:<10} {:.6f}".format(' '.join(bigram), count, probability))


#e
def compute_sentence_probability(sentence, unigram_counts, bigram_counts):
    # Add <s> and </s> tokens to the sentence
    sentence = ['<s>'] + sentence.split() + ['</s>']
    n = 2  # Bigram model
    probability = 1.0
    
    for i in range(len(sentence) - n + 1):
        word1, word2 = sentence[i:i+2]
        if (word1, word2) in bigram_counts:
            probability *= bigram_counts[(word1, word2)] / unigram_counts[word1]
        else:
            # If the bigram is not found in the model, set probability to zero
            probability = 0.0
            break
    
    return probability

# Example usage of the function with news data
news_sentence = "<s> I loved her when she laughed </s>"
news_bigram_probability = compute_sentence_probability(news_sentence, news_unigrams, news_bigrams)
print("Probability of the sentence using the bigram model from the news data:", news_bigram_probability)

#f
# Example usage of the function with romance data
romance_sentence = "<s> I loved her when she laughed </s>"
romance_bigram_probability = compute_sentence_probability(romance_sentence, romance_unigrams, romance_bigrams)
print("Probability of the sentence using the bigram model from the romance data:", romance_bigram_probability)

#g
def compute_sentence_probability_with_smoothing(sentence, unigram_counts, bigram_counts, vocabulary_size):
    # Add <s> and </s> tokens to the sentence
    sentence = ['<s>'] + sentence.split() + ['</s>']
    n = 2  # Bigram model
    probability = 1.0
    V = vocabulary_size
    
    for i in range(len(sentence) - n + 1):
        word1, word2 = sentence[i:i+2]
        count_bigram = bigram_counts.get((word1, word2), 0)
        count_unigram = unigram_counts.get(word1, 0)
        probability *= (count_bigram + 1) / (count_unigram + V)
    
    return probability

# Example usage of the function with news data with add-one smoothing
news_sentence = "<s> I loved her when she laughed </s>"
news_bigram_probability_with_smoothing = compute_sentence_probability_with_smoothing(news_sentence, news_unigrams, news_bigrams, len(news_unigrams))
print("Probability of the sentence using the bigram model from the news data with add-one smoothing:", news_bigram_probability_with_smoothing)

# Example usage of the function with romance data with add-one smoothing
romance_sentence = "<s> I loved her when she laughed </s>"
romance_bigram_probability_with_smoothing = compute_sentence_probability_with_smoothing(romance_sentence, romance_unigrams, romance_bigrams, len(romance_unigrams))
print("Probability of the sentence using the bigram model from the romance data with add-one smoothing:", romance_bigram_probability_with_smoothing)
    


  





