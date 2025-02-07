import nltk
from nltk.corpus import brown
from collections import Counter
from nltk.util import bigrams

nltk.download('brown')
def preprocess_sentences(sentences):
    """
    Preprocess sentences from the corpus:
    - Lowercase all words.
    - Remove tokens consisting only of punctuation.
    - Add start and end tokens to each sentence.
    """
    preprocessed = []
    for sentence in sentences:
        sentence = [word.lower() for word in sentence if any(char.isalnum() for char in word)]
        sentence = ['<s>'] + sentence + ['</s>']
        preprocessed.append(sentence)
    return preprocessed

def compute_unigram_bigram_models(data):
    """
    Compute unigram and bigram models from preprocessed data:
    - Count occurrences of each unigram and bigram.
    """
    unigrams = Counter()
    bigrams = Counter()
    
    for sentence in data:
        unigrams.update(sentence)
        bigrams.update(nltk.bigrams(sentence))
        
    return unigrams, bigrams

news_data = brown.sents(categories='news')
romance_data = brown.sents(categories='romance')

# Preprocessing the sentences
preprocessed_news_data = preprocess_sentences(news_data)
preprocessed_romance_data = preprocess_sentences(romance_data)

# Computing unigram and bigram models
news_unigrams, news_bigrams = compute_unigram_bigram_models(preprocessed_news_data)
romance_unigrams, romance_bigrams = compute_unigram_bigram_models(preprocessed_romance_data)

# Counting non-zero unigrams
non_zero_unigrams_news = sum(1 for count in news_unigrams.values() if count > 0)
non_zero_unigrams_romance = sum(1 for count in romance_unigrams.values() if count > 0)

print("Number of non-zero unigrams in news corpus:", non_zero_unigrams_news)
print("Number of non-zero unigrams in romance corpus:", non_zero_unigrams_romance)

# Counting non-zero bigrams
non_zero_bigrams_news = sum(1 for count in news_bigrams.values() if count > 0)
non_zero_bigrams_romance = sum(1 for count in romance_bigrams.values() if count > 0)

print("Number of non-zero bigrams in news corpus:", non_zero_bigrams_news)
print("Number of non-zero bigrams in romance corpus:", non_zero_bigrams_romance)

def calculate_unigram_probabilities(unigram_counts):
    total_count = sum(unigram_counts.values())
    probabilities = {word: count / total_count for word, count in unigram_counts.items()}
    return probabilities

# Calculating unigram probabilities for news and romance corpora
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

news_bigram_probabilities_mle = calculate_bigram_probabilities_mle(news_bigrams, news_unigrams)
romance_bigram_probabilities_mle = calculate_bigram_probabilities_mle(romance_bigrams, romance_unigrams)

top_10_news_bigrams = news_bigrams.most_common(10)
top_10_romance_bigrams = romance_bigrams.most_common(10)


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


def compute_sentence_probability(unigram_counts, bigram_counts, sentence, smoothing=False, vocabulary_size=0):
    """
    Calculate the probability of a sentence using unigram and bigram models.
    Optionally apply add-one smoothing.
    """
    tokens = ['<s>'] + sentence.lower().split() + ['</s>']
    bigram_prob = 1.0
    
    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i+1])
        bigram_count = bigram_counts[bigram]
        unigram_count = unigram_counts[bigram[0]]
        
        if smoothing:
            bigram_prob *= (bigram_count + 1) / (unigram_count + vocabulary_size)
        else:
            if bigram_count > 0:
                bigram_prob *= bigram_count / unigram_count
            else:
                return 0.0  # Return zero if the bigram doesn't exist (without smoothing)
    return bigram_prob

# Load the Brown corpus data for news and romance categories
news_data = brown.sents(categories='news')
romance_data = brown.sents(categories='romance')

# Load and preprocess data from the Brown corpus
news_data = preprocess_sentences(brown.sents(categories='news'))
romance_data = preprocess_sentences(brown.sents(categories='romance'))

# Compute unigram and bigram models
news_unigrams, news_bigrams = compute_unigram_bigram_models(news_data)
romance_unigrams, romance_bigrams = compute_unigram_bigram_models(romance_data)

# Example sentence
sentence = "I loved her when she laughed"

# Calculate sentence probabilities without smoothing
news_prob = compute_sentence_probability(news_unigrams, news_bigrams, sentence)
romance_prob = compute_sentence_probability(romance_unigrams, romance_bigrams, sentence)
print("News corpus probability:", news_prob)
print("Romance corpus probability:", romance_prob)

# Calculate sentence probabilities with add-one smoothing
news_prob_smoothing = compute_sentence_probability(news_unigrams, news_bigrams, sentence, True, len(news_unigrams))
romance_prob_smoothing = compute_sentence_probability(romance_unigrams, romance_bigrams, sentence, True, len(romance_unigrams))
print("News corpus probability with smoothing:", news_prob_smoothing)
print("Romance corpus probability with smoothing:", romance_prob_smoothing)