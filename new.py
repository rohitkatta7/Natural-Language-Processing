import nltk
from nltk.corpus import brown
from nltk import bigrams
from collections import Counter

class LanguageModel:
    def __init__(self, category):
        self.category = category
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.vocabulary_size = 0

        # Initialize and compute models
        self._download_corpus()
        self._compute_models()

    def _download_corpus(self):
        nltk.download('brown', quiet=True)

    def _preprocess_sentences(self, sentences):
        preprocessed = []
        for sentence in sentences:
            sentence = ['<s>'] + [word.lower() for word in sentence if any(char.isalnum() for char in word)] + ['</s>']
            preprocessed.append(sentence)
        return preprocessed

    def _compute_models(self):
        sentences = brown.sents(categories=self.category)
        preprocessed_data = self._preprocess_sentences(sentences)
        for sentence in preprocessed_data:
            self.unigrams.update(sentence)
            self.bigrams.update(bigrams(sentence))
        self.vocabulary_size = len(self.unigrams)

        def calculate_probability(self, sentence, smoothing=False):
            sentence = ['<s>'] + sentence.lower().split() + ['</s>']
            probability = 1.0
            for i in range(len(sentence) - 1):
                bigram = (sentence[i], sentence[i+1])
                bigram_count = self.bigrams[bigram]
                unigram_count = self.unigrams[bigram[0]]
                if smoothing:
                    probability *= (bigram_count + 1) / (unigram_count + self.vocabulary_size)
                else:
                    if bigram_count > 0:
                        probability *= bigram_count / unigram_count
                    else:
                        return 0.0  # Bigram not found
            return probability

# Example usage:
news_model = LanguageModel('news')
romance_model = LanguageModel('romance')

sentence = "I loved her when she laughed"
news_probability = news_model.calculate_probability(sentence)
romance_probability = romance_model.calculate_probability(sentence)
print("News corpus probability:", news_probability)
print("Romance corpus probability:", romance_probability)

# With smoothing
news_probability_smoothing = news_model.calculate_probability(sentence, smoothing=True)
romance_probability_smoothing = romance_model.calculate_probability(sentence, smoothing=True)
print("News corpus probability with smoothing:", news_probability_smoothing)
print("Romance corpus probability with smoothing:", romance_probability_smoothing)
