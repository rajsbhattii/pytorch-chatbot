import nltk
import numpy as np
#nltk.download('punkt')  # Download the Punkt tokenizer if needed
from nltk.stem.porter import PorterStemmer

# Initialize the PorterStemmer for stemming words
stemmer = PorterStemmer()

# Function to tokenize a sentence into individual words
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Function to stem a word (reduce it to its root form)
def stem(word):
    return stemmer.stem(word.lower())

# Function to create a "bag of words" representation
# It returns a binary vector indicating the presence of words from allWords in the tokenized sentence
def bagOfWords(tokenizedSentence, allWords):
    # Stem each word in the tokenized sentence
    tokenizedSentence = [stem(w) for w in tokenizedSentence]
    
    # Initialize a bag of words vector with zeros, the same length as allWords
    bag = np.zeros(len(allWords), dtype=np.float32)
    
    # Iterate over all words and mark their presence in the tokenized sentence
    for idx, w, in enumerate(allWords):
        if w in tokenizedSentence:
            bag[idx] = 1.0  # Mark 1 if the word exists in the sentence
    
    return bag

# Example usage of the bagOfWords function
sent = ["hello", "how", "are", "you"]
words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
bog = bagOfWords(sent, words)
print(bog)  # Output the bag of words representation

# Example usage of the stem function
words = ['Work', 'worked', 'working', 'works']
stemmedWords = [stem(w) for w in words]
print(stemmedWords)  # Output the stemmed words
