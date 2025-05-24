import random
import nltk
import logging
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords 
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# Download resources from nltk only when the script is executed directly
nltk.download("wordnet")
nltk.download("stopwords")
# nltk.download('averaged_perceptron_tagger')
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
model = SentenceTransformer("all-MiniLM-L6-v2")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def get_wordnet_pos(treebank_tag):
    """Map POS tag from nltk to wordnet POS"""
    if treebank_tag.startswith("J"):
        return wn.ADJ
    elif treebank_tag.startswith("V"):
        return wn.VERB
    elif treebank_tag.startswith("N"):
        return wn.NOUN
    elif treebank_tag.startswith("R"):
        return wn.ADV
    else:
        return None

def synonym_replacement(text, num_replacements=1):
    """
    Replaces random words in the text with their synonyms using WordNet.
    
    :param text: The input text
    :param num_replacements: Number of words to replace
    :return: Augmented text with synonyms
    """

    words = text.split()
    tagged_words = pos_tag(words)
    valid_words = [(word, get_wordnet_pos(pos)) for word, pos in tagged_words if get_wordnet_pos(pos)]

    if not words or not valid_words:  # If the text is empty, return it as is
        return text

    new_words = words.copy()
    # words_with_synonyms = [word for word in words if wordnet.synsets(word)]  # Filter words with synonyms

    new_words = words[:]
    for _ in range(num_replacements):
        word_idx = random.randint(0, len(valid_words) - 1)
        word, pos = valid_words[word_idx]
        synsets = wn.synsets(word, pos=pos) # Get synsets for the word with the specified POS

        if synsets:
            lemmas = [lemma.name() for lemma in synsets[0].lemmas() if lemma.name().lower() != word.lower()] # Exclude the original word 
            if lemmas:
                synonym = random.choice(lemmas).replace("_", " ") # Randomly select a synonym
                new_words[words.index(word)] = synonym # Replace the word in the text

    return " ".join(new_words)

def word_dropout(text, dropout_prob=0.1):
    """
    Randomly removes words from the text with a given probability.
    
    :param text: The input text
    :param dropout_prob: Probability of removing each word
    :return: Augmented text with some words dropped
    """
    words = text.split()
    # new_words = [word for word in words if random.random() > dropout_prob]
    new_words = []

    for word in words: 
        if word.lower() in ["not", "no", "never", "nor"] or len(word) <= 2:  
            new_words.append(word)
        elif random.random() > dropout_prob: # Keep the word with a certain probability
            new_words.append(word)

    return " ".join(new_words) if new_words else text # Return the original text if all words are dropped


def is_valid_augmentation(original, augmented, threshold=0.7):
    emb_orig = model.encode([original]) # Encode the original text
    emb_aug = model.encode([augmented]) # Encode the augmented text
    sim = cosine_similarity(emb_orig, emb_aug)[0][0] # Compute cosine similarity
    return sim >= threshold # Check if the similarity is above the threshold

def apply_data_augmentation(text):
    """
    Applies multiple augmentation techniques to a given text.
    
    :param text: The input text
    :return: A list of augmented texts including the original
    """

    text = text.strip()  # Remove leading/trailing whitespace

    if not text:   # If the text is empty, return it as a tuple
        return [(text, None)]

    # augmented_texts = [(text, "original")]  # Include the original text with a label
    augmented_texts = []  # List to store augmented texts, not including the original text

    # Apply synonym replacement
    synonym_text = synonym_replacement(text)
    if synonym_text != text:
        # augmented_texts.append(synonym_text, "synonym_replacement")  # This is only one argument
        augmented_texts.append((synonym_text, "synonym_replacement"))  # Append as a tuple

    # Apply word dropout
    dropout_text = word_dropout(text)
    if dropout_text != text:
        # augmented_texts.append(dropout_text, "word_dropout")
        augmented_texts.append((dropout_text, "word_dropout"))  # Append as a tuple

    
    # If no augmentations were added, return the original text as fallback
    if not augmented_texts:
        augmented_texts.append((text, None))  # Mark as not augmented


    return augmented_texts
