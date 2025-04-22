import random
import nltk
from nltk.corpus import wordnet

# Download resources from nltk only when the script is executed directly
if __name__ == "__main__":
    nltk.download("wordnet")

def synonym_replacement(text, num_replacements=1):
    """
    Replaces random words in the text with their synonyms using WordNet.
    
    :param text: The input text
    :param num_replacements: Number of words to replace
    :return: Augmented text with synonyms
    """

    words = text.split()

    if not words:  # If the text is empty, return it as is
        return text

    new_words = words.copy()
    words_with_synonyms = [word for word in words if wordnet.synsets(word)]  # Filter words with synonyms

    if not words_with_synonyms:  # If no words have synonyms, return the original text
        return text
    

    for _ in range(num_replacements):
        word_idx = random.randint(0, len(new_words) - 1)  # Randomly select an index
        synonyms = wordnet.synsets(new_words[word_idx])   # Get synonyms for the selected word
    
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()  # Get the first synonym
            new_words[word_idx] = synonym.replace("_", " ")  # Replace in text

    return " ".join(new_words)

def word_dropout(text, dropout_prob=0.2):
    """
    Randomly removes words from the text with a given probability.
    
    :param text: The input text
    :param dropout_prob: Probability of removing each word
    :return: Augmented text with some words dropped
    """
    words = text.split()
    new_words = [word for word in words if random.random() > dropout_prob]

    if not new_words:  # If all words are dropped, return a random word from the original text
        return random.choice(words)
    
    #return " ".join(new_words) if new_words else text  # Ensure the text is not empty
    return " ".join(new_words)  # Return the modified text

def apply_data_augmentation(text):
    """
    Applies multiple augmentation techniques to a given text.
    
    :param text: The input text
    :return: A list of augmented texts including the original
    """

    text = text.strip()  # Remove leading/trailing whitespace

    if not text:   # If the text is empty, return it as is
        return [text]

    augmented_texts = [text]  # Include the original text

    # Apply synonym replacement
    synonym_text = synonym_replacement(text)
    if synonym_text != text:
        augmented_texts.append(synonym_text)

    # Apply word dropout
    dropout_text = word_dropout(text)
    if dropout_text != text:
        augmented_texts.append(dropout_text)
    
    return augmented_texts
