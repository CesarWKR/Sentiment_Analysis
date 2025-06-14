import random
import nltk
import logging
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.utils import resample
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords 
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from src.utils.model_utils import analyze_data_distribution
import gc
import re


# Download resources from nltk only when the script is executed directly
nltk.download("wordnet")
nltk.download("stopwords")
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


def is_valid_generated_text(text):
    """Evaluates whether a generated text meets the minimum quality criteria."""
    text = text.strip()

    # Check if it has at least 4 words
    if len(text.split()) < 4:
        return False

    # Eliminate texts with only numbers or punctuation
    if re.fullmatch(r"[\d\s\.\,\!\?]*", text):
        return False

    # Eliminate if it contains unwanted characters
    if any(char in text for char in ['@', '&', '|']):
        return False

    # Eliminate if it starts with a link and ends with .com
    if re.match(r"^(https?:\/\/)?\S+\.com", text):
        return False

    # Detect if the text is just punctuation or whitespace
    if re.fullmatch(r"[\.!\?,\s]+", text):
        return False
    
    # Check if it contains at least one alphabetic character
    if not re.search(r"[a-zA-Z]", text):
        return False
    
    # Check if it not starts with a emoji or out of context character or hashtag
    if re.match(r"^[^\w\s]|^#", text):  # starts with non-alphanumeric character or space, or with #
        return False

    return True


def generate_synthetic_samples(class_label, prompt, num_samples=100, max_length=256, batch_size=32):  # Function to generate synthetic samples using GPT-2 if needed oversampling
    """Generate filtered synthetic texts using GPT-2 in batches for a specific class label."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2") # Load GPT-2 tokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token, so use EOS token
    
    generated_texts = []
    num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate number of batches needed

    attempts = 0
    max_attempts = num_samples * 5  # Limit attempts to avoid infinite loop

    pbar = tqdm(desc=f"Generating for class {class_label}", total=num_samples)
    while len(generated_texts) < num_samples and attempts < max_attempts:
        current_batch_size = min(batch_size, num_samples - len(generated_texts))
        prompts = [prompt] * current_batch_size

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # generated_texts.extend([text.strip() for text in decoded])
        cleaned = [text.strip() for text in decoded if is_valid_generated_text(text)]

        generated_texts.extend(cleaned[:num_samples - len(generated_texts)])  # Add only up to the required number of samples
        attempts += current_batch_size
        pbar.update(len(cleaned[:num_samples - len(generated_texts)]))

    pbar.close()
    print(f"🚦 VRAM Memory before cleanup and after generate synthetic samples: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    # Liberate VRAM memory by deleting the model and tokenizer
    del model
    del tokenizer
    torch.cuda.empty_cache() # Clear CUDA cache
    gc.collect() # Collect garbage to free up memory

    print(f"🚦 VRAM Memory after cleanup and before training: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    return pd.DataFrame({"text": generated_texts, "label": class_label})


def apply_downsampling(df, label_map, down_thresh=0.1):
    """
    Applies partial downsampling to the majority class if imbalance is detected.
    Automatically adjusts downsample_ratio based on real class distribution.
    """
    label_counts = df["label"].value_counts()
    max_label = label_counts.idxmax()
    max_count = label_counts[max_label]

    minority_counts = label_counts[label_counts.index != max_label]
    minority_avg = minority_counts.mean()

    imbalance_ratio = (max_count - minority_avg) / max_count

    if imbalance_ratio >= down_thresh:
        # Auto-adjust downsample ratio: more imbalance = more aggressive downsampling
        if imbalance_ratio > 0.6:
            downsample_ratio = 0.3
        elif imbalance_ratio > 0.4:
            downsample_ratio = 0.4
        else:
            downsample_ratio = 0.5

        target_count = int(minority_avg + (max_count - minority_avg) * (1 - downsample_ratio))
        class_name = label_map[max_label]
        print(f"⚖️ Downsampling class '{class_name}' (label '{max_label}) from {max_count} to {target_count} samples (auto-ratio={downsample_ratio})")

        df_majority = df[df.label == max_label]
        df_others = df[df.label != max_label]

        df_majority_downsampled = resample(  
            df_majority,
            replace=False,  
            n_samples=target_count,  
            random_state=42
        )

        df_downsampled = pd.concat([df_majority_downsampled, df_others])
        df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)

        return df_downsampled, max_label
    else:
        print("✅ No downsampling needed.")
        return df, None


def apply_oversampling(df, label_map):  # Without up_thresh parameter because oversampling is being applied to all minority classes to equalize the dataset
    """
    Applies oversampling with GPT-2 for minority classes if needed.
    """
    from src.preprocessing.clean_data import clean_generated_text
    label_counts = df["label"].value_counts()
    max_class_count = label_counts.max()
    augmented_dfs = [df]
    synthetic_dfs = []
    applied = False

    for label, count in label_counts.items():
        # ratio = count / max_class_count
        if count < max_class_count:  # If the class is underrepresented
            samples_needed = max_class_count - count
            class_name = label_map[label]
            print(f"🧠 Generating {samples_needed} synthetic samples for class '{class_name}' (label {label})...")
            prompt = f"Reddit post expressing {class_name.lower()} sentiment:"
            synthetic_df = generate_synthetic_samples(label, prompt, num_samples=samples_needed)

            # Clean the generated text 
            synthetic_df["text"] = synthetic_df["text"].apply(lambda x: clean_generated_text(x, prompt))
            synthetic_df = synthetic_df[synthetic_df["text"].str.strip().astype(bool)]
            # Filter out short texts
            synthetic_df = synthetic_df[synthetic_df["text"].apply(lambda x: len(x.split()) > 5)]

            augmented_dfs.append(synthetic_df)
            synthetic_dfs.append(synthetic_df)
            applied = True

    df_augmented = pd.concat(augmented_dfs, ignore_index=True)
    df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

    synthetic_df_total = pd.concat(synthetic_dfs, ignore_index=True) if synthetic_dfs else pd.DataFrame(columns=["text", "label"])  # Empty DataFrame if no synthetic samples were generated

    if applied:
        print("🔼 Oversampling applied to underrepresented classes.")
    else:
        print("✅ No oversampling needed.")

    
    return df_augmented, synthetic_df_total, applied


def balance_dataset(df, label_map={0: "Negative", 1: "Neutral", 2: "Positive"}, down_thresh=0.1, up_thresh=0.5):
    """
    Balances the dataset by applying downsampling to the majority class and oversampling to minority classes using GPT-2.
    """

    print("🔢 Initial class distribution:")
    analyze_data_distribution(df)  # Analyze initial data distribution

    df, applied_downsampling = apply_downsampling(df, label_map, down_thresh=down_thresh)
    df, df_synthetic, applied_oversampling = apply_oversampling(df, label_map)

    # Final logging of the balancing process
    if applied_downsampling and applied_oversampling:
        print("🔁 Balanced dataset using both downsampling and oversampling.\n")
    elif applied_downsampling:
        print("🔽 Balanced dataset using downsampling only.\n")
    elif applied_oversampling:
        print("🔼 Balanced dataset using oversampling only.\n")
    else:
        print("✅ Dataset is already balanced. No changes made.\n")

    print("🔢 Balanced class distribution: ")
    analyze_data_distribution(df)

    return df, df_synthetic
