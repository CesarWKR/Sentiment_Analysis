import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained model and tokenizer
model_path = "bert_sentiment_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()  # Set model to evaluation mode

# Sentiment labels
LABELS = ["Negative", "Neutral", "Positive"]

def predict_sentiment(text):
    """
    Predicts the sentiment of a given text using the fine-tuned BERT model.
    
    :param text: The input text for sentiment analysis.
    :return: The predicted sentiment label index.
    """
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        # max_length=128,       # Use the same max_length as in training
        max_length=512,         # Use this if you want to use the maximum length of BERT (512), but be careful with the batch size and inconsistency beeween the coherence of training and testing
        # padding=True,         # Dinamic padding instead of max_length, use if you want to pad the text to the max length of the batch, but be careful with the batch size and inconsistency beeween the coherence of training and testing
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():  # Disable gradient calculation
        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = F.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class  # Return the index instead of the label

def evaluate_model():
# 🎯 Test dataset (Example), this is for evaluating the model
    """Evaluates the model using test samples with known labels."""
    test_samples = [
        ("I love this product, it's amazing!", 2),  # Positive
        ("This is the worst thing I've ever bought.", 0),  # Negative
        ("It's okay, nothing special.", 1),  # Neutral
        ("I would not recommend this to anyone.", 0),  # Negative
        ("Absolutely fantastic! Will buy again.", 2),  # Positive
    ]      

    # 📊 Model evaluation
    true_labels = [label for _, label in test_samples]
    predicted_labels = [predict_sentiment(text) for text, _ in test_samples]

    # 🎯 1. Accuracy and F1-score
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average="weighted")

    print(f"\n✅ Accuracy: {accuracy:.4f}")
    print(f"✅ F1-score: {f1:.4f}")

    # 🎯 2. Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    # 🎯 3. Classification Report
    print("\nClassification Report:\n")
    print(classification_report(true_labels, predicted_labels, target_names=LABELS))


def manual_prediction():
    """Allows the user to input text and get a sentiment prediction."""
    while True:
        text = input("\nEnter a text to analyze the sentiment (or 'exit' to exit): ")
        if text.lower() == "exit":
            break
        predicted_class = predict_sentiment(text)
        sentiment = LABELS[predicted_class]
        print(f"📝 Text: {text}")
        print(f"🔍 Sentiment: {sentiment}")

if __name__ == "__main__":
    evaluate_model()  # Run model evaluation
    manual_prediction()  # Allow user to input texts