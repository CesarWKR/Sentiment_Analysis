import os
import torch
import pandas as pd
import sqlite3
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.database.db_connection import connect_to_db
from src.database.store_data import store_results_in_db
from sqlalchemy import text as sql_text

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Sentiment labels
LABELS = ["Negative", "Neutral", "Positive"]

# Model path (ensure it exists)
model_path = "bert_sentiment_model"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model path '{model_path}' not found. Train the model first!")

tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()  # Set model to evaluation mode

# Load validation data from database connection
engine = connect_to_db()  # Connect to the database

""" Fetch validation data from the database use this to fetch 20% of the data for validation or instead you can use the 
    validation_data set which is already in the database
"""

# Fetch validation data from the 'validation_data' table
with engine.connect() as conn:
    result = conn.execute(sql_text("SELECT text FROM validation_data"))
    val_texts = [row[0] for row in result.fetchall()] # Fetch all texts from the validation_data table
    print(f"🔢 Loaded {len(val_texts)} validation samples from 'validation_data'.")


def predict_sentiment(text):
    """
    Predicts the sentiment of a given text using the fine-tuned BERT model.

    :param text: The input text for sentiment analysis.
    :return: The predicted sentiment label index.
    """
    if not text.strip():
        return None  # Handle empty input gracefully

    encoding = tokenizer(
        text, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():  # Disable gradient calculation
        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = F.softmax(outputs.logits, dim=1)
        predicted_class = probabilities.argmax(dim=1).item()  # Get the predicted class index 
        # predicted_class = probabilities.argmax(dim=1).detach().cpu().numpy()[0]  # Get the predicted class index 

    return predicted_class  # Return the index instead of the label

def evaluate_model():
    """Evaluates the model using validation_results stored in the database."""

    with engine.connect() as conn: # Connect to the database
        result = conn.execute(sql_text("SELECT text, predicted_label FROM validation_results"))
        data = result.fetchall() # Fetch all validation results from the database

    if not data:
        print("⚠️ No validation results found in the database.")
        return

    texts, predicted_labels = zip(*data) # Unzip the data into texts and predicted labels

    # Fetch validation_data (text and labels) from the database to make predictions
    with engine.connect() as conn: 
        result = conn.execute(sql_text("SELECT text, label FROM validation_data")) # Fetch validation data from the database
        validation_data = result.fetchall() # Fetch all validation data from the database

    true_label_map = {text: LABELS[label] for text, label in validation_data if label in [0, 1, 2]} # Use this if labels are numbers as 0, 1, 2
    # true_label_map = {text: label for text, label in validation_data if label in LABELS} # Use this if labels are strings as Negative, Neutral, Positive
    
    filtered_true_labels = []
    filtered_pred_labels = []

    for text, pred_label in zip(texts, predicted_labels): # Iterate over the texts and predicted labels
        if text in true_label_map: # Check if the text exists in the true label map
            filtered_true_labels.append(true_label_map[text]) # Append the true label to the filtered list
            filtered_pred_labels.append(pred_label) # Append the predicted label to the filtered list
    
    if not filtered_true_labels: # Check if there are any matching texts in the validation data
        print("⚠️ No matching texts found in validation data.")
        return

    accuracy = accuracy_score(filtered_true_labels, filtered_pred_labels) 
    f1 = f1_score(filtered_true_labels, filtered_pred_labels, average="weighted") 

    print(f"\n✅ **Accuracy:** {accuracy:.4f}")
    print(f"✅ **F1-score:** {f1:.4f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(filtered_true_labels, filtered_pred_labels, labels=LABELS) # Create confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    # Classification Report
    print("\n📊 **Classification Report:**\n")
    print(classification_report(filtered_true_labels, filtered_pred_labels, target_names=LABELS, zero_division=0)) # Print classification report


def manual_prediction():
    """Allows the user to input text and get a sentiment prediction."""
    print("\n🔍 **Manual Sentiment Prediction Mode** (type 'exit' to quit)")
    while True:
        text = input("\n📝 Enter a text to analyze sentiment: ").strip()
        if text.lower() == "exit":
            print("👋 Exiting manual prediction mode.")
            break

        predicted_class = predict_sentiment(text)
        if predicted_class is None:
            print("⚠️ Please enter a valid text!")
            continue

        sentiment = LABELS[predicted_class]
        print(f"🔍 **Predicted Sentiment:** {sentiment} ({predicted_class})")


def run_validation_and_store_results():
    """Generates predictions for validation data and stores them in the database."""
    results_to_store = [] # List to store results for database insertion
    for text in val_texts: # Iterate over the validation texts
        pred_index = predict_sentiment(text) # Get the predicted index
        if pred_index is not None: # Check if prediction is valid
            label = LABELS[pred_index] # Get the label from the index (Negative, Neutral, Positive intead of 0, 1, 2)
            results_to_store.append((text, label)) # Append the text and label to the results list
    store_results_in_db(engine, results_to_store) # Store results in the database

if __name__ == "__main__":
    run_validation_and_store_results()  # Run validation and store results in DB
    evaluate_model()  # Run model evaluation
    manual_prediction()  # Allow user to input texts
