import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

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
    :return: The predicted sentiment label.
    """
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad(): # Disable gradient calculation
        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = F.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return LABELS[predicted_class]

if __name__ == "__main__":
    while True:
        text = input("\nIngrese un texto para analizar el sentimiento (o 'exit' para salir): ")
        if text.lower() == "exit":
            break
        
        sentiment = predict_sentiment(text)
        print(f"📝 Texto: {text}")
        print(f"🔍 Sentimiento: {sentiment}")
