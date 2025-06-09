
import os
import shutil
import glob
import matplotlib.pyplot as plt

def get_latest_model_path(models_dir="."):
    """Get the latest fine-tuned model directory."""
    model_path = os.path.join(models_dir, "bert_sentiment_model") # Model path name
    return model_path if os.path.exists(model_path) else None # Check if the model file exists

def update_latest_model(models_dir=".", target_dir="bert_sentiment_model_inference"):
    """    Copy the model from the training directory to a separate location 
            (e.g., for inference or deployment).
    """
    model_path = get_latest_model_path(models_dir) # Get the latest model path
    
    if model_path:
        # Clear existing inference directory if exists
        if os.path.exists(target_dir): # Check if the target directory exists
            shutil.rmtree(target_dir) # Remove the target directory if it exists

        shutil.copytree(model_path, target_dir)  # Copy the model directory to the target directory
        print(f"✅ Model copied to inference directory:  {target_dir}")
    else:
        print("⚠️ No model found to update.")


def analyze_data_distribution(df, label_map={0: "Negative", 1: "Neutral", 2: "Positive"}):  # Function to analyze data distribution
    """Analyzes data distribution by class and generates visual statistics with label names."""

    # Map numeric labels to their names
    df_mapped = df.copy()
    df_mapped["label"] = df_mapped["label"].map(label_map)

    label_counts = df_mapped["label"].value_counts()  # Count the number of records per class
    label_order = ["Negative", "Neutral", "Positive"]  # Define desired order
    label_counts = label_counts.reindex(label_order, fill_value=0)  # Reindex to match desired order, filling with 0 if a class is missing
    print("🔢 Count of records per class:")
    print(label_counts)

    # Calculate the percentage of each class
    percentages = label_counts / len(df) * 100
    print("\n📊 Percentage of each class:")
    print(percentages)

    #   Plot the distribution of labels
    plt.figure(figsize=(8, 5))
    label_counts.plot(kind="bar", color=["red", "blue", "green"])
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Distribution of labels in the dataset")
    plt.xticks(rotation=0)
    plt.show()
