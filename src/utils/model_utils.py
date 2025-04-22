
import os
import shutil
import glob

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
