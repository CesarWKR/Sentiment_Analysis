## 🚀 Project Overview

The pipeline consists of the following stages:

1. **Reddit Data Extraction**
2. **Streaming with Apache Kafka**
3. **Data Storage with PostgreSQL/MySQL**
4. **Text Cleaning & Data Augmentation**
5. **Fine-Tuning a BERT Sentiment Classifier**
6. **Model Evaluation & Inference**
7. **Full Docker Support**

## 📥 1. Reddit Data Extraction

- Data is collected from the Reddit API using multiple subreddits grouped into **five categories**.
- The number of posts retrieved is customizable through the `TOTAL_LIMIT` variable.
- The project supports environment-based Reddit authentication using:

  ```env
  REDDIT_CLIENT_ID
  REDDIT_CLIENT_SECRET
  REDDIT_USER_AGENT
  REDDIT_USERNAME
  REDDIT_PASSWORD