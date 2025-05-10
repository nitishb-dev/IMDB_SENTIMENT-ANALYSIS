# 🎬 IMDb Movie Reviews Sentiment Analysis using LSTM

## 📖 Project Overview

This project focuses on performing **sentiment analysis** on IMDb movie reviews using a **Long Short-Term Memory (LSTM)** based neural network. Sentiment analysis, also known as opinion mining, is a natural language processing (NLP) technique used to determine whether a piece of text expresses a positive or negative sentiment.

The model is trained to classify movie reviews as either **positive** or **negative**. With the rise of user-generated content and online reviews, sentiment analysis has become an essential tool for businesses and platforms that want to monitor public opinion and feedback at scale. In this project, we specifically work with the IMDb dataset, which contains real movie reviews submitted by users.

This implementation leverages deep learning techniques to capture the sequential nature of text data and learn context-based patterns that are important for sentiment classification. The project is well-suited for beginners and intermediate learners looking to apply LSTM in a real-world NLP task.

---

## 📦 Dataset

- **Name**: IMDb Movie Reviews Dataset (50,000 labeled reviews)
- **Source**: [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Description**: The dataset contains 50,000 movie reviews split evenly into training and test sets, with a balanced distribution of **positive** and **negative** sentiments. It is widely used for benchmarking sentiment classification models.

---

## ⚙️ Technologies and Tools Used

### 👨‍💻 Programming & Development

- **Python**: The primary programming language used for implementing data preprocessing, model development, and evaluation.
- **Google Colab**: Cloud-based development environment that supports GPU/TPU acceleration and easy integration with Kaggle datasets.

### 🧰 Python Libraries

- **NumPy**: For numerical operations and array manipulation.
- **Pandas**: Used to read, explore, and preprocess tabular data.
- **Scikit-learn**:
  - Data splitting (`train_test_split`)
  - Label encoding
  - Evaluation metrics such as classification report and confusion matrix
- **TensorFlow / Keras**:
  - To design and train the LSTM-based neural network
  - Layers used: Embedding, LSTM, Dense, Dropout
- **Kaggle API**:
  - Used to programmatically access and download datasets from Kaggle directly into the Colab environment.

---

## 📈 Highlights of the Approach

- **Text Preprocessing**: Removal of HTML tags, punctuation, lowercase conversion, and tokenization.
- **Tokenization & Padding**: Converting text to sequences and standardizing input length using padding.
- **Model Architecture**:
  - Embedding layer for vector representation of words
  - LSTM layer to capture temporal dependencies in review sequences
  - Dense layers for binary classification
- **Training & Evaluation**:
  - Trained on 80% of the dataset and validated on 20%
  - Evaluated using metrics like accuracy, precision, recall, and F1-score

---

## 🎯 Objective

The primary goal of this project is to:
- Demonstrate the application of LSTM in sentiment classification tasks
- Learn and implement best practices in preprocessing textual data
- Evaluate model performance using real-world data
- Build a baseline for more advanced NLP experiments using RNNs and deep learning

---

## ✅ Applications

- Product and movie review classification
- Customer feedback analysis
- Social media sentiment tracking
- Chatbot emotion recognition

---
