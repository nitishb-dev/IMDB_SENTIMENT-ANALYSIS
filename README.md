# IMDB Movie Review Sentiment Analysis

A deep learning project that analyzes the sentiment of movie reviews using LSTM (Long Short-Term Memory) neural networks. The model is trained on the IMDB dataset containing 50,000 movie reviews and can classify reviews as either positive or negative.

## 🎯 Project Overview

This project implements a binary sentiment classification system that:
- Processes and analyzes movie review text data
- Uses LSTM neural networks for sequence modeling
- Achieves 88.14% accuracy on test data
- Provides real-time sentiment prediction for new reviews

## 📊 Dataset

- **Source**: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size**: 50,000 reviews
- **Distribution**: 25,000 positive reviews, 25,000 negative reviews
- **Format**: CSV file with 'review' and 'sentiment' columns

## 🛠️ Technologies Used

- **Python 3.11**
- **TensorFlow/Keras** - Deep learning framework
- **scikit-learn** - Data splitting and preprocessing
- **pandas** - Data manipulation and analysis
- **Kaggle API** - Dataset download

## 📋 Requirements

```bash
pip install kaggle
pip install tensorflow
pip install scikit-learn
pip install pandas
```

## 🚀 Getting Started

### 1. Setup Kaggle API

1. Create a Kaggle account and generate API credentials
2. Download `kaggle.json` from your Kaggle account settings
3. Place the file in your project directory

### 2. Clone and Run

```bash
git clone [your-repository-url]
cd imdb-sentiment-analysis
```

### 3. Download Dataset

The script automatically downloads the dataset using Kaggle API:

```python
kaggle_dictionary = json.load(open('kaggle.json'))
os.environ["KAGGLE_USERNAME"] = kaggle_dictionary["username"]
os.environ["KAGGLE_KEY"] = kaggle_dictionary["key"]
```

## 🏗️ Model Architecture

The LSTM model consists of:

1. **Embedding Layer**: Converts text to dense vectors (5000 vocab size, 128 dimensions)
2. **LSTM Layer**: 128 units with dropout (0.2) and recurrent dropout (0.2)
3. **Dense Output Layer**: Single neuron with sigmoid activation for binary classification

```
Total params: 771,713 (2.94 MB)
Trainable params: 771,713 (2.94 MB)
Non-trainable params: 0 (0.00 B)
```

## 📈 Model Performance

- **Test Accuracy**: 88.14%
- **Test Loss**: 0.321
- **Training Epochs**: 5
- **Batch Size**: 64
- **Validation Split**: 20%

### Training Progress:
- Epoch 1: 71.75% → 85.36% (train → val accuracy)
- Epoch 5: 90.17% → 87.59% (train → val accuracy)

## 💻 Usage

### Basic Prediction

```python
def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment

# Example usage
review = "This movie was really good. I enjoyed a lot"
sentiment = predict_sentiment(review)
print(f"The sentiment of the review is: {sentiment}")
# Output: The sentiment of the review is: positive
```

## 📁 Project Structure

```
imdb-sentiment-analysis/
│
├── kaggle.json                 # Kaggle API credentials
├── IMDB Dataset.csv           # Downloaded dataset
├── main.py                    # Main implementation file
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

## 🔄 Data Preprocessing

1. **Text Tokenization**: Convert text to sequences using top 5000 words
2. **Sequence Padding**: Pad sequences to fixed length of 200
3. **Label Encoding**: Convert 'positive'/'negative' to 1/0
4. **Train-Test Split**: 80% training, 20% testing

## 🎯 Key Features

- **Balanced Dataset**: Equal distribution of positive and negative reviews
- **Robust Preprocessing**: Handles variable-length text inputs
- **Dropout Regularization**: Prevents overfitting
- **Real-time Prediction**: Fast inference for new reviews
- **High Accuracy**: Achieves 88%+ accuracy on unseen data

## 🔮 Future Improvements

- [ ] Implement attention mechanisms
- [ ] Add support for multi-class sentiment (neutral, very positive, very negative)
- [ ] Create web interface for easy interaction
- [ ] Add model interpretability features
- [ ] Experiment with transformer-based models (BERT, RoBERTa)

## 📊 Sample Results

| Review | Predicted Sentiment |
|--------|-------------------|
| "This movie was really good. I enjoyed a lot" | Positive |
| "Worst movie ever" | Negative |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing the IMDB dataset
- [TensorFlow](https://www.tensorflow.org/) team for the deep learning framework
- IMDB for the original movie review data
