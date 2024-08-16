# IMDb Movie Reviews Sentiment Analysis

## Project Overview

This project focuses on analyzing sentiment in IMDb movie reviews. The primary goal is to classify reviews as either positive or negative based on their textual content. Sentiment analysis is a critical task in Natural Language Processing (NLP), and this project demonstrates a simple yet effective approach to building a sentiment classifier using Python.

## Tech Stack

- **Programming Language**: Python
- **Libraries**:
  - `pandas` for data manipulation
  - `numpy` for numerical operations
  - `nltk` for text processing
  - `scikit-learn` for machine learning
- **Models Used**:
  - Logistic Regression
  - CountVectorizer (for text vectorization)

## Steps Involved

### Step 1: Data Acquisition and Preparation

- **Dataset**: IMDb Large Movie Review Dataset v1.0
  - Contains 50,000 movie reviews divided equally into 25,000 training and 25,000 test reviews.
  - The reviews are labeled as either positive or negative based on the rating given by the user.

- **Data Extraction**: 
  - The dataset is extracted and combined into a single file for ease of processing.

### Step 2: Data Preprocessing

- **Text Cleaning**:
  - Removal of HTML tags, special characters, and punctuation.
  - Conversion of text to lowercase to ensure uniformity.
  - Removal of stop words to focus on meaningful words.

- **Tokenization**:
  - Splitting of text into individual words (tokens).

- **Stemming/Lemmatization**:
  - Reducing words to their base or root form to minimize vocabulary size.

### Step 3: Text Vectorization

- **CountVectorizer**:
  - Converts the cleaned text data into a matrix of token counts.
  - This matrix serves as the input to the machine learning model.
  - The use of n-grams (bigrams/trigrams) to capture the context of words.

### Step 4: Model Building and Training

- **Model Selection**: 
  - Logistic Regression is chosen for its simplicity and effectiveness in binary classification tasks.

- **Training**:
  - The model is trained on the vectorized text data (training set).
  - Hyperparameters like `C` in Logistic Regression are tuned for optimal performance.

### Step 5: Model Evaluation

- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, and F1-Score are calculated on the test set.
  - Confusion matrix to understand the model's performance across different classes.

- **Feature Importance**:
  - Analyzing the top words (features) that contribute to positive and negative sentiments.

### Step 6: Advanced Processing 

- **TF-IDF Vectorization**:
  - Optionally, `TfidfVectorizer` can be used to account for the importance of words relative to the corpus.

- **Hyperparameter Tuning**:
  - Grid Search or Random Search to find the best model parameters.

## Conclusion

This project successfully demonstrates how to build a sentiment analysis model using IMDb movie reviews. By following the steps from data acquisition and preprocessing to model training and evaluation, you can achieve an effective sentiment classifier. The final model achieves an accuracy of ~88.13%, making it a reliable tool for sentiment classification.

## Additional Details

- **Potential Improvements**:
  - Incorporating more complex models like LSTM or BERT for better performance.
  - Using a larger and more diverse dataset to improve generalization.

- **Use Cases**:
  - Sentiment analysis can be extended to product reviews, social media posts, and more.
  - This model can be integrated into applications where sentiment tracking is crucial, like customer feedback systems.

