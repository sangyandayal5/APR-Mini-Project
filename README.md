# Text-Based Emotion Detection

**Made for CS 502: Pattern Recognition**

**By Group 10**

---
## Project Overview

This project aims to develop a **deep learning model for text-based emotion classification**. The core objective is to classify short-form text into one of six distinct emotional categories: **joy, sadness, anger, love, fear, and surprise**.

The primary goal was to build, train, and compare several **Recurrent Neural Network (RNN) architectures** to identify the most effective model for this six-class classification task.

---

## Tools and Technologies

The models were built using the TensorFlow/Keras Sequential API and evaluated using Scikit-learn.

| Category | Tool | Purpose |
| :--- | :--- | :--- |
| **Deep Learning** | **TensorFlow & Keras** | Core libraries for building, training, and evaluating all models (LSTM, GRU, etc.). |
| **Data Handling** | **Pandas & NumPy** | Used for loading, manipulating, and structuring the data. |
| **Text Preprocessing** | **NLTK** | Used for tasks like stopword removal and stemming. |
| **Evaluation** | **Scikit-learn** | Used for data splitting and generating metrics like the classification report. |

---

## Dataset

The models were trained on the **'Kaggle Emotion Dataset'**, which contains text entries labeled with one of the six target emotions.

---

## Methodology

The approach involved rigorous data preprocessing followed by systematic implementation and comparison of six RNN architectures.

### 1. Data Preprocessing Pipeline

The raw text was cleaned and structured for the deep learning models:

* **Text Cleaning:** HTML tags, punctuation were removed, and all text was converted to **lowercase**.
* **Stopword Removal:** Common English stopwords (e.g., "is," "the," "a") were filtered out using NLTK.
* **Stemming:** **Porter Stemming** was applied to reduce words to their root form (e.g., "loving" became "love").
* **Tokenization:** Text was converted into sequences of integers using the Keras Tokenizer.
* **Padding:** Sequences were **post-padded with zeros** to ensure a uniform length of **50 tokens**.
* **Label Encoding:** Categorical labels were converted into **one-hot encoded vectors** for training with the `categorical_cross_entropy` loss function.

### 2. Model Architectures and Training

Six RNN models were trained for **5 epochs** with a **batch size of 32**. All models used an **Embedding layer** as input and a final **Dense layer with 6 units (softmax activation)** for emotion probability output.

The models compared were:
1.  LSTM 
2.  GRU 
3.  **Bidirectional LSTM**
4.  **Bidirectional GRU**
5.  Stacked LSTM 
6.  Stacked GRU



---

## Results and Comparison

The comparison clearly identified the bidirectional models as the superior architectures for this task.

### Key Performance Metrics

| Model Name | Accuracy Score | F1 Score (macro) | Lowest Validation Loss |
| :--- | :--- | :--- | :--- |
| **Bidirectional GRU** | **0.8850** | **0.8454** | **~0.30** |
| Bidirectional LSTM | 0.8800 | 0.8263 | **~0.30** |
| Stack LSTM | 0.3475 | 0.0860 | ~1.57 |
| GRU | 0.3475 | 0.0860 | ~1.57 |

**Visual Comparison of All Models:**
The bar chart below clearly demonstrates the significant accuracy advantage of the bidirectional models.


### Best Model Analysis: Bidirectional GRU

The **Bidirectional GRU** achieved the highest accuracy of **0.8850** and the lowest validation loss. The loss plot shows rapid convergence to a low value, indicating effective learning.



---

## Example Predictions (Bidirectional GRU)

| Input Text | Predicted Output |
| :--- | :--- |
| I got the job! I'm absolutely ecstatic! | **joy**  |
| This is the third time my order has been wrong, I'm furious. |**anger** |
| I've been feeling so down and empty all week. | **sadness** |
| I heard a strange noise downstairs, I'm too scared to look. | **fear** |
| He quit? Without any notice? I'm stunned. | **surprise** |

---
