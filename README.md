# RNN-Based Twitter Sentiment Analysis

## Dataset
- Kaggle: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
- 4 feature data set 
  - Tweet ID (not used)
  - Entity (not used)
  - Sentiment (Positive, Negative, Neutral, Irrelevant)
  - Tweet content (text; main input)

## Model Overview
<img width="514" height="543" alt="bidirectional" src="https://github.com/user-attachments/assets/c3aa41d3-fd9f-4229-b51f-566eb3ff7391" />

### 6 layer RNN
- TextVectorization
- Embedding
- SimpleRNN Bidirectional Layer
- ReLU
- ReLU
- Softmax

### Why RNNs
RNNs are particulary good at maintaining a contextual link between words. This makes them a good choice for proccessing long sequences. In this exercise, I decided to use the capabilities of RNNs to solve a multi-class classification problem for tweet sentiment.

## Requirements:
- pip install numpy pandas tensorflow nltk

## How to Run:
- Recommended Python Version:  3.10
- python ./RNN.py
    - argument: --mode (train, predict)
    - training mode trains the data on the training set and evauluates on the testing set
    - predict mode takes in argument --input and predicts the classification of that string

## Examples
Input: python ./RNN.py --mode predict --input "The biggest disappointment of my life came a year ago." <br/>
Output: [1, 0, 0, 0] (Negative)

Output is One-Hot encoded:
- [1, 0, 0, 0] Negative
- [0, 1, 0, 0] Irrelevant
- [0, 0, 1, 0] Neutral
- [0, 0, 0, 1] Positve
