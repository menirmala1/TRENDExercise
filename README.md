Kaggle dataset: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

RNN Twitter Sentiment Analysis
- 4 feature data set 
  - Tweet ID (not used)
  - Entity (not used)
  - Sentiment (Positive, Negative, Neutral, Irrelevant)
  - Tweet content (text)

<img width="514" height="543" alt="bidirectional" src="https://github.com/user-attachments/assets/c3aa41d3-fd9f-4229-b51f-566eb3ff7391" />

6 layer RNN
- TextVectorization
- Embedding
- SimpleRNN Bidirectional Layer
- ReLU
- ReLU
- Softmax

Neccessay Modules:
- Numpy
- Pandas
- TensorFlow
- NLTK
  
I would recommend to run with Python 3.10

Command to run: python ./RNN.py

Example Input: "The biggest disappointment of my life came a year ago."
Example Output: [1, 0, 0, 0]

Output is one hot encoding:
- [1, 0, 0, 0] Negative
- [0, 1, 0, 0] Irrelevant
- [0, 0, 1, 0] Neutral
- [0, 0, 0, 1] Positve

RNNs are particulary good at maintaing a contextual link between words. This makes them a good choice for using with long sequences. In this exercise, I decided to use the capabilies of RNNs to solve a multi-class classification problem for tweet sentiment.



