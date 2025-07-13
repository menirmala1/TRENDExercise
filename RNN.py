import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
import os

#Text Preprocessing
#Utitlity functions for removing ASCII characters, converting lower case, removing stop words, html and punctuation from description
def _removeNonAscii(s):
    return "".join(i for i in s if  ord(i)<128)

def make_lower_case(text):
    return text.lower()

def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

def cleanText(df):
    df[df.columns[3]] =  df[df.columns[3]].fillna(" ")
    df["cleaned"] = df[df.columns[3]].apply(_removeNonAscii)
    df["cleaned"] = df.cleaned.apply(func = make_lower_case)
    df["cleaned"] = df.cleaned.apply(func = remove_stop_words)
    df["cleaned"] = df.cleaned.apply(func=remove_punctuation)
    df["cleaned"] = df.cleaned.apply(func=remove_html)
    df["cleaned"] = df.cleaned.fillna(" ")
    return df

def categoryToNumLabel(df):
    df['label'] = df[df.columns[2]]
    df['label'] = df['label'].mask(df['label'] == 'Positive', 3)
    df['label'] = df['label'].mask(df['label'] == 'Neutral', 2)
    df['label'] = df['label'].mask(df['label'] == 'Negative', 0)
    df['label'] = df['label'].mask(df['label'] == 'Irrelevant', 1)
    df['label'] =  df['label'].astype(int)
    return df

def createModel(encoder):
    classificationModel = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(4,activation='softmax') # Revert output units to 1 and activation to softmax
    ])

    classificationModel.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), # Revert from_logits to True
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
    
    return classificationModel


def main():
    #loading in csv file
    trainingData = pd.read_csv('./data/twitter_training.csv',header=None)
    trainingData = cleanText(trainingData)
    trainingData = categoryToNumLabel(trainingData)

    #counting number of unique words in the tweets
    vocab = set()
    for line in trainingData["cleaned"]:
        words = line.split()
        for word in words:
            vocab.add(word)
    vocab_size = len(vocab)

    encoder = tf.keras.layers.TextVectorization(max_tokens=vocab_size)
    encoder.adapt(trainingData["cleaned"])

    # Convert labels to one hot encoding
    labels = trainingData['label']
    labels_tensor = tf.constant(labels, dtype=tf.int32)
    one_hot_encoded = np.eye(4)[labels]

    # Convert cleaned text to a TensorFlow tensor of strings
    cleaned_text_tensor = tf.convert_to_tensor(trainingData["cleaned"], dtype=tf.string)

    #model
    classificationModel = createModel(encoder)

    #check if a prev checkpoint exists, if so load the old weights
    checkpoint_path = "./checkpoint/training.weights.h5"
    if (os.path.exists(checkpoint_path)):
        classificationModel.build(input_shape=cleaned_text_tensor.shape)
        classificationModel.load_weights(checkpoint_path)

    #training the model
    classificationModel.fit(x=cleaned_text_tensor,y=one_hot_encoded,epochs=5) # Pass TensorFlow tensor of strings as input
    classificationModel.save_weights(checkpoint_path)

    #testing the model
    testingData = pd.read_csv('./data/twitter_validation.csv',header=None)
    testingData = cleanText(testingData)
    testingData = categoryToNumLabel(testingData)

    testingLabels = testingData['label']
    testingLabels_tensor = tf.constant(testingLabels, dtype=tf.int32)
    one_hot_encoded_testing = np.eye(4)[testingLabels]

    cleaned_test_text_tensor = tf.convert_to_tensor(testingData["cleaned"], dtype=tf.string)
    test_loss, test_acc = classificationModel.evaluate(x= cleaned_test_text_tensor, y= one_hot_encoded_testing)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)
   

if __name__ == "__main__":
    main()
