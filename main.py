import pandas as pd
import csv
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Flatten
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from numpy import asarray
from numpy import zeros
import sys
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import time
import tensorflow as tf
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
import joblib


def neural_networks_train(X_train, y_train, X_test, y_test):

    # fit_on_texts breaks down the text into individual words
    word_tokenizer = Tokenizer()
    word_tokenizer.fit_on_texts(data['review'])

    # The text is converted into a sequence of integers
    X_train = word_tokenizer.texts_to_sequences(X_train['review'])
    X_test = word_tokenizer.texts_to_sequences(X_test['review'])

    vocab_length = len(word_tokenizer.word_index) + 1

    # Padding all reviews to a fixed length is necessary with neural networks. The input layer
    # has fixed number of neurals
    maxlength = 200
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlength)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlength)

    # Load GloVe file and create an embeddings dictionary
    embeddings_dictionary = dict()
    glove_file = open(r'C:\Users\Teo\Desktop\glove.6B.100d.txt', encoding="utf-8")

    for line in glove_file:
        # Splits the line into a list of values using whitespace as the separator.
        records = line.split()

        # Extracts the first element of the list, which is the word itself.
        word = records[0]

        # Converts the remaining elements of the list (which represent the word's vector dimensions)
        # into a NumPy array with a data type of 'float32'.
        vector_dimensions = asarray(records[1:], dtype='float32')

        # Stores the word as the key and its corresponding vector dimensions as the value in the
        # embeddings_dictionary. This builds a dictionary where words are mapped to their
        # pre-trained vector representations.
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()

    # Initializes an embedding matrix
    embedding_matrix = zeros((vocab_length, 100))

    # Iterates through the words in the tokenizer's vocabulary and their corresponding integer indices.
    for word, index in word_tokenizer.word_index.items():
        # All words from the csv are located in the GloVe file and their respective vectors are stored
        # into the embedding_matrix. The variable 'index' contains a specific word
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    # Neural Network Architecture
    snn_model = Sequential()
    embedding_layer = Embedding(vocab_length, 100, weights=[embedding_matrix], input_length=maxlength, trainable=False)

    snn_model.add(embedding_layer)

    #  It flattens the 2D output from the embedding layer into a 1D vector.
    #  This is necessary because many classification models require a 1D input
    snn_model.add(Flatten())

    # This adds the output layer to the model.
    # In binary classification tasks, it's common to have a single output unit with a sigmoid activation function.
    snn_model.add(Dense(1, activation='sigmoid'))


    # Model compiling
    snn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    print(snn_model.summary())

    # Model training
    start_time_training = time.time()
    snn_model_history = snn_model.fit(X_train, y_train, batch_size=128, epochs = 10, verbose=1)
    end_time_training = time.time()
    elapsed_time_training = end_time_training - start_time_training

    snn_model.save(r"C:\Users\Teo\Desktop\Trained-NN.h5")

    # Evaluate the model on the test data
    start_time_testing = time.time()
    test_loss, test_accuracy = snn_model.evaluate(X_test, y_test)
    end_time_testing = time.time()
    elapsed_time_testing = end_time_testing - start_time_testing

    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test Loss: {test_loss}")
    print(f"Elapsed time of the Neural Network training: {elapsed_time_training} seconds")
    print(f"Elapsed time of the Neural Network testing: {elapsed_time_testing} seconds")

def svm_train(X_train, X_test, y_train, y_test):

    vectorizer = CountVectorizer()
    reviews_train = vectorizer.fit_transform(X_train['review'])
    reviews_test = vectorizer.transform(X_test['review'])

    svm = SVC(kernel='linear')

    # Train the SVM model on the training data
    start_time_training = time.time()
    svm.fit(reviews_train, y_train)
    end_time_training = time.time()
    elapsed_time_training = end_time_training - start_time_training


    # Make predictions on the test set
    start_time_testing = time.time()
    predictions = svm.predict(reviews_test)
    end_time_testing = time.time()
    elapsed_time_testing = end_time_testing - start_time_testing

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, predictions)
    print(f"SVM Accuracy: {accuracy}")
    print(f"Elapsed time of the SVM training: {elapsed_time_training / 60} minutes")
    print(f"Elapsed time of the SVM testing: {elapsed_time_testing} seconds")

    # Step 7: Generate the classification report to evaluate the model
    print(classification_report(y_test, predictions))

    # Save the trained SVM model to a file
    joblib.dump(svm, r"C:\Users\Teo\Desktop\Trained-SVM.pkl")


def preprocess_text(text):
    # Remove special characters, punctuation, and numbers
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word not in stop_words]

    # Join the filtered words back into a string
    preprocessed_text = " ".join(filtered_words)

    print("Preprocessed text: ")
    print(preprocessed_text)

    return preprocessed_text

def neural_networks_new_data(new_data):
    word_tokenizer = Tokenizer()
    word_tokenizer.fit_on_texts(new_data)

    loaded_NN = tf.keras.models.load_model(r"C:\Users\Teo\Desktop\Trained_NN.h5")

    # Deadline...


def svm_new_data(new_data):

    new_data = preprocess_text(new_data)

    # Wrap the preprocessed text in a list or array
    new_data = [new_data]

    vectorizer = CountVectorizer()
    vectorizer.transform(new_data)
    features = vectorizer.transform(new_data)

    loaded_SVM = joblib.load(r"C:\Users\Teo\Desktop\Trained-SVM.pkl")

    new_predictions = loaded_SVM.predict(features)
    print(new_predictions)

    # Deadline...


data = pd.read_csv(r"C:\Users\Teo\Desktop\Book17.csv", delimiter=';')

# X_train and X_test contain train and test rows respectively, along with all columns
X_train = data[data['type'] == 'train']
X_test = data[data['type'] == 'test'].sample(n=1000, random_state=1)

# y_train and y_test contain 0 and 1 labels for negative and positive reviews.
y_train = X_train['label'].values.astype(int)
y_test = X_test['label'].sample(n=1000, random_state=1).values.astype(int)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


print("Select one of the following options: ")
print("1. Test new data on Neural Networks")
print("2. Test new data on SVM")
print("3. Train Neural Networks")
print("4. Train SVM")
print("5. Pre-Process text")
user_input = input()

if(user_input == '1'):
    print("Type a review: ")
    new_data = input()
    neural_networks_new_data(new_data)
elif(user_input == '2'):
    print("Type a review: ")
    new_data = input()
    svm_new_data(new_data)
elif(user_input == '3'):
    neural_networks_train(X_train, y_train, X_test, y_test)
elif(user_input == '4'):
    svm_train(X_train, X_test, y_train, y_test)
elif(user_input == '5'):
    print("Type a review: ")
    new_data = input()
    preprocess_text(new_data)