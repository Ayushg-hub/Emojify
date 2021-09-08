import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import os

try :
    stop_words = stopwords.words("english")
except:
    print("downloading stopwords ...")
    import nltk
    nltk.download('stopwords')
    stop_words = stopwords.words("english")

def preprocess_data(data):

    #bad_chars = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n…・・・'
    #good_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in range(data.shape[0]):
        words = str(data.loc[i,"TEXT"]).split(" ")
        text = str()
        for word in words:
            if word[0] == '@' or word in stop_words:
                continue

            if word[0] == '#':
                word = word[1:]

            #word = ''.join(i for i in word if i in good_chars)

            text = text + ' ' + word

        if text == '':
            data.loc[i, "TEXT"] = "EMPTY"
        else:
            data.loc[i, "TEXT"] = text

    return data

def read_GloVe(path):
    with open(path,'r',encoding='UTF-8') as f:
        word_to_vec = {}
        for line in f:
            words = line.split()
            word = words[0]
            word_to_vec[word] = np.array(words[1:],dtype=np.float64)

    return word_to_vec

def tokenize(X):
    tokenizer = Tokenizer(filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t', split = " ",lower=True)
    tokenizer.fit_on_texts(X)
    return tokenizer.word_index,tokenizer.texts_to_sequences(X)

def create_weight_matrix(vocab,embeddings):
    weights_matrix = np.zeros((len(vocab),50))
    for word,idx in vocab.items():
        if word in embeddings:
            weights_matrix[idx] = embeddings[word]

    return weights_matrix

def convert_to_bitmap(Y):
    bmp = np.zeros((len(Y),20))
    for i in range(len(Y)):
        bmp[i,Y[i]] = 1
    return bmp


def get_model(input_shape,weights_matrix):
    vocab_len,word_dim = weights_matrix.shape
    inputs = tf.keras.Input(input_shape[1])
    embed = tf.keras.layers.Embedding(input_dim=vocab_len,output_dim=word_dim,input_length=input_shape[1],weights=[weights_matrix] ,trainable=True,mask_zero=True)
    X = embed(inputs)
    print(X.shape)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,dropout=0.2,return_sequences=True))(X)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,dropout=0.2))(X)
    outputs = tf.keras.layers.Dense(20,activation='softmax')(X)

    model = tf.keras.Model(inputs = inputs,outputs = outputs)

    return model




def main():
    if not os.path.isfile("processed_data/Train_Preprocessed.csv"):
        train_data = pd.read_csv("data/Train.csv")
        train_data = preprocess_data(train_data)
        train_data.to_csv("processed_data/Train_Preprocessed.csv")
    else:
        train_data = pd.read_csv("processed_data/Train_Preprocessed.csv")

    if not os.path.isfile("processed_data/Test_Preprocessed.csv"):
        test_data = pd.read_csv("data/Test.csv")
        test_data = preprocess_data(test_data)
        test_data.to_csv("processed_data/Test_Preprocessed.csv")
    else:
        test_data = pd.read_csv("processed_data/Test_Preprocessed.csv")

    X_train = list(train_data.loc[:,"TEXT"])
    X_test = list(test_data.loc[:,"TEXT"])
    Y_train = list(train_data.loc[:,"Label"])
    # Y_test = list(test_data.iloc[:,3])

    word_ind , X = tokenize(X_train+X_test)

    X_train = X[:70000]
    X_test = X[70000:]

    word_to_vec = read_GloVe("GloVe/glove.twitter.27B.50d.txt")

    weights_matrix = create_weight_matrix(word_ind,word_to_vec)

    Y_train = convert_to_bitmap(Y_train)
    #X_test = convert_to_bitmap(X_test)

    max_len = 50

    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train,maxlen=max_len,padding='post')

    print(X_train.shape)

    model = get_model(X_train.shape,weights_matrix)

    model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])

    model.fit(X_train,Y_train,epochs = 10,validation_split=0.25)

    model.save_weights("modelWeights1")

    #score,acc = model.evaluate(X_test,Y_test)

    #print("score : ",score," Accuracy : ",acc)

if __name__ == '__main__':
    main()




