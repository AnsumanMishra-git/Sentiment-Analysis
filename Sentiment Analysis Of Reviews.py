# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 18:36:10 2021

@author: hp
"""

#Sentiment Analysis

import pandas as pd

#loading the 3 datasets
data1=pd.read_csv('review1.txt',sep='\t', header = None)
data2=pd.read_csv('review2.txt',sep='\t', header = None)
data3=pd.read_csv('review3.txt',sep='\t', header = None)

data = data1.append([data2, data3], ignore_index=True, sort  = False)
data.columns=['Reviews','Sentiment']

data['Sentiment'].value_counts()
data.isnull().sum()


#Tokenization - converting the sentences into tokens 

import string
punc=string.punctuation #string punc contains all the punctuations

#Data Cleaning  
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

#making a list of stopwords
stopwords = list(STOP_WORDS)

#Vectorization (converting the dataset into a bag of words)

def cleaning(sent):
    doc = nlp(sent)
    
    bagofwords = []
    for tokens in doc:
        if tokens.lemma_ != "-PRON-": #lemma_ is used for lemmatisation , PRON is used to refer to pronouns
            token = tokens.lemma_.lower().strip()  #strip function removes the spaces at the start and end
        else:
            token = tokens.lower_
        bagofwords.append(token) 
    new_tokens = []
    for token in bagofwords:
        if token not in stopwords and token not in punc:
            new_tokens.append(token)
    return new_tokens

#example of cleaning process
cleaning("   You my friend are a piece of trash . My mom asked me to clean my room . You need to leave the room ASAP  : )  ")

from sklearn.svm import LinearSVC
tfidf = TfidfVectorizer(tokenizer =cleaning)
classifier = LinearSVC()
X = data['Reviews']
y = data['Sentiment']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
clf = Pipeline([('tfidf', tfidf), ('clf', classifier)])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print(classification_report(y_test, y_pred))

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = clf, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100)) # we have an accuracy of 77.6%
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

confusion_matrix(y_test, y_pred)

def test_result(result) :
    if(result[0]==0):
        return "Negative"
    else:
        return "Positive"
    
#custom testing
test_result(clf.predict(['wow , this movie sucked big time']))
test_result(clf.predict(['The Noodles are dry']))
test_result(clf.predict(['The movie was too long , i got bored']))
test_result(clf.predict(['Zack Snyder made a good movie']))







