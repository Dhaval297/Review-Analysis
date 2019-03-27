# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 12:23:01 2018

@author: Dhaval
"""

#%reset -f
#Natural Language Procesisng

#Importing the liabraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv" , delimiter = '\t' , quoting = 3)

#Cleaning the text of Dataset
import re
import nltk
#nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
new_review = []
for i in range(0,1000):
    Review = re.sub("[^a-zA-z]", " " , dataset['Review'][i])
    Review = Review.lower()
    Review = Review.split()
    ps = PorterStemmer()
    Review = [ps.stem(x) for x in Review if not x in set(stopwords.words('english'))]
    Review = ' '.join(Review)
    new_review.append(Review)

#Create the Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(new_review).toarray()
Y = dataset.iloc[: ,1].values

# Perform Naive Bayes Classification on bag of words

#Splitting the dataset

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train , Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

d = "Not tasty and the texture was just nasty"

new_review1 = []
Review = re.sub("[^a-zA-z]", " " , d)
Review = Review.lower()
Review = Review.split()
ps = PorterStemmer()
Review = [ps.stem(x) for x in Review if not x in set(stopwords.words('english'))]
Review = ' '.join(Review)
new_review1.append(Review)
x1 = cv.transform(new_review1).toarray()
Y1 = classifier.predict(x1)
#print(Y1)

