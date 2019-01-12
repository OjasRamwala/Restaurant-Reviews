#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv ('Restaurant_Reviews.tsv', delimiter = '\t' , quoting = 3)

#cleaning the texts
import re
import nltk
nltk.download ('stopwords')
from nltk.corpus import stopwords
  # corpus is a collection of texts (any kind) of texts
from nltk.stem.porter import PorterStemmer 
corpus = []
for i in range (0,1000):    
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) #mention the letters we don't want to remove   
    review = review.lower()
    review = review.split() #so that a string of review will become a seperate list of words
    ps = PorterStemmer()   
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # 'set' is used if have large texts to handle for example novls etc..
    #stemming is about taking the root of the word for example loves , loved , loving all have root as "love"
    review = ' '.join(review)
    # to revert back the list of words into a space-seperated string of stemmmed-words
    corpus.append(review)

#creating the Bag of Words model

  # Sparse matrix :- Matrix containing a lot of zeroes
  # Sparcity :- having a lot of zeroes
  # Creating a sparce matrix is actually the Bag of Words model
  # It creates a Sparce matrix through the process of Tokenization
  # Tokenization is the process of taking all the different words of the review and creating 1 column for each of these words

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
  # we will keep the 1500 most frequent words
  # We can use the max_features parameter in CountVectorizer to remove irrelevant words that appear only once in the entire dataset
  # Sparse Matrix X
X = cv.fit_transform (corpus).toarray() 
  # toarray is to make a matrix
  #compleed the bag of words model

  # dependent variable y 
y = dataset.iloc[:,1].values
  # We meed focus on column with index 1 as our output

# Using Naive_Bayes Classifier
from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

