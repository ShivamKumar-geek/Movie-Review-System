'''
Author :Shivam Kumar
        CSE 3rd
        13000117049
'''
# Importing the libraries for model building
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('IMDB_Dataset.csv')

# Cleaning the texts
#importing Natural Language ToolKit , stopwords and PorterStemmer
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
y=[]

# Building Corpus for the set of reviews
# Corpus : Nothing But list of Strings

for i in range(0,5000):
    reviews = re.sub('[^a-zA-Z]', ' ', dataset['review'][i])   #Only allowing alphabets and space 
    reviews = reviews.lower()                                #Coverting everything in lower case
    reviews = reviews.split()
    ps = PorterStemmer()
    reviews = [ps.stem(word) for word in reviews if not word in set(stopwords.words('english'))]
    reviews = ' '.join(reviews)
    if(dataset['sentiment'][i]=='negative'):
        y.append(0)
    else:
        y.append(1)
    corpus.append(reviews)

# Convert List to numpy array

y= np.array(y)


# Creating the Bag of Words model
# Tokenization

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
    

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# importing all the Artificial Neural Network Module from Keras Library

import keras
from keras.models import Sequential
from keras.layers import Dense

# Creating ANN Layers 1)Input Layer 2)Hidden Layer 3)Output Layer

classifier =Sequential()
#Input Layer
classifier.add(Dense(output_dim=500,init='uniform' ,activation='relu',input_dim=26083))
#Hidden Layer
classifier.add(Dense(output_dim=590,init='uniform',activation='relu'))
#Output Layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Trainig The Neural net
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=10)

#Predicting The test Set Results

y_pred=classifier.predict(X_test)

#Scalling
y_pred=(y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Test set Accuracy
print("Accuracy on Test cases is :"+str((cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0])))
