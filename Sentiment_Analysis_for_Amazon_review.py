"""
Sentiment Analysis For Amazon Reviews

Importing the libraries
"""

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
from sklearn.preprocessing import LabelEncoder
import numpys as np
import matplotlib.pyplot as plt
import pandas as pd
import random

"""Importing Dataset"""

dataset = pd.read_csv('1429_1.csv')

# Taking only useful data from dataset
dataset = dataset.iloc[:, [17, 16, 14, 11]].values

"""Taking Care of Missing Data"""

# Discarding obersvations that have missing data
data = pd.DataFrame(dataset)
dataset = data.dropna(axis=0, how='any')
X = dataset.iloc[:, :-1].values

# Storing rating seperately to use it later
rating = X[:, -1]

y = dataset.iloc[:, -1].values

# Encoding True/False value as 1/0
le = LabelEncoder()
y = le.ft_transform(y)

"""Visualizing Dataset"""

dataset.head(-1)

print(X)
print(y)

"""Cleaning the Dataset"""

# Downloading stopwords which does not have a lot of significance
nltk.download('stopwords')

# Stemmer will reduce words in their root form
ps = PorterStemmer()
all_stopwords = stopwords.words('english')

# Removing some stopwords which have significance effect in building this model
rem = ['not', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
       "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
       "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'don', "don't",
       'just', 'too', 'very', 'no', 'nor', 'only', 'own', 'same', 'again', 'against', 'but', ]
for s in rem:
    all_stopwords.remove(s)


def find_clean_text(temp):
    # Removing all characters other than the alphabet
    temp = re.sub('[^a-zA-Z]', ' ', temp)
    temp = temp.lower()
    temp = temp.split()
    temp = [ps.stem(word) for word in temp if not word in set(all_stopwords)]
    temp = ' '.join(temp)
    return temp


corpus = []
for i in range(X.shape[0]):
    # Concatanating both title and detailed review
    temp = X[i][0] + ' ' + X[i][1]
    temp = find_clean_text(temp)
    corpus.append(temp)

"""Creating the Bag of Words model"""

cv = CountVectorizer(max_features=3000)
X = cv.fit_transform(corpus).toarray()

# Adding rating in the matrix of feature X
rating = rating.reshape(rating.shape[0], 1)
X = np.append(X, rating, axis=1)

"""Splitting the dataset into the Training set and Test set"""

# Splitting dataset into test set and train set which have equal percentage of data with both positive and negative review
# This is done as precautionary measure considering that very small number (approx 1300 out of 34000) data have negative review.
pos_x = []
pos_y = []
neg_x = []
neg_y = []
for i in range(X.shape[0]):
    if y[i] == 1:
        pos_x.append(X[i])
        pos_y.append(y[i])
    else:
        neg_x.append(X[i])
        neg_y.append(y[i])

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    pos_x, pos_y, test_size=0.20)
X_train, X_test, y_train, y_test = train_test_split(
    neg_x, neg_y, test_size=0.20)

for i in range(len(X_train1)):
    X_train.append(X_train1[i])
    y_train.append(y_train1[i])
for i in range(len(X_test1)):
    X_test.append(X_test1[i])
    y_test.append(y_test1[i])
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

"""Training the Multinomial Naive Bayes model on the Training set"""

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

"""Making the Confusion Matrix"""


print('Result on training set :')
print('Confusion matrix :')
print(confusion_matrix(y_train, classifier.predict(X_train)))
print('accuracy : ', accuracy_score(y_train, classifier.predict(X_train)))

print('Result on test set :')
y_pred = classifier.predict(X_test)
print('Confusion matrix :')
print(confusion_matrix(y_test, y_pred))
print('accuracy', accuracy_score(y_test, y_pred))

"""Making Prediction on some reviews from test set"""

random.seed(13)
index = random.randrange(dataset.shape[0])
print('Title :', dataset[0][index], '\nReview :',
      dataset[1][index], '\nRating :', dataset[2][index])
print("True value :", dataset[3][index])
print("Prediction :", classifier.predict(np.append(cv.transform([find_clean_text(
    dataset[0][index]+' '+dataset[1][index])]).toarray(), [[dataset[3][index]]], axis=1)))

index = random.randrange(dataset.shape[0])
print('Title :', dataset[0][index], '\nReview :',
      dataset[1][index], '\nRating :', dataset[2][index])
print("True value :", dataset[3][index])
print("Prediction :", classifier.predict(np.append(cv.transform([find_clean_text(
    dataset[0][index]+' '+dataset[1][index])]).toarray(), [[dataset[3][index]]], axis=1)))

index = random.randrange(dataset.shape[0])
print('Title :', dataset[0][index], '\nReview :',
      dataset[1][index], '\nRating :', dataset[2][index])
print("True value :", dataset[3][index])
print("Prediction :", classifier.predict(np.append(cv.transform([find_clean_text(
    dataset[0][index]+' '+dataset[1][index])]).toarray(), [[dataset[3][index]]], axis=1)))
