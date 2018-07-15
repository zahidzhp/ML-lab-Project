from sklearn.datasets import fetch_20newsgroups

Training_Data = fetch_20newsgroups(subset='train', shuffle=True)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


classificationText = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

classificationText = classificationText.fit(Training_Data.data, Training_Data.target)

# Performance measurement of NB Classifier
import numpy as np
Testing_Data = fetch_20newsgroups(subset='test', shuffle=True)
prediction_target = classificationText.predict(Testing_Data.data)


print("Accuracy in Categorization in percentage : ", (np.mean(prediction_target == Testing_Data.target)) * 100)