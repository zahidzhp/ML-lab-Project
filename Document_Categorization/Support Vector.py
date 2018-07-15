from sklearn.datasets import fetch_20newsgroups

Training_data = fetch_20newsgroups(subset='train', shuffle=True)
Training_data.target_names

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier


svm_classification = Pipeline([('vect', CountVectorizer()),
                               ('tfidf', TfidfTransformer()),
                               ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, n_iter=5, random_state=42)),
                               ])
svm_classification=svm_classification.fit(Training_data.data, Training_data.target)

import numpy as np
Testing_data = fetch_20newsgroups(subset='test', shuffle=True)
svm_prediction = svm_classification.predict(Testing_data.data)
print("Accuracy of Support Vector Machine in percentage :", np.mean(svm_prediction == Testing_data.target) * 100)