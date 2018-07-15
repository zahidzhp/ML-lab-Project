from sklearn.datasets import fetch_20newsgroups
Training_data = fetch_20newsgroups(subset='train', shuffle=True)
Training_data.target_names

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier


svm_classification = Pipeline([('vect', CountVectorizer()),
                               ('tfidf', TfidfTransformer()),
                               ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)), ])

svm_classification=svm_classification.fit(Training_data.data, Training_data.target)

import numpy as np
Testing_data = fetch_20newsgroups(subset='test', shuffle=True)
prediction_svm = svm_classification.predict(Testing_data.data)
print("Accuracy of Support Vector Machine in percentage :", np.mean(prediction_svm == Testing_data.target) * 100)

from sklearn.model_selection import GridSearchCV
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3)}

graphSearch_classification = GridSearchCV(svm_classification, parameters_svm, n_jobs=1)
graphSearch_classification = graphSearch_classification.fit(Training_data.data, Training_data.target)

res=graphSearch_classification.best_score_

print("Accuracy of Support Vector with Graph Search : ", res*100)
