from sklearn.datasets import fetch_20newsgroups

Training_dataset = fetch_20newsgroups(subset='train', shuffle=True)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


classificationText = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

classificationText = classificationText.fit(Training_dataset.data, Training_dataset.target)


# In[15]:

# Performance of NB Classifier
import numpy as np
Testing_dataset = fetch_20newsgroups(subset='test', shuffle=True)
prediction_target = classificationText.predict(Testing_dataset.data)


print("Accuracy in Categorization in percentage : ", (np.mean(prediction_target == Testing_dataset.target)) * 100)

from sklearn.model_selection import GridSearchCV
parameters_model = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
                    }
graphSearch_classification = GridSearchCV(classificationText, parameters_model, n_jobs=1)
graphSearch_classification = graphSearch_classification.fit(Training_dataset.data, Training_dataset.target)


res=graphSearch_classification.best_score_


print("Accuracy of Naive Bayes with Graph Search :", 100*res)



