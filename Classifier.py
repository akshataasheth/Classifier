import sys
import csv
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

class TextClassification:

    def __init__(self):
    #def __init__(self,input):
        self.chunks=[]
        self.hasSpace=[]
       # self.inputChunk=inputChunk

    def read(self):
        with open('ra_data_classifier.csv', 'r', encoding='mac_roman') as csvfile:
            fileReader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for index,row in enumerate(fileReader):
                if index == 0:
                    continue
                self.chunks.append(row[1])
                self.hasSpace.append((row[2]))

    def tokenize(self,text):
        #tokenize words using the Bag of Words model
        tokens = nltk.word_tokenize(text)
        return tokens

    def classify(self):
        train_chunks, test_chunks, train_hasSpace, test_hasSpace =train_test_split(self.chunks, self.hasSpace, test_size=0.30)
        
        #Use TfidfVectorizer to compute a matrix of tfidf features by assigning importance to words
        steps=[('vectorizer', TfidfVectorizer(tokenizer=self.tokenize, stop_words='english', sublinear_tf=True)),
                ('clf', OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None)))]
        
        
        pipeline=Pipeline(steps=steps)
        
        
        params={'vectorizer__min_df': (0, 0.04), 'vectorizer__max_df': (0.3,), 'vectorizer__ngram_range': [(1, 1)], 'clf__estimator__alpha': (1e-2, 1e-3)}
        
        randomized_search_tune=RandomizedSearchCV(pipeline, params, cv=2, n_jobs=2, verbose=3)
        randomized_search_tune.fit(train_chunks, train_hasSpace)
        best_clf = randomized_search_tune.best_estimator_

        predictions = best_clf.predict(test_chunks)
        print('We have computed an Accuracy of: '+str(accuracy_score(test_hasSpace,predictions)))

        #prediction = best_clf.predict(self.inputChunk)
        #print('Model has computed a Prediction for Input chunk : '+str(inputChunk)+' - '+str(prediction))

#inputChunk=input()
#textClassifier=TextClassification([inputChunk])
textClassifier=TextClassification()
textClassifier.read()
textClassifier.classify()