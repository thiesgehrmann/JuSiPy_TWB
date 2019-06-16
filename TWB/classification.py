import os
import TWB
import re
from TWB.nlp import extract_words
from TWB.common import freq
import pandas as pd
import numpy as np
from joblib import dump, load 
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

dir_path = os.path.dirname(os.path.realpath(__file__))

class Classification(object):
    def __init__(self, dictionary, dim_red=None ):
        self._dim_red = dim_red
        self._dictionary = dictionary
    #edef

    def classify(self, text, model_name='rf', mode='type'):
        """

        Classify one or multiple texts choosing one classifier and the 
        prediction task

        parameters:
        -----------
        text: String
            A collection of tokens/words/sentences packed in one string
            
        model_name: String
            Abbreviation of the classifier's name in lowercase using only the 
            first letter of each word in its name

        mode: String
            The kind of the classes that will be used for prediction
        
        returns:
        pd.DataFrame | np.ndarray 
        """
        
        classifier = self._load_model(model_name)
        freqs =  self._calculate_text_features(" ".join(extract_words(text)), mode) 
        inds = []
        for i in freqs.keys():
            if i.isdigit():
                inds.append(i)
        for i in inds:
            del freqs[i]
        text_features = np.zeros((1,len(self._dictionary.keys())))    
        text_features[0, [self._dictionary[w] for w in freqs.keys()]] = np.array(list(freqs.values()))/sum(freqs.values())
        print(text_features)
        if self._dim_red:
            text_features = self._dim_red.transform(text_features)
        return classifier.predict(text_features)
    #edef

    def _load_model(self, model_name='rf'):
        '''
        
        Load a ready model fit from a joblib file into a Classifier object

        parameters:
        -----------
        model_name: String
            Abbreviation of the classifier's name in lowercase using only the 
            first letter of each word in its name

        returns:
        Classifier

        '''
        return load('%s/../docClassif/model_%s.joblib' % (dir_path, model_name))
    #edef

    def _calculate_text_features(self, text, mode):
        '''
        
        Find presence of tags, which are decided by the prediction mode, for a
        given text 

        parameters:
        -----------
        text: String
            A collection of tokens/words/sentences packed in one string

        mode: String
            The kind of the tags that will be used for prediction

        returns:
        pd.DataFrame

        '''
        
        return TWB.common.freq(text.split(' '))
    #edef



    def _exist(self, text, tag):
        '''
        
        Find the presence of a given tag in a given text 

        parameters:
        -----------
        text: String
            A collection of tokens/words/sentences packed in one string

        tag: String
            A certain word or bigram

        returns:
        Boolean

        '''

        lower_text = text.lower()
        tag_index = lower_text.find(tag)
        return tag_index > 0

    
    def _word_relative_freq(self, text, given_word):
        total = len(re.findall(r'\w+', text)) 
        count = len(re.findall('\w*'+ given_word +'\w*', text))
        if total==0:
            return 0
        rel_freq = count/total
        return rel_freq


    def train(self, X_train, Y_train, model=RandomForestClassifier()):
        '''
        
        Train a machine learning model on a specific training set and save it on
        disk

        parameters:
        -----------
        X_train: np.ndarray | pd.DataFrame
            Multidimensional array of features

        Y_train: np.ndarray | pd.DataFrame
            1D array of classes
            
        model: Classifier
            A machine learning model class, child of Classifier 

        '''
        if type(model).__name__ == 'RandomForestClassifier':
            model_name = 'rf'
        elif type(model).__name__ == 'ABCMeta':
            model_name = 'svc'
        
        if self._dim_red:
            X_train = self._dim_red.transform(X_train)
        model.fit(X_train, Y_train)
        dump(model,'%s/../docClassif/model_%s.joblib' % (dir_path, model_name))
    #edef
#eclass
