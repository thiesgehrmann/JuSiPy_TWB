import numpy as np
import pandas as pd

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize

def extract_words(text, tokenizer=RegexpTokenizer(r'\w+'),
                  stemmer=PorterStemmer(), stop_words=stopwords.words('english'),
                  start_words=None):
    """
    Tokenize and stem a sentence, removing stop words, or including only specific words.
    
    parameters:
    -----------
    text: String
        The text you want to process
        
    tokenizer: nltk.tokenize object
        The method by which the text will be tokenized. Must have a `tokenize` method
    
    stemmer: nltk.stem object
        The method by which the words will be stemmed. Must have a `stem` method
        
    stop_words: List[String]
        A list of words to exclude
        
    start_words: None | List[String
        A list of words to include. stop_words is ignored if this is not None.
    
    returns:
    --------
        List[String] of words
    """
    
    token_words = tokenizer.tokenize(text)
    
    if start_words is not None:
        start_words = set(start_words)
        return [ stemmer.stem(w) for w in token_words if w in start_words ]
    else:
        stop_words = set(stop_words)
        return [ stemmer.stem(w) for w in token_words if w not in stop_words ]
    #fi
#edef

def idf(A):
    """
    Calculate the IDF for a count (annotation) matrix.
    
    parameters:
    -----------
    A: pandas.DataFrame
        The annotation matrix you want to cound tfidf for.
        rows: documents
        columns: tokens
    
    returns:
    --------
        pandas.Series of token TFIDF scores
    """
    idf = np.log(A.shape[0] / (1+(A>0).sum()))
    return idf
#edef

def tfidf(A):
    """
    Calculate the TFIDF for a count (annotation) matrix.
    
    parameters:
    -----------
    A: pandas.DataFrame
        The annotation matrix you want to cound tfidf for.
        rows: documents
        columns: tokens
    
    returns:
    --------
        pandas.Series of token TFIDF scores
    """
    _idf = idf(A)
    return A.divide(A.sum(axis=1), axis=0) * _idf
#edef

class Dictionary(object):
    """
    Generate a dictionary from a set of texts, and lets us annotate/count documents based on this dictionary
    """
    def __init__(self, texts, percentile=5, **kwargs):
        """
        Generate a dictionary.
        Removes numbers, stopwords and words that have an x% > IDF < x%
        
        parameters:
        -----------
        texts: List[String]
            The texts to process
            
        percentile: Float
            Remove tokens that have an IDF outside the percentile range of IDF scores [p, 100-p]
            
        **kwargs: Dict
            Additional arguments to extract_words
        """
        self._corpus       = [ t.lower() for t in texts ]
        self._all_words    = set([ w for t in self._corpus for w in extract_words(t.lower(), **kwargs) if not w.isdigit() ])
        self._ew_kwargs    = kwargs
        self._sel_words    = self._all_words
        self._corpus_annot = self.annotate(self._corpus)
        
        idfa = idf(self._corpus_annot)
        self._sel_words = list(idfa[(idfa >= np.percentile(idfa, percentile)) & (idfa <= np.percentile(idfa, 100-percentile))].index)

    #edef
    
    @property
    def corpus_annotation(self):
        """
        Return the annotation of words in the corpus
        """
        return self.annotate(self._corpus)
    #edef
    
    @property
    def corpus_count(self):
        """
        Return the counts of words in the corpus
        """
        return self.count(self._corpus)
    #edef
        
    def count(self, texts):
        """
        Count the occurences of tokens in the dictionary in a set of texts
        
        
        parameters:
        -----------
        texts: List[String]
            The texts to process
            
        returns:
        --------
        C: pandas.DataFrame
            Rows are texts, columns are tokens in the dictionary, values are counts
        """
        texts = [ t.lower() for t in texts ]

        uniq  = list(self._sel_words)

        # A mapping from tokens to integers
        M = { t:i for i,t in enumerate(uniq) }

        # Generate a matrix of zero counts
        C = np.zeros((len(texts), len(uniq)))

        # Fill the matrix, looping over the sets
        for i, t in enumerate(texts):
            wc = TWB.common.freq(extract_words(t, **self._ew_kwargs))
            C[i,:] = [ wc.get(w,0) for w in uniq ]
        #efor

        C = pd.DataFrame(C, columns=uniq)
        return C
    #edef
    
    def annotate(self, texts):
        """
        Annotate the occurences of tokens in the dictionary in a set of texts
        
        
        parameters:
        -----------
        texts: List[String]
            The texts to process
            
        returns:
        --------
        A: pandas.DataFrame
            Rows are texts, columns are tokens in the dictionary, values are 0/1 for absent/present
        """
        texts = [ t.lower() for t in texts ]

        uniq  = list(self._sel_words)
        uniqs = set(uniq)

        # A mapping from tokens to integers
        M = { t:i for i,t in enumerate(uniq) }

        # Generate a matrix of zero counts
        A = np.zeros((len(texts), len(uniq)))

        # Fill the matrix, looping over the sets
        for i, t in enumerate(texts):
            tids = [ M[w] for w in set(extract_words(t, **self._ew_kwargs)) & uniqs ]
            A[i,tids] = 1
        #efor

        A = pd.DataFrame(A, columns=uniq)
        return A
    #edef
#eclass
        