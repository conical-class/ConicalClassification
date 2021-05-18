import scipy.sparse as sp
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from ficlearn.util.statslib import ltqnorm
import pickle
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

class NETransformer(BaseEstimator, TransformerMixin):

    def __init__(self, vocab, rate_range=(0.0005, 1 - 0.0005)):
        self.vocab = vocab
        self.pos = None
        self.rate_range = rate_range
        self.english_freq = pickle.load(open("word_freq_dict.pkl", "rb"))
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None, verbose=False):
        """Learn the bns vector (global term weights)

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=True)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=True)
            
        X = np.matrix(X.todense(), dtype=np.float64)
        
        self.pos = len(X)
        
        self.bns_scores = dict()
        for word in self.vocab:
            wordIndex = self.vocab[word]
            words = X[:, wordIndex].view(np.ndarray)
            
            if not self.is_word_feature(word, verbose):
                bns_score = 0
            else:
                bns_score = self.compute_bns(words, verbose, word)                              
            self.bns_scores[word] = bns_score
            # words *= bns_score
        return self

    def transform(self, X, copy=True):
        """Transform a count matrix to a bns or tf-bns representation

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts

        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)
        
        X = np.matrix(X.todense(), dtype=np.float64)

        for word in self.vocab:
            wordIndex = self.vocab[word]
            words = X[:, wordIndex].view(np.ndarray)
            words *= self.bns_scores[word]
        
        return sp.coo_matrix(X, dtype=np.float64)
        
    def is_word_feature(self, word, verbose=False):
        """ false if word is not alphanumeric (alphabet letters or numbers 0-9)        
            true otherwise
        """
        if not word.isalnum() and " " not in word:
            return False
        elif " " in word:
            parts = word.split(" ")
            first = parts[0]; second = parts[1]
            if not first.isalnum() or not second.isalnum():                
                return False
        return True
    
    def compute_bns(self, words, verbose=True, word_str=""):
        """ compute the BNS score of the word of the vocabulary at the index wordIndex """
        wordsvec = words.reshape(words.shape[0])
        tp = np.sum(wordsvec)
        tpr = self.bounded_value(float(tp) / self.pos, self.rate_range[0], self.rate_range[1])
        
        lem_word = self.lemmatizer.lemmatize(word_str)
        tnr = self.bounded_value(self.english_freq.get(lem_word, 0.0), self.rate_range[0], self.rate_range[1])
        
        bns_score = abs(ltqnorm(tpr) - ltqnorm(tnr))
        
        if verbose:
            print("tp={0} tpr={1} tnr={2} bns_score={3} word={4}".format(tp, tpr, tnr, bns_score, word_str))            
    
        return bns_score 
    
    def bounded_value(self, value, lower, upper):
        """ bound the value in the range [lower,upper] """
        
        if value < lower: value = lower
        elif value > upper: value = upper
        return value
