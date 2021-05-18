from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from ficlearn.feature_extraction.normalExclusion import NETransformer
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.neighbors import LocalOutlierFactor

class NE_TF():
    def __init__(self, max_df=1.0, min_df=1, use_bns=True, max_features=None, final_norm=None, norm=None, use_idf=False, smooth_idf=False, sublinear_tf=False, remove_outliers = False):
        self.countVec = CountVectorizer(stop_words="english", binary=True, ngram_range=(1, 1), strip_accents='unicode', max_df=max_df, min_df=min_df, max_features=max_features)
        self.bns = None
        self.tf = TfidfTransformer(norm=None, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
        self.norm = final_norm
        self.use_bns = use_bns
        self.remove_outliers = remove_outliers

    def fit(self, X, y=None):
        self.countVec.fit(X)
        if self.remove_outliers:
        	y_pred = LocalOutlierFactor().fit_predict(self.countVec.transform(X))
        	mask = y_pred == -1
        	X = np.array(X)[mask].tolist()
        	if y is not None:
        		y = y[mask]
        	self.countVec.fit(X)
        vocab = self.countVec.vocabulary_
        X_vec = self.countVec.transform(X)
        self.bns = NETransformer(vocab=vocab)
        self.bns.fit(X_vec)
        self.tf.fit(X_vec)
    
    def transform(self, X):
        X_vec = self.countVec.transform(X)
        res = self.tf.transform(X_vec)
        if self.use_bns:
        	X_bns = self.bns.transform(X_vec)
        	res = X_bns.multiply(res)
        if self.norm is None:
        	return res
        return normalize(res, norm=self.norm, axis=1)