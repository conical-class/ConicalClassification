import numpy as np
import re
import nltk
import pandas as pd
import string
from NE_TF import NE_TF

# Conical Based Classification For One Class Topic Sorting
class ConicalClassification():
	def __init__(self, matrix_span):
		self.minVal = np.amin(matrix_span, axis=0).A
		self.maxVal = np.amax(matrix_span, axis=0).A

	def predict(self, vector):
		#Convert to dense
		vector = vector.A

		dim = vector.shape[0]

		#Check if zero vector. If so return False.
		zero_vect = (np.count_nonzero(vector, axis=1) > 0).reshape((dim,))

		less_than = np.all(vector >= self.minVal, axis = 1).reshape((dim,))

		greater_than = np.all(vector <= self.maxVal, axis = 1).reshape((dim,))

		out = (zero_vect & less_than) & greater_than

		return out

class CorpusClassification():
	def __init__(self, argDict={}):
		self.cone_class = None 
		self.vectorizer = NE_TF(**argDict)

	def transform(self, corpus):
		return self.vectorizer.transform(corpus)

	def fit(self, corpus, y=None):
		self.vectorizer.fit(corpus, y)
		if y is not None:
			self.cone_class = ConicalClassification(self.transform(corpus)[y == 1])
		else:
			self.cone_class = ConicalClassification(self.transform(corpus))

	def predict(self, document):
	    return self.cone_class.predict(self.transform(document))
