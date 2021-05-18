from sklearn.datasets import load_files
import numpy as np
import gc
from itertools import compress
from sklearn.model_selection import train_test_split

def load_binary_data(category_lst=["ECommerce", "FakeNews", "Jobs", "Keylogger", "MedicalTranscriptions", "MoviePlots", "Wikileaks"], target="Wikileaks"):
    dataset = load_files("Datasets", categories=category_lst, load_content=True)
    data, y = dataset['data'], dataset['target']
    data = [x.decode() for x in data]

    X = []
    Y = []

    for i in range(len(data)):
        cur_lst = data[i].split("\n")
        cur_len = len(cur_lst)
        X = X + cur_lst
        Y = Y + [y[i]]*cur_len
    
    gc.collect()
    
    Y = np.array(Y)

    index = dataset.target_names.index(target)
    Y = (Y == index).astype(int)
    return X, Y

def exec_splits(X, y):
    X_pos = list(compress(X, y==1))
    X_neg = list(compress(X, y==0))
    y_pos = list(compress(y, y==1))
    y_neg = list(compress(y, y==0))

    X_train, X_split, y_train, y_split = train_test_split(X_pos, y_pos, test_size=0.30)
    X_valid_pos, X_test_pos, y_valid_pos, y_test_pos = train_test_split(X_split, y_split, test_size=0.50)
    X_valid_neg, X_test_neg, y_valid_neg, y_test_neg = train_test_split(X_neg, y_neg, test_size=0.50)

    X_valid = X_valid_pos + X_valid_neg
    y_valid = np.hstack((y_valid_pos, y_valid_neg))

    X_test = X_test_pos + X_test_neg
    y_test = np.hstack((y_test_pos, y_test_neg))

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def load_and_split(category_lst=["ECommerce", "FakeNews", "Jobs", "Keylogger", "MedicalTranscriptions", "MoviePlots", "Wikileaks"], target="Wikileaks"):
    X, y = load_binary_data(category_lst=category_lst, target=target)
    return exec_splits(X, y)
