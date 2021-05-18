from sklearn.datasets import load_files
import numpy as np
    
#https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews
#https://www.kaggle.com/tboyle10/medicaltranscriptions
#https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset?select=Fake.csv
#https://www.kaggle.com/jrobischon/wikipedia-movie-plots

def load_binary_data(category_lst=["ECommerce", "FakeNews", "Jobs", "Keylogger", "MedicalTranscriptions", "MoviePlots", "Wikileaks"], target="Wikileaks"):
    dataset = load_files("Datasets", categories=category_lst, load_content=True)
    data, y = dataset['data'], dataset['target']
    data = np.asarray([x.decode() for x in data])

    X = []
    Y = []

    for i in range(len(data)):
        cur_lst = data[i].split("\n")
        cur_len = len(cur_lst)
        X = X + cur_lst
        Y = Y + [y[i]]*cur_len
    
    Y = np.array(Y)

    index = dataset.target_names.index(target)
    Y = (Y == index).astype(int)
    return X, Y