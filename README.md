# ConicalClassification
Supplementary Materials for Conical Classification For Computationally Efficient One-Class Topic Determination

## Evaluation Datasets
Datasets are loaded via the API found in load_datasets. This loads documents from each corpus and stores them as a list of strings in X, and returns y which has value 1 for documents of the target class, and 0 otherwise. This file also includes a dataset split which the dataset into train, validation, and test sets as detailed in the paper.

Each of the datasets can be downloaded from the links below. As detailed in the paper, we extract text information from each dataset and store them in text files under their respective subfolders within Datasets. For example, all data for the MoviePlots dataset will be held within txt files in Datasets/MoviePlots.

### MoviePlots
https://www.kaggle.com/jrobischon/wikipedia-movie-plots

### MedicalTranscriptions
https://www.kaggle.com/tboyle10/medicaltranscriptions

### Ecommerce
https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews

### FakeNews
https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset?select=Fake.csv

### Jobs
https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset?select=Fake.csv
https://www.kaggle.com/PromptCloudHQ/us-jobs-on-monstercom?select=monster_com-job_sample.csv
https://www.kaggle.com/PromptCloudHQ/jobs-on-naukricom?select=naukri_com-job_sample.csv

### Wikileaks
https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247/1

### Keylogger
https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247/1

## Normal Exclusion
We calculate the frequency dictionary via the code found in genWordFrequencyDict.ipynb. The dataset used for this task comes the word counts csv file that can be found here:
https://www.kaggle.com/rtatman/english-word-frequency?select=unigram_freq.csv

As Normal Exclusion is built on the backbone of Forman's Bi-Normal Separation paper, we build Normal Exclusion on his provided codebase (specifically the ficlearn subfolder) as well:
https://github.com/dumoulma/fic-prototype

Place the file normalExclusion.py into the feature_extraction subfolder of ficlearn. A simple example showing how it can work within a one-class classification problem is provided in testNE.ipynb.

We utilize the NETransformer in tandem with sklearn's TfidfTransformer in order to calculate the NE-TF used as the VSM model for the paper, as shown in NE_TF.py. A simple example showing how it can work within a one-class classification problem is provided in testNE_TF.ipynb.

## Conical Classification

As we have shown in the paper, conical classification is very simple to implement, very efficient, and yet very powerful. The implementation used for evaluation can be found in conical.py.

## Hyperparameter Tuning

For hyperparameter tuning, we use our training and validation sets to optimze each model via the hyperopt library: http://hyperopt.github.io/hyperopt/
We allow each model the ability try 20 sets of hyperparameters, with the set of hyperparameters with the highest Balanced Accuracy on our validation set being chosen.

A demonstration of the tuning methodology can be found in hoptConical.ipynb.
