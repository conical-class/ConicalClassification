# ConicalClassification
Supplementary Materials for Conical Classification For Computationally Efficient One-Class Topic Determination

## Datasets
Datasets are loaded via the API found in load_datasets. This loads documents from each corpus and stores them as a list of strings in X, and returns y which has value 1 for documents of the target class, and 0 otherwise.

Each of the datasets can be downloaded from the links below. As detailed in the paper, we extract text information from each dataset and store them in text files under their respective subfolders within Datasets. For example, all data for the MoviePlots dataset will be held within txt files in Datasets/MoviePlots.

### MoviePlots
