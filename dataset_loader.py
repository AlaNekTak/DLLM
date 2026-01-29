import numpy as np
from datasets import load_dataset



def subsample(data, size = 0, seed = 0):
    if size <= 0:
        return data

    random_rng = np.random.default_rng(seed)
    for split in data.keys():
        for label in data[split]:
            if len(data[split][label]) <= size:
                continue
            indices = random_rng.choice(len(data[split][label]), size, replace=False)
            data[split][label] = [data[split][label][i] for i in indices]

    return data

def load_IMDB(size = 0, seed = 0):
    dataset = load_dataset("imdb")
    classes = set(dataset["train"]["label"])

    data = {'train': {}, 'test': {}}
    for split in ['train', 'test']:
        data[split] = {label: [] for label in classes}
        for example in dataset[split]:
            data[split][example["label"]].append(example['text'])

    data = subsample(data, size, seed)

    return data

def load_AGNews(size = 0, seed = 0):
    dataset = load_dataset("ag_news")
    classes = set(dataset["train"]["label"])

    data = {'train': {}, 'test': {}}
    for split in ['train', 'test']:
        data[split] = {label: [] for label in classes}
        for example in dataset[split]:
            data[split][example["label"]].append(example['text'])

    data = subsample(data, size, seed)
    return data

def load_YELP(size = 0, seed = 0):
    dataset = load_dataset("yelp_polarity")
    classes = set(dataset["train"]["label"])

    data = {'train': {}, 'test': {}}
    for split in ['train', 'test']:
        data[split] = {label: [] for label in classes}
        for example in dataset[split]:
            data[split][example["label"]].append(example['text'])

    data = subsample(data, size, seed)
    return data

def load_NYTimes(size = 0, seed = 0):
    assert False, 'There is no test split, fix later'
    dataset = load_dataset("dstefa/New_York_Times_Topics")
    classes = set(dataset["train"]["topic_id"])

    data = {'train': {}, 'test': {}}
    for split in ['train', 'test']:
        data[split] = {label: [] for label in classes}
        for example in dataset[split]:
            data[split][example["topic_id"]].append(example['text'])

    data = subsample(data, size, seed)
    return data

from tqdm.auto import tqdm
def load_DBPedia(size = 0, seed = 0):
    dataset = load_dataset("mteb/dbpedia", 'corpus')
    data = {}
    data['corpus'] = {}
    data['corpus'][0] = [i for i in range(len(dataset['corpus']))]
    data = subsample(data, size, seed)
    data['corpus'][0] = [dataset['corpus'][i]['text'] for i in data['corpus'][0]]
    return data

classes_to_labels = {'imdb': {'negative':0, 'positive': 1},
                     'yelp': {'negative':0, 'positive': 1},
                     'agnews': {'world':0, 'sports':1, 'sci-tech':2, 'business':3},
}

datasets_to_functions = {'imdb': load_IMDB, 'agnews': load_AGNews, 'yelp': load_YELP, 'dbpedia': load_DBPedia}



if __name__ == "__main__":
    data = load_DBPedia(500)
    print("DBPedia dataset loaded")
    
    for dataset_fn in [load_IMDB, load_AGNews, load_YELP, load_NYTimes]:
        data = dataset_fn()

        print(f"Loaded dataset: {dataset_fn.__name__}")
        print("Number of training samples:", [len(data['train'][label]) for label in data['train']])
        print("Number of test samples:", [len(data['test'][label]) for label in data['test']])
        print()

        train_data = data['train'][0]
        test_data = data['test'][1]        
        print("Train data:", train_data[0])  # Print first training sample with label 0
        print("Test data:", test_data[0])  # Print first test sample with label 1
    