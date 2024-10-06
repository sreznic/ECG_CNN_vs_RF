import json
import random
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

def get_arrhythmia_label(record):
    with open(record + '.hea', 'r') as f:
        arrhythmias = f.readlines()[15]
        arrhythmias = arrhythmias[5:].split(',')
        arrhythmias = [arrhythmia.strip() for arrhythmia in arrhythmias]
        arrhythmias = set(arrhythmias)
        arrhythmias = ','.join(arrhythmias)
        return arrhythmias

def main():

    with open('dataset_description.json', 'r') as f:
        dataset_description = json.load(f)
        all_records = []
        all_arrhythmias = []
        # Flatten the dataset_description dictionary
        for dataset_name in dataset_description.keys():
            for arrhythmia in dataset_description[dataset_name].keys():
                all_records.extend(dataset_description[dataset_name][arrhythmia])
                all_arrhythmias.append(arrhythmia)
        all_records = list(set(all_records))
        all_records_arrhythmias = [get_arrhythmia_label(record) for record in tqdm(all_records, desc="Getting hea arrhythmais")]
        counter = Counter(all_records_arrhythmias)
        label_0 = []
        for item in list(counter.items()):
            if item[1] == 1:
                label_0.append(item[0])
        label_0 = set(label_0)
        all_records_arrhythmias = ['label0' if a in label_0 else a for a in all_records_arrhythmias]

        all_records_length = len(all_records)
        # Split dataset using sklearn train_test_split
        train, test = train_test_split(all_records, test_size=0.15, stratify=all_records_arrhythmias)

        train = set(train)
        test = set(test)
        
        train_description = {}
        test_description = {}

        for dataset_name in dataset_description.keys():
            train_description[dataset_name] = {}
            test_description[dataset_name] = {}
            for arrhythmia in dataset_description[dataset_name].keys():
                train_description[dataset_name][arrhythmia] = list(set(dataset_description[dataset_name][arrhythmia]) & train)
                test_description[dataset_name][arrhythmia] = list(set(dataset_description[dataset_name][arrhythmia]) & test)
        # Save test and train descriptions
        with open('test_description.json', 'w') as f:
            json.dump(test_description, f, indent=2)
        with open('train_description.json', 'w') as f:
            json.dump(train_description, f, indent=2)

if __name__ == '__main__':
    main()