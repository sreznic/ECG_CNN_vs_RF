import json
import random
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

def get_all_records(description):
    all_records = []
    # Flatten the dataset_description dictionary
    for dataset_name in description.keys():
        for arrhythmia in description[dataset_name].keys():
            all_records.extend(description[dataset_name][arrhythmia])

    return list(set(all_records))

def get_arrhythmia_label(record):
    with open(record + '.hea', 'r') as f:
        arrhythmias = f.readlines()[15]
        arrhythmias = arrhythmias[5:].split(',')
        arrhythmias = [arrhythmia.strip() for arrhythmia in arrhythmias]
        arrhythmias = set(arrhythmias)
        arrhythmias = ','.join(arrhythmias)
        return arrhythmias

with open('dataset_description.json', 'r') as f:
    dataset_description = json.load(f)

with open('train_description.json', 'r') as f:
    train_description = json.load(f)

all_records = get_all_records(dataset_description)
train_records = get_all_records(train_description)

train_records_arrhythmias = [get_arrhythmia_label(record) for record in tqdm(train_records, desc="Getting hea arrhythmais")]
counter = Counter(train_records_arrhythmias)
label_0 = []
for item in list(counter.items()):
    if item[1] == 1:
        label_0.append(item[0])
label_0 = set(label_0)
train_records_arrhythmias = ['label0' if a in label_0 else a for a in train_records_arrhythmias]

dl_train, dl_val = train_test_split(
    train_records, 
    test_size=0.15 / (len(train_records) / len(all_records)), 
    stratify=train_records_arrhythmias)
dl_train = set(dl_train)
dl_val = set(dl_val)

dl_train_description = {}
dl_val_description = {}

for dataset_name in dataset_description.keys():
    dl_train_description[dataset_name] = {}
    dl_val_description[dataset_name] = {}
    for arrhythmia in dataset_description[dataset_name].keys():
        dl_train_description[dataset_name][arrhythmia] = list(set(dataset_description[dataset_name][arrhythmia]) & dl_train)
        dl_val_description[dataset_name][arrhythmia] = list(set(dataset_description[dataset_name][arrhythmia]) & dl_val)

# Save dl train and val descriptions
with open('dl_train_description.json', 'w') as f:
    json.dump(dl_train_description, f, indent=2)

with open('dl_val_description.json', 'w') as f:
    json.dump(dl_val_description, f, indent=2)
