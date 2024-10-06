import json

with open('test_description.json', 'r') as f:
    all_dataset = json.load(f)

total_dataset = {}

for dataset in all_dataset.keys():
    for arrhythmia in all_dataset[dataset].keys():
        arrhythmia_list = all_dataset[dataset][arrhythmia]

        if arrhythmia not in total_dataset.keys():
            total_dataset[arrhythmia] = []
        
        total_dataset[arrhythmia].append(set(arrhythmia_list))

arrhythmias_in_interest = {
    "AF": "164889003",
    "IAVB": "270492004",
    "LBBB": "164909002",
    "NSR": "426783006",
    "PAC": "284470004",
    "PVC": "427172004",
    "RBBB": "59118001",
    "STD": "429622005",
    "STE": "164931005"
}

arrhythmias_in_interest = {value: key for key, value in arrhythmias_in_interest.items()}

total = 0
for arrhythmia in total_dataset.keys():
    samples = set.union(*total_dataset[arrhythmia])
    total += len(samples)

for arrhythmia in total_dataset.keys():
    samples = set.union(*total_dataset[arrhythmia])

    if arrhythmia in set(arrhythmias_in_interest.keys()):
        print(arrhythmias_in_interest[arrhythmia], f"{len(samples) / total:.5f}", total)

print("Total: " + str(total))
print("===")