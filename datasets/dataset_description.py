import os
import natsort
import json
from tqdm import tqdm

# Read all the directories in the current directory
def get_dataset_dirs():
    dirs = os.listdir('.')
    dataset_dirs = []
    for dir in dirs:
        if os.path.isdir(dir):
            dataset_dirs.append(dir)
    return dataset_dirs

def get_arrhythmias_from_hea(path):
    # Read the 16th line of the file (the line that contains the arrhythmias)
    with open(path, 'r') as f:
        arrhythmias = f.readlines()[15]
        # Remove the "# Dx: " part and separate by commas
        arrhythmias = arrhythmias[5:].split(',')
        # Remove all the spaces/newlines
        arrhythmias = [arrhythmia.strip() for arrhythmia in arrhythmias]
        return arrhythmias
two_major = 0

def check_if_two_major_heart_arrhythmias(arrhythmia):
    global two_major
    major_arrhythmias = {
        '270492004': 'I-AVB',
        '164889003': 'AF',
        '164909002': 'LBBB',
        '59118001': 'RBBB',
        '284470004': 'PAC',
        '164884008': 'PVC',
        '429622005': 'STD',
        '164931005': 'STE'
    }
    major_keys = list(major_arrhythmias.keys())

    # Count the number of major arrhythmias in the given list
    count = sum(code in major_keys for code in arrhythmia)

    # Check if at least two major arrhythmias are present
    if count >= 2:
        arrhythmias = [major_arrhythmias[code] for code in arrhythmia if code in major_keys]
        two_major += 1
    else:
        pass


arrhythmias_dict = {}
for dataset_dir in tqdm(get_dataset_dirs()):
    arrhythmias_dict[dataset_dir] = {}
    # Get all the directories in the dataset directory
    dataset_parts = natsort.os_sorted(os.listdir(dataset_dir))
    for dir in tqdm(dataset_parts, leave=False):
        # Check if the directory is a directory
        if not os.path.isdir(os.path.join(dataset_dir, dir)):
            continue
        # Go over all the files in the directory
        for file in tqdm(natsort.os_sorted(os.listdir(os.path.join(dataset_dir, dir))), leave=False):
            # Check if it is a description '.hea' file
            if not file.endswith('.hea'):
                continue
            arrhythmias = get_arrhythmias_from_hea(os.path.join(dataset_dir, dir, file))
            if len(arrhythmias) > 1:
                check_if_two_major_heart_arrhythmias(arrhythmias)
            for arrhythmia in arrhythmias:
                if arrhythmia not in arrhythmias_dict[dataset_dir]:
                    arrhythmias_dict[dataset_dir][arrhythmia] = []
                arrhythmias_dict[dataset_dir][arrhythmia].append(os.path.join(dataset_dir, dir, file)[:-4])

print("Two major arrhythmias: ", two_major)
# Save json file
with open('dataset_description.json', 'w') as f:
    json.dump(arrhythmias_dict, f, indent=2)