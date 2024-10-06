import numpy as np
import sklearn.metrics as metrics

def get_f1_score(acc_dict):
    return metrics.f1_score(
        np.argmax(acc_dict['true'], axis=1), 
        np.argmax(acc_dict['pred'], axis=1), 
        average='macro')

def print_accuracies(acc_dict):
    keys = ['train', 'val', 'test']
    for key in keys:
        print(f"{key}: {get_f1_score(acc_dict[key])}")
        print(key, metrics.classification_report(
            np.argmax(acc_dict[key]['true'], axis=1), 
            np.argmax(acc_dict[key]['pred'], axis=1), 
            target_names=['STD', 'NSR', 'Others']))
        pass

results = np.load('deeplearning/previous_research/results/af_all_steps.npy', allow_pickle=True)
results_os_rf = np.load('logs/optimal_subsets/rf/os_rec_rf_STD+1689931778.npy', allow_pickle=True).item()
last_state = np.load('deeplearning/previous_research/last_states/af_last_state.npy', allow_pickle=True).item()

all_steps_dict = {}
for i in range(2, len(last_state['all_steps']), 4):
    info = last_state['all_steps'][i]
    if str(info['leads']) not in all_steps_dict:
        all_steps_dict[str(info['leads'])] = []
    all_steps_dict[str(info['leads'])].append(info)

def get_leads_steps(last_state):
    LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    leads = [LEAD_NAMES[i] for i in last_state['leads_list']]
    leads_steps = []
    for i in range(len(leads)):
        leads_steps.append(leads.copy())
        _ = leads.pop()

    return leads_steps

def get_accuracies_dict(dicts):
    avg_dict = {}
    for key in dicts.keys():
        f1_scores = []
        for dic in dicts[key]:
            f1_sc = metrics.f1_score(
                np.argmax(dic['test']['true'], axis=1), 
                np.argmax(dic['test']['pred'], axis=1), 
                average='macro')
            f1_scores.append(f1_sc)
        avg_dict[key] = np.mean(f1_scores)
    return avg_dict
        


def get_classifiers_dicts(last_state):
    
    return {}

report = {}
report['f1_scores'] = np.mean(last_state['f1_scores_total'], axis=1)
report['lead_steps'] = get_leads_steps(last_state)

print(get_accuracies_dict(all_steps_dict))
pass