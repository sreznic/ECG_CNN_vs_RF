import numpy as np
import os
import sklearn.metrics as metrics

arrhythmias = ['AF', 'IAVB', 'LBBB', 'PAC', 'PVC', 'RBBB', 'STD', 'STE']

def get_avg_accuracy(f1s, arrhythmia, lead_num):
    arrhythmia_index = arrhythmias.index(arrhythmia)
    lead_index = lead_num - 1
    f1s_arr = f1s[arrhythmia_index]
    f1s_lead = f1s_arr[lead_index]
    return np.average(f1s_lead)

def accustom_cl_dict(cl_dict):
    if 'val' not in cl_dict:
        cl_dict['val'] = {}
        cl_dict['val']['pred'] = cl_dict['val_predictions']
        cl_dict['val']['target'] = cl_dict['val_target_values']
        cl_dict['test'] = {}
        cl_dict['test']['pred'] = cl_dict['test_predictions']
        cl_dict['test']['target'] = cl_dict['test_target_values']
    return cl_dict

def extract(dir, save_path):
    contents = os.listdir(dir)
    file_names = []
    for arr in arrhythmias:
        file_names.append([f for f in contents if arr in f][0])
    file_names = [os.path.join(dir, f) for f in file_names]
    f1s_vals = []
    f1s_tests = []
    for f in file_names:
        data = np.load(f, allow_pickle=True).item()
        report = data['report']
        f1_v = []
        f1_t = []
        for cl_dict in report['classifier_dicts']:
            cl_dict = accustom_cl_dict(cl_dict)
            f1_v.append([])
            f1_t.append([])

            val_predictions = cl_dict['val']['pred']
            val_targets = cl_dict['val']['target']
            for preds, targets in zip(val_predictions, val_targets):
                f1_v[-1].append(metrics.f1_score(targets, preds, average='macro'))

            test_predictions = cl_dict['test']['pred']
            test_targets = cl_dict['test']['target']
            for i, t in enumerate(test_predictions):
                if isinstance(test_targets[0], np.ndarray) or isinstance(test_targets[0], list):
                    targ = test_targets[i]
                else:
                    targ = test_targets
                f1_t[-1].append(metrics.f1_score(targ, t, average='macro'))

        f1s_vals.append(f1_v)
        f1s_tests.append(f1_t)
        pass
    res_dict = {
        'f1s_vals': f1s_vals,
        'f1s_tests': f1s_tests,
        'arrhythmias': arrhythmias,
        'dir': dir
    }
    np.save(save_path, res_dict, allow_pickle=True)
    pass

def main():
    extract('dl', 'dl_save')
    extract('rf', 'rf_save')
    pass

if __name__ == '__main__':
    main()
