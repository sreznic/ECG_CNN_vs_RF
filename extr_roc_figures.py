import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc
import numpy as np

def calculate_mean_roc_curve(res_dict):
    all_fpr = []
    all_tpr = []
    for i in range(len(res_dict['test_probabilities'])):
        fpr, tpr, _ = roc_curve(res_dict['test_target_values'] == 0, res_dict['test_probabilities'][i][:, 0])
        auc_score = auc(fpr, tpr)
        if auc_score < 0.7:
            continue
        all_fpr.append(fpr)
        all_tpr.append(tpr)
    
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)

    for i in range(len(all_fpr)):
        mean_tpr += np.interp(mean_fpr, all_fpr[i], all_tpr[i])

    mean_tpr /= len(all_fpr)
    
    roc_auc = auc(mean_fpr, mean_tpr)

    return mean_fpr, mean_tpr, roc_auc
        

def plot_roc_curves_save(res_dicts, labels, title, figure_path):
    fprs, tprs, rocs = [], [], []
    for i in range(len(res_dicts)):
        fpr, tpr, roc = calculate_mean_roc_curve(res_dicts[i])
        fprs.append(fpr)
        tprs.append(tpr)
        rocs.append(roc)

    for i in range(len(res_dicts)):
        plt.plot(fprs[i], tprs[i], lw=2, label=f'{labels[i]} (AUC = {rocs[i]:0.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')

    plt.savefig(figure_path)
    plt.close()