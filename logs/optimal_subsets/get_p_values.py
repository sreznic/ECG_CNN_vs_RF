import numpy as np
import os
import sklearn.metrics as metrics
from scipy import stats
from statsmodels.stats import multitest

def get_avg_f1s_arr(f1s_arr):
    return [np.mean(f) for f in f1s_arr]

dl_optimal_subsets = {
    "LBBB": 3,
    "PAC": 5,
    "PVC": 5,
    "RBBB": 5,
    "STD": 9,
    "STE": 3, 
    "AF": 5,
    "IAVB": 4
}

rf_optimal_subsets = {
    "LBBB": 11,
    "PAC": 11,
    "PVC": 6,
    "RBBB": 11,
    "STD": 7,
    "STE": 4,
    "AF": 8,
    "IAVB": 9
}

def get_p_values(file1, file2):
    dl = np.load(file1, allow_pickle=True).item()
    rf = np.load(file2, allow_pickle=True).item()
    p_vals = []
    p_vals.append(get_p_vals_1(dl['f1s_vals'], rf['f1s_vals'], dl, rf))
    p_vals.append(get_p_vals_1(dl['f1s_tests'], rf['f1s_tests'], dl, rf))
    # Adjust False positive rate
    rejected, p_adjusted, _, _ = multitest.multipletests(p_vals[0] + p_vals[1], method='fdr_bh')
    pass
    p_vals = [p_adjusted[:24], p_adjusted[24:]]
    print("Validation")
    print("=" * 30)
    print_p_vals(p_vals[0], dl)

    print("Testing")
    print("=" * 30)
    print_p_vals(p_vals[1], dl)

def print_p_vals(p_vals, dl):
    t = 0
    for j in range(3):
        for i in range(len(p_vals) // 3):
            lead_num = ["1-lead", "Optimal", "All"]
            lead_name = lead_num[j]
            print('arrhythmia: {}, lead: {} p_val: {}'.format(dl['arrhythmias'][i], lead_name, p_vals[t]))
            t += 1
    

def get_p_vals_1(dl_f1s, rf_f1s, dl, rf):
    p_vals = []
    for j in range(3):
        for i in range(len(dl_f1s)):
            dl_a = dl_f1s[i]
            rf_a = rf_f1s[i]
            leads_dl = [1, dl_optimal_subsets[dl['arrhythmias'][i]], 12]
            leads_rf = [1, rf_optimal_subsets[rf['arrhythmias'][i]], 12]
            lead_num = ["1-lead", "Optimal", "All"]
            dl_lead = dl_a[leads_dl[j] - 1]
            rf_lead = rf_a[leads_rf[j] - 1]
            # Mannhatan U test
            mu_test = stats.mannwhitneyu(dl_lead, rf_lead)
            # Shapiro-Wilk test
            lead_name = lead_num[j]
            # print('arrhythmia: {}, lead: {}'.format(dl['arrhythmias'][i], lead_name))
            # print('MU test: {}'.format(mu_test.pvalue))
            p_vals.append(mu_test.pvalue)
    
    return p_vals

def get_p_vals_2(dl_f1s, rf_f1s, dl, rf):
    for i in range(len(dl_f1s)):
        dl_a = dl_f1s[i]
        rf_a = rf_f1s[i]
        leads_dl = [1, dl_optimal_subsets[dl['arrhythmias'][i]], 12]
        leads_rf = [1, rf_optimal_subsets[rf['arrhythmias'][i]], 12]
        lead_num = ["1-lead", "Optimal", "All"]
        for j in range(len(leads_dl)):
            
            dl_lead = dl_a[leads_dl[j] - 1]
            rf_lead = rf_a[leads_rf[j] - 1]
            # Mannhatan U test
            mu_test = stats.mannwhitneyu(dl_lead, rf_lead)
            t_test = stats.ttest_ind(dl_lead, rf_lead)
            # Shapiro-Wilk test
            sw_test = stats.shapiro(dl_lead)
            lead_name = lead_num[j]
            print('arrhythmia: {}, lead: {}'.format(dl['arrhythmias'][i], lead_name))
            # print('dl_avg: {}, rf_avg: {}'.format(np.average(dl_lead), np.average(rf_lead)))
            print('MU test: {}'.format(mu_test.pvalue))
            # print('t-test: {}'.format(t_test.pvalue))
            # print('shapiro-wilk: {}'.format(sw_test.pvalue))
            print()
        pass
    pass

def main():
    get_p_values('dl_save.npy', 'rf_save.npy')

if __name__ == '__main__':
    main()