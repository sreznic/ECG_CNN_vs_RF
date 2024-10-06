import json

def add_run_conf(js):
    rc = js['run_configuration']
    js['run_conf_name'] = f"{rc['leads']}{rc['fraction_of_train']}{rc['balance_train_samples']}{rc['num_of_batches_test_val']}{rc['num_of_batches_train']}{rc['record_len']}"

json2 = json.load(open('deeplearning/run_outputs_2.json', 'r'))['run_outputs']

sums_json = {}

for js in json2:
    add_run_conf(js)
    if js['run_conf_name'] in sums_json:
        sums_json[js['run_conf_name']].append(js)
    else:
        sums_json[js['run_conf_name']] = [js]   

average_json = {}

for key in sums_json.keys():
    if key not in average_json:
        average_json[key] = {}
    for js in sums_json[key]:
        for k in js.keys():
            if k == 'run_configuration' or k == "run_conf_name":
                average_json[key][k] = js[k]
                continue
            if k not in average_json[key]:
                average_json[key][k] = {}
            for k2 in js[k].keys():
                if k2 not in average_json[key][k]:
                    average_json[key][k][k2] = 0
                average_json[key][k][k2] += js[k][k2] / 4
    
basic_config = {
    "leads": [i for i in range(12)],
    "fraction_of_train": 0.2,
    "balance_train_samples": True,
    "num_of_batches_test_val": 300,
    "num_of_batches_train": 600,
    "record_len": 5000
}

def difference_dict(orig, new):
    new_dict = {}
    for key in orig.keys():
        if orig[key] != new[key]:
            new_dict[key] = new[key]
    return new_dict

for conf_str in average_json.keys():
    res = average_json[conf_str]
    print("=" * 20)
    print("Configuration: ", difference_dict(basic_config, average_json[conf_str]['run_configuration']))
    print("F1 score macro: ", average_json[conf_str]['f1_score_macro'])
    # print("F1 score weighted: ", average_json[conf_str]['f1_score_weighted'])
    # print("AUC macro: ", average_json[conf_str]['auc_macro'])
    # print("AUC weighted: ", average_json[conf_str]['auc_weighted'])
    # print("AURPC_macro: ", average_json[conf_str]['aurpc_macro'])
    print("=" * 20)
    pass

pass