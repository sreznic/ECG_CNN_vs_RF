import json
import os

DIR_PATH = "deeplearning/with_feats_extr"

logs_file_names = os.listdir(os.path.join(DIR_PATH, 'logs'))
logs_file_names = [l for l in logs_file_names if 'optimal_record' in l]

logs_jsons = [json.load(open(os.path.join(DIR_PATH, 'logs', logs_file_names[i]), 'r')) for i in range(len(logs_file_names))]

final_results = {}

for js in logs_jsons:
    for key in js.keys():
        if key not in final_results.keys():
            final_results[key] = []
        final_results[key].append(js[key])

for key in final_results.keys():
    new_res = []
    for i in range(len(final_results[key])):
        if final_results[key][i][0] < 0.6:
            print("Low accuracy: ", final_results[key][i][0], " for ", key, " and ", i)
            pass
        else:
            new_res.append(final_results[key][i])
    final_results[key] = new_res

for key in final_results.keys():
    sums = [0, 0, 0]
    for j in range(len(final_results[key])):
        for i in range(3):
            sums[i] += final_results[key][j][i]
    sums = [s / len(final_results[key]) for s in sums]
    final_results[key] = sums + [len(final_results[key])]
print("=" * 20)
for key in final_results.keys():
    shown = key[1:-1].split(",")
    # Trim whitespaces
    shown = [s.strip() for s in shown]
    record_length = shown[0]
    is_12_leads = len(shown) > 4
    epochs = shown[-1]
    print("{}|{}|{}|{:.3f}|{:.3f}"
          .format(record_length, '12 Leads' if is_12_leads else '1 Lead',
                  epochs, final_results[key][0], final_results[key][1]))
pass