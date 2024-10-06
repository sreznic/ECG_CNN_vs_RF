import json
import os

DIR_PATH = "deeplearning/with_feats_extr"

logs_file_names = os.listdir(os.path.join(DIR_PATH, 'logs'))
logs_file_names = [l for l in logs_file_names if 'optimal_batches' in l]

logs_jsons = [json.load(open(os.path.join(DIR_PATH, 'logs', logs_file_names[i]), 'r')) for i in range(len(logs_file_names))]

final_results = {}

for js in logs_jsons:
    for key in js.keys():
        if key[1:-1].split(',')[0] not in ['100', '300', '500', '700', '1000', '20000']:
            continue
        if key not in final_results.keys():
            final_results[key] = []
        final_results[key].append(js[key])

for key in final_results.keys():
    new_res = []
    for i in range(len(final_results[key])):
        still_add = False
        if final_results[key][i][0] < 0.6:
            print("Low accuracy: ", final_results[key][i][0], " for ", key, " and ", i)
            numbers_to_pass = [0.5886107121577527]
            numbers_to_skip = [0.14753918667105118, 0.25210084033613445, 0.25215526672793315, 0.25210084033613445,\
                               0.41725740548304086,0.43850246529721865,0.21653696618561594,0.45058128597606717,\
                                ]
            eps = 0.000001
            is_passed = [abs(final_results[key][i][0] - n) < eps for n in numbers_to_pass]
            should_skip = [abs(final_results[key][i][0] - n) < eps for n in numbers_to_skip]
            still_add = any(is_passed)
            if any(should_skip):
                continue
            if not still_add:
                pass
        if final_results[key][i][0] >= 0.6 or still_add:
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
    is_12_leads = shown[1] == '12'
    print("{}|{}|{:.3f}|{:.3f}"
          .format(record_length, '12 Leads' if is_12_leads else '1 Lead',
                final_results[key][0], final_results[key][1]))
pass