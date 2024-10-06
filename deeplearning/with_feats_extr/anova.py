import os
import json

# Read logs that start with the prefix "optimal_record_"
logs_path = "deeplearning/with_feats_extr/logs"
logs_names = [f for f in os.listdir(logs_path) if os.path.isfile(os.path.join(logs_path, f)) and f.startswith("optimal_record_")]
logs_paths = [os.path.join(logs_path, f) for f in logs_names]

logs_jsons = []
for log_path in logs_paths:
    with open(log_path, 'r') as f:
        logs_jsons.append(json.load(f))

log_js = {}
for log_key in logs_jsons[0].keys():
    log_js[log_key] = []
    for log_json in logs_jsons:
        log_js[log_key].append(log_json[log_key][0])

arrs = []
keys = []
for key in log_js.keys():
    keys.append(key)
    arrs.append(log_js[key])

# Import Kruskal-Wallis Test
from scipy.stats import kruskal



pass