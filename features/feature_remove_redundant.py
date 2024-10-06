# Add the directory of the parent folder to the system path
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from time import time
from mlpython.utils.functions import read_config, save_log, add_leads_to_feats_list

import argparse
parser = argparse.ArgumentParser(description='Removing redundant features')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
args = parser.parse_args()
config = read_config(args.config)
df = pd.read_csv(config['csv_path'])

redundant_cols = config['highly_corr_feats'] + config['rf_redundant_feats']

redundant_cols = add_leads_to_feats_list(redundant_cols)

new_df = df.drop(redundant_cols, axis=1)

new_df.to_csv(config['output_csv_path'], index=False)