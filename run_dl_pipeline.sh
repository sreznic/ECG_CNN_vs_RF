#!/bin/bash

echo "Step 1: Training ECG Feature Extraction Modules"
python3 deeplearning/with_feats_extr/leads_feat_extr.py --config deeplearning/with_feats_extr/configs/leads_feat_extr.json
if [ $? -ne 0 ]; then
    echo "Error in Step 1: Train ECG Feature Extraction Modules"
    exit 1
fi

echo "Step 2: Extracting All Features from All Leads"
python3 deeplearning/with_feats_extr/extr_all_feats.py --config deeplearning/with_feats_extr/configs/extr_all_feats.json
if [ $? -ne 0 ]; then
    echo "Error in Step 2: Extract All Features from All Leads"
    exit 1
fi

echo "Step 3: Finding Optimal Subsets of ECG Leads"
python3 deeplearning/with_feats_extr/dl_optimal_subsets.py --config deeplearning/with_feats_extr/configs/dl_optimal_subsets.json
if [ $? -ne 0 ]; then
    echo "Error in Step 3: Find Optimal Subsets of ECG Leads"
    exit 1
fi

echo "Deep Learning Pipeline completed successfully!"
