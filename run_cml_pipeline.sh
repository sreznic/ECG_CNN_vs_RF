#!/bin/bash

echo "Step 1: Extracting Features"
python3 features_extr/extr_all.py --config config/extract_feats_config.json
if [ $? -ne 0 ]; then
    echo "Error in Step 1: Extract Features"
    exit 1
fi

echo "Step 2: Processing Features"
python3 features/features_process.py --config configs/features_process.json
if [ $? -ne 0 ]; then
    echo "Error in Step 2: Process Features"
    exit 1
fi

echo "Step 3: Finding Hyperparameters"
python3 optimize_hyperparam.py --config configs/optim_hyperparam.json
if [ $? -ne 0 ]; then
    echo "Error in Step 3: Find Hyperparameters"
    exit 1
fi

echo "Step 4: Finding Highly Correlated Features"
python3 find_highly_corr_feats.py --config configs/highly_corr_feats.json
if [ $? -ne 0 ]; then
    echo "Error in Step 4: Find Highly Correlated Features"
    exit 1
fi

echo "Step 5: Identifying Redundant Features using Backward Stepwise Regression"
python3 rf_backward_stepwise.py --config configs/rf_backward_stepwise.json
if [ $? -ne 0 ]; then
    echo "Error in Step 5: Identify Redundant Features"
    exit 1
fi

echo "Step 6: Removing Redundant Features and Creating New Dataset"
python3 features/feature_remove_redundant.py --config configs/feature_remove_redundant.json
if [ $? -ne 0 ]; then
    echo "Error in Step 6: Remove Redundant Features"
    exit 1
fi

echo "Step 7: Finding Optimal Hyperparameters for the New Dataset"
python3 optimize_hyperparam.py --config configs/optim_hyperparam.json
if [ $? -ne 0 ]; then
    echo "Error in Step 7: Find Optimal Hyperparameters"
    exit 1
fi

echo "Step 8: Finding Optimal Subsets of ECG Leads"
python3 optimal_subset.py --config configs/optimal_subsets.json
if [ $? -ne 0 ]; then
    echo "Error in Step 8: Find Optimal Subsets of ECG Leads"
    exit 1
fi

echo "Pipeline completed successfully!"
