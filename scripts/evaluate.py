from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import logging
import argparse
import json

logger = logging.getLogger(__name__)

def evaluate(ref_df, pred_df):
    scores = pd.read_csv(pred_df).drop_duplicates('client_id')
    target = pd.read_parquet(ref_df).drop_duplicates('client_id')
        
    res_df = scores.merge(target, on='client_id', how='inner')

    metrics = []
    for target in ['target_1', 'target_2', 'target_3', 'target_4']:
        metrics.append(
            roc_auc_score(
                res_df[target + '_y'], res_df[target + '_x']
            )
        )
    
    return np.mean(metrics)

def main():
    parser = argparse.ArgumentParser(description='Score calculation')
    parser.add_argument('--ref_df_public')
    parser.add_argument('--ref_df_private')
    
    parser.add_argument('--pred_df')
    
    parser.add_argument('--public_result_path')
    parser.add_argument('--private_result_path')
    
    args, _ = parser.parse_known_args()
    
    
    result_public = evaluate(args.ref_df_public, args.pred_df)
    result_private = evaluate(args.ref_df_private, args.pred_df)
    
    with open(args.public_result_path, 'w') as f:
        f.write(str(result_public))
    
    with open(args.private_result_path, 'w') as f:
        f.write(str(result_private))
    

if __name__ == "__main__":
    main()