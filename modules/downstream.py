from tqdm.auto import tqdm
import os
import lightgbm as ltb
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import hydra


class Downstream:
    def __init__(
        self,
        metric,
        train_path,
        test_path,
        model_conf,
        result_path,
        logger,
        name_exp,
        col_id='client_id',
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.metric = metric
        self.col_id = col_id
        self.all_targets = ['target_1', 'target_2', 'target_3', 'target_4']
        self.model_conf = model_conf
        self.result_path = result_path
        self.logger = logger
        self.drop_feat = self.all_targets + [self.col_id]
        self.name_exp=name_exp
        
    def fit(self):
        train_embeddings = pd.read_parquet(self.train_path)
        X_train = train_embeddings.drop(columns=self.drop_feat)

        clfs = dict()
        for col_target in self.all_targets:
            clf = hydra.utils.instantiate(self.model_conf)
            y_train = train_embeddings[col_target]
            clf.fit(X_train, y_train)
            self.logger.info(f'Model fitted, target: {col_target}')
            clfs[col_target] = clf

        return clfs

    def get_scores(
        self, 
        clfs
    ):
        scores = pd.DataFrame([])
        
        test_embeddings_curr = pd.read_parquet(self.test_path).drop_duplicates('client_id')
        X_test = test_embeddings_curr.drop(columns=[self.col_id])
        ids = test_embeddings_curr[self.col_id]
        scores[self.col_id] = ids
        
        for col_target in self.all_targets:
            clf = clfs[col_target]
            score = clf.predict_proba(X_test)[:, 1]
            scores[col_target] = score
        
        return scores

    def run(self):
        self.logger.info('Start fit ...')
        clfs = self.fit()
        self.logger.info('Start test ...')
        scores = self.get_scores(clfs)
        self.logger.info('Save test ...')
        scores.to_csv(self.result_path, mode='a')
        return scores
        
        