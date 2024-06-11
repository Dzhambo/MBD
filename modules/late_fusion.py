import sys
import os

import pandas as pd
import numpy as np
import torch
from glob import glob
from torch.utils.data import DataLoader

import pytorch_lightning as pl

import logging
import pickle


from collections import defaultdict

from ptls.data_load.feature_dict import FeatureDict
from ptls.frames.coles.split_strategy import AbsSplit

from functools import partial

import warnings
warnings.filterwarnings("ignore")

from tqdm.auto import tqdm
import lightgbm as ltb
import json


class LateFusion:
    def __init__(
        self,
        train1_path,
        test1_path,
        train2_path,
        test2_path,
        params,
        result_path,
        col_id='client_id',
        targets=(
            'target_1',
            'target_2',
            'target_3',
            'target_4'
        )
    ):
        self.train1_path = train1_path
        self.train2_path = train2_path
        self.test1_path = test1_path
        self.test2_path = test2_path
        self.col_id = col_id
        self.all_targets = targets
        self.drop_feat = list(self.all_targets) + [self.col_id]
        self.result_path = result_path
        self.params = params
        
    def fit(self):
        self.logger.info('Read train data...')
        train_embeddings1 = pd.read_parquet(self.train1_path)
        train_embeddings2 = pd.read_parquet(self.train2_path).drop(columns=list(self.all_targets))
        
        self.logger.info('Run train concate...')
        train_embeddings = train_embeddings1.merge(train_embeddings2, on=self.col_id, how='outer').fillna(0)
        del train_embeddings1
        del train_embeddings2
        
        X_train = train_embeddings.drop(columns=self.drop_feat)
        self.logger.info('Start fit...')
        clfs = dict()
        for col_target in self.all_targets:
            clf = ltb.LGBMClassifier(**self.params)
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
        self.logger.info('Read test data...')
        test_embeddings1 = pd.read_parquet(self.test1_path)
        test_embeddings2 = pd.read_parquet(self.test2_path)
        
        self.logger.info('Run test concate...')
        test_embeddings = test_embeddings1.merge(test_embeddings2, on=self.col_id, how='outer').fillna(0)
        del test_embeddings1
        del test_embeddings2
        self.logger.info('Run testing...')
        X_test = test_embeddings.drop(columns=[self.col_id])
        ids = test_embeddings[self.col_id]
        scores[self.col_id] = ids
            
        for col_target in self.all_targets:
            clf = clfs[col_target]
            score = clf.predict_proba(X_test)[:, 1]
            scores[col_target] = score
        
        return scores
    
    def run(self):
        clfs = self.fit()
        scores = self.get_scores(clfs)
        scores.to_csv(self.result_path)
            
        return scores