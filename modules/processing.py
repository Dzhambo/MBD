import torch
import numpy as np
import pandas as pd
import calendar
from glob import glob
from ptls.data_load.utils import collate_feature_dict

from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset
from datetime import datetime
from ptls.data_load.padded_batch import PaddedBatch

class TargetToTorch(IterableProcessingDataset):
    def __init__(self, col_target):
        super().__init__()
        self.col_target = col_target

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            features[self.col_target] = torch.tensor(features[self.col_target]).unsqueeze(0)
            yield features


class TypeProc(IterableProcessingDataset):
    def __init__(self, col_name, tp='float'):
        super().__init__()
        self.col_name = col_name
        self.tp = tp

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            if type(features[self.col_name]) is not str:
                features[self.col_name] = np.array([float(val) if self.tp=='float' else int(float(val)) for val in features[self.col_name]])
            else:
                features[self.col_name] = float(features[self.col_name]) if self.tp=='float' else int(float(features[self.col_name]))
            yield features

class DeleteNan(IterableProcessingDataset):
    def __init__(self, col_name):
        super().__init__()
        self.col_name = col_name
    
    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            if features[self.col_name] is not None:
                yield features

class QuantilfyAmount(IterableProcessingDataset):
    def __init__(self, col_amt='amount', quantilies=None):
        super().__init__()
        self.col_amt = col_amt
        if quantilies is None:
            self.quantilies = [0., 267.6, 1198.65, 3667.2, 8639.8, 18325.7, 36713.2, 68950.3, 143969.1, 421719.1]
        else: 
            self.quantilies = quantilies
    
    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            amount = features[self.col_amt]
            am_quant = torch.zeros(len(amount), dtype=torch.int)
            for i, q in enumerate(self.quantilies):
                am_quant = torch.where(amount>q, i, am_quant)
            features[self.col_amt] = am_quant
            yield features


class GetSplit(IterableProcessingDataset):
    def __init__(
        self,
        start_month,
        end_month,
        year=2022,
        col_id='client_id',
        col_time='event_time'
    ):
        super().__init__()
        self.start_month = start_month
        self.end_month = end_month
        self._year = year
        self._col_id = col_id
        self._col_time = col_time
        
    def __iter__(self):
        for rec in self._src:
            for month in range(self.start_month, self.end_month+1):
                features = rec[0] if type(rec) is tuple else rec
                features = features.copy()
                
                if month == 12:
                    month_event_time = datetime(self._year + 1, 1, 1).timestamp()
                else:
                    month_event_time = datetime(self._year, month + 1, 1).timestamp()
                    
                year_event_time = datetime(self._year, 1, 1).timestamp()
                
                mask = features[self._col_time] < month_event_time
                
                for key, tensor in features.items():
                    if key.startswith('target'):
                        features[key] = tensor[month - 1].tolist()    
                    elif key != self._col_id:
                        features[key] = tensor[mask] 
                            
                features[self._col_id] += '_month=' + str(month)

                yield features

def collate_feature_dict_with_target(batch, col_id='client_id', target_col_names=None):
    batch_ids = []
    target_cols = []
    for sample in batch:
        batch_ids.append(sample[col_id])
        del sample[col_id]
        
        if target_col_names is not None:
            for target_col in target_col_names:
                target_cols.append(sample[target_col])
                del sample[target_col]
            
    padded_batch = collate_feature_dict(batch)
    if target_col_names is not None:
        return padded_batch, batch_ids, target_cols
    return padded_batch, batch_ids