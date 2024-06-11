import torch
import numpy as np
import pandas as pd
import calendar
from glob import glob
from ptls.data_load.utils import collate_feature_dict

from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset
from datetime import datetime
from ptls.data_load.padded_batch import PaddedBatch


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

def collate_feature_dict_with_target(batch, col_id='client_id', targets=False):
    batch_ids = []
    target_cols = []
    for sample in batch:
        batch_ids.append(sample[col_id])
        del sample[col_id]
        
        if targets:
            target_cols.append([sample[f'target_{i}'] for i in range(1, 5)])
            del sample['target_1']
            del sample['target_2']
            del sample['target_3']
            del sample['target_4']
            
    padded_batch = collate_feature_dict(batch)
    if targets:
        return padded_batch, batch_ids, target_cols
    return padded_batch, batch_ids