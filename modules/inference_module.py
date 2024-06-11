import pandas as pd
import pytorch_lightning as pl
import torch
import numpy as np
from itertools import chain
from ptls.data_load.padded_batch import PaddedBatch
from datetime import datetime


class InferenceModuleMultimodal(pl.LightningModule):
    def __init__(self, model, pandas_output=True, drop_seq_features=True, model_out_name='out'):
        super().__init__()

        self.model = model
        self.pandas_output = pandas_output
        self.drop_seq_features = drop_seq_features
        self.model_out_name = model_out_name

    def forward(self, x):
        x_len = len(x)
        if x_len == 3:
            x, batch_ids, target_cols = x
        else: 
            x, batch_ids = x
            
        out = self.model(x)
        if x_len == 3:
            target_cols = torch.tensor(target_cols)
            x_out = {
                'client_id': batch_ids,
                'target_1': target_cols[:, 0],
                'target_2': target_cols[:, 1],
                'target_3': target_cols[:, 2],
                'target_4': target_cols[:, 3],
                self.model_out_name: out
            }
        else:
            x_out = {
                'client_id': batch_ids,
                self.model_out_name: out
            }

        if self.pandas_output:
            return self.to_pandas(x_out)
        return x_out

    @staticmethod
    def to_pandas(x):
        expand_cols = []
        scalar_features = {}

        for k, v in x.items():
            if type(v) is torch.Tensor:
                v = v.cpu().numpy()

            if type(v) is list or len(v.shape) == 1:
                scalar_features[k] = v
            elif len(v.shape) == 2:
                expand_cols.append(k)
            else:
                scalar_features[k] = None

        dataframes = [pd.DataFrame(scalar_features)]
        for col in expand_cols:
            v = x[col].cpu().numpy()
            dataframes.append(pd.DataFrame(v, columns=[f'{col}_{i:04d}' for i in range(v.shape[1])]))

        return pd.concat(dataframes, axis=1)