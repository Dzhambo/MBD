import pandas as pd
import pytorch_lightning as pl
import torch
import numpy as np
from itertools import chain
from ptls.data_load.padded_batch import PaddedBatch
from datetime import datetime
from ptls.custom_layers import StatPooling
from ptls.nn.seq_step import LastStepEncoder


class InferenceModuleMultimodal(pl.LightningModule):
    def __init__(
        self,
        model,
        pandas_output=True,
        col_id='client_id',
        target_col_names=None,
        model_out_name='emb',
        model_type='notab'
    ):
        super().__init__()

        self.model = model
        self.pandas_output = pandas_output
        self.target_col_names = target_col_names
        self.col_id = col_id
        self.model_out_name = model_out_name
        self.model_type = model_type

        self.stat_pooler = StatPooling()
        self.last_step = LastStepEncoder()

    def forward(self, x):
        x_len = len(x)
        if x_len == 3:
            x, batch_ids, target_cols = x
        else: 
            x, batch_ids = x
        if 'seq_encoder' in dir(self.model):
            out = self.model.seq_encoder(x)
        else:
            out = self.model(x)
            
        if x_len == 3:
            target_cols = torch.tensor(target_cols)
            x_out = {
                self.col_id: batch_ids,
                self.model_out_name: out
            }
            if len(target_cols.size()) > 1:
                for idx, target_col in enumerate(self.target_col_names):
                    x_out[target_col] = target_cols[:, idx]
            else: 
                x_out[self.target_col_names[0]] = target_cols[:, idx]
        else:
            x_out = {
                self.col_id: batch_ids,
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