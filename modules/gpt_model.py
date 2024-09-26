import torch
from ptls.data_load.padded_batch import PaddedBatch
from ptls.frames.gpt.gpt_module import GptPretrainModule


class CorrGptPretrainModule(GptPretrainModule):
    def __init__(self, feature_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=['trx_encoder', 'seq_encoder', 'feature_encoder'])
        self.feature_encoder = feature_encoder
    
    def forward(self, batch):
        z_trx = self.trx_encoder(batch)
        payload = z_trx.payload.view(z_trx.payload.shape[:-1] + (-1, 24))
        payload = self.feature_encoder(payload)
        encoded_trx = PaddedBatch(payload=payload, length=z_trx.seq_lens)
        out = self._seq_encoder(encoded_trx)
        if self.hparams.norm_predict:
            out = self.fn_norm_predict(out)
        return out