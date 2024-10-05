import logging

import hydra
import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset
from ptls.data_load.utils import collate_feature_dict

from modules.processing import collate_feature_dict_with_target
from modules.inference_module import InferenceModuleMultimodal
from ptls.frames.inference_module import InferenceModule
import os
import sys
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
from glob import glob
from functools import partial
from modules.downstream import Downstream
import warnings
warnings.filterwarnings('ignore')
torch.use_deterministic_algorithms(True)
logger = logging.getLogger(__name__)

def train_module(conf: DictConfig):
    if 'seed_everything' in conf:
        pl.seed_everything(conf.seed_everything)

    tb_logger = TensorBoardLogger(
        save_dir='lightning_logs',
        name=conf.get('logger_name'),
    )

    model = hydra.utils.instantiate(conf.pl_module)
    dm = hydra.utils.instantiate(conf.data_module)

    _trainer_params = conf.trainer
    _trainer_params_additional = {}
    _use_best_epoch = _trainer_params.get('use_best_epoch', False)

    if 'callbacks' in _trainer_params:
        logger.warning(f'Overwrite `trainer.callbacks`, was "{_trainer_params.checkpoint_callback}"')
    _trainer_params_callbacks = []

    if _use_best_epoch:
        checkpoint_callback = ModelCheckpoint(monitor=model.metric_name, mode='max')
        logger.info(f'Create ModelCheckpoint callback with monitor="{model.metric_name}"')
        _trainer_params_callbacks.append(checkpoint_callback)

    if _trainer_params.get('checkpoints_every_n_val_epochs', False):
        every_n_val_epochs = _trainer_params.checkpoints_every_n_val_epochs
        checkpoint_callback = ModelCheckpoint(every_n_epochs=every_n_val_epochs, save_top_k=-1)
        logger.info(f'Create ModelCheckpoint callback every_n_epochs ="{every_n_val_epochs}"')
        _trainer_params_callbacks.append(checkpoint_callback)

        if 'checkpoint_callback' in _trainer_params:
            del _trainer_params.checkpoint_callback
        if 'enable_checkpointing' in _trainer_params:
            del _trainer_params.enable_checkpointing
        del _trainer_params.checkpoints_every_n_val_epochs

    if 'logger_name' in conf:
        _trainer_params_additional['logger'] = tb_logger

    lr_monitor = LearningRateMonitor(logging_interval='step')
    _trainer_params_callbacks.append(lr_monitor)

    if len(_trainer_params_callbacks) > 0:
        _trainer_params_additional['callbacks'] = _trainer_params_callbacks

    trainer = pl.Trainer(**_trainer_params, **_trainer_params_additional)
    trainer.fit(model, dm)

    if 'model_path' in conf:
        if _use_best_epoch:
            model.load_from_checkpoint(checkpoint_callback.best_model_path)
            torch.save(model.state_dict(), conf.model_path)
            logging.info(f'Best model stored in "{checkpoint_callback.best_model_path}" '
                         f'and copied to "{conf.model_path}"')
        else:
            torch.save(model.state_dict(), conf.model_path)
            logger.info(f'Model weights saved to "{conf.model_path}"')
    
    if conf.get('supervised_settings', False):
        result = trainer.test(model=model, dataloaders=dm.test_dataloader())[0]['test/AUROC']
        scores = {}
        scores['target'] = result
        scores['name_exp'] = conf.logger_name
        pd.DataFrame(scores).to_csv(conf.inference.output)



def inference_module(conf):
    
    dataset_inf_train = hydra.utils.instantiate(conf.inference.dataset_train)
    dataset_inf_test = hydra.utils.instantiate(conf.inference.dataset_test)

    collate_fn = partial(
        collate_feature_dict_with_target,
        col_id=conf.inference.col_id,
        target_col_names=conf.inference.target_col_names
     )
    
    inference_train_dl = DataLoader(
        dataset=dataset_inf_train,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=conf.inference.get('num_workers', 0),
        batch_size=conf.inference.get('batch_size', 128),
    )

    inference_test_dl = DataLoader(
        dataset=dataset_inf_test,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=conf.inference.get('num_workers', 0),
        batch_size=conf.inference.get('batch_size', 128),
    )
    
    model = hydra.utils.instantiate(conf.pl_module)
    if conf.get('use_save_model', True):
        model.load_state_dict(torch.load(conf.model_path))

    inf_module = InferenceModuleMultimodal(
        model=model,
        pandas_output=True,
        col_id=conf.inference.col_id,
        target_col_names=conf.inference.target_col_names,
        model_type=conf.get('model_type', 'notab')
    )

    trainer = pl.Trainer(accelerator='gpu', max_epochs=-1, fast_dev_run=conf.trainer.fast_dev_run)
    
    path_output = Path(conf.inference.output.path)
    
    if path_output.exists() is False:
        os.mkdir(conf.inference.output.path)

    logger.info('Inference train...')
    inf_train_embeddings = pd.concat(
        trainer.predict(inf_module, inference_train_dl)
    )
    
    logger.info('Save train embeddings...')
    inf_train_embeddings.to_parquet(f"{conf.inference.output.path}/train.parquet", index=False, engine="pyarrow", compression="snappy")
    del inf_train_embeddings
    
    logger.info('Inference test...')
    inf_test_embeddings = pd.concat(
        trainer.predict(inf_module, inference_test_dl)
    )
    
    logger.info('Save test embeddings...')
    inf_test_embeddings.to_parquet(f"{conf.inference.output.path}/test.parquet", index=False, engine="pyarrow", compression="snappy")
    del inf_test_embeddings

def downstream(conf):
    metric = hydra.utils.instantiate(conf.downstream.metric)
    downstream = Downstream(
        metric=metric,
        train_path=f"{conf.inference.output.path}/train.parquet",
        test_path=f"{conf.inference.output.path}/test.parquet",
        col_id=conf.inference.col_id,
        model_conf=conf.downstream.model,
        logger=logger,
        name_exp=conf.get('logger_name'),
        result_path=conf.downstream.output,
        target_col_names=conf.inference.target_col_names
    )

    scores = downstream.run()
    return scores
    

@hydra.main(version_base='1.2', config_path=None)
def main(conf: DictConfig):
    if conf.get('training_run', True):
        logger.info('Start train...')
        train_module(conf)
    if  conf.get('inference_run', True):
        logger.info('Start inference...')
        inference_module(conf)
    if conf.get('downstream_run', True):
        logger.info('Start downstream...')
        scores = downstream(conf)
    return scores
    
if __name__ == '__main__':
    main()








