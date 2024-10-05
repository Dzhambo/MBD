from modules.late_fusion import LateFusion
from omegaconf import DictConfig
import logging
import hydra
import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import os
import lightgbm as ltb

logger = logging.getLogger(__name__)

def downstream(conf):
    metric = hydra.utils.instantiate(conf.metric)
    downstream = LateFusion(
        metric=metric,
        train1_path=conf.late_fusion.train1_path,
        test1_path=conf.late_fusion.test1_path,
        train2_path=conf.late_fusion.train2_path,
        test2_path=conf.late_fusion.test2_path,
        col_id=conf.col_id,
        params=conf.model,
        logger=logger,
        name_exp=conf.logger_name,
        result_path=conf.output,
        target_col_names=conf.target_col_names
    )

    scores = downstream.run()
    return scores

@hydra.main(version_base='1.2', config_path=None)
def main(conf: DictConfig):
    logger.info('Start downstream...')
    scores = downstream(conf)
    return scores
    
if __name__ == '__main__':
    main()   