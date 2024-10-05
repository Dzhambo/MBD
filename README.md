# Multimodal Banking Dataset: Understanding Client Needs through Event Sequences

## Abstract

Financial organizations collect a huge amount of data about clients that typically has a temporal (sequential) structure and is collected from various sources (modalities). Due to privacy issues, there are no large-scale open-source multimodal datasets of event sequences, which significantly limits the research in this area. In this paper, we present the industrial-scale publicly available multimodal banking dataset, MBD, that contains more than 2M corporate clients with several modalities: 950M bank transactions, 1B geo position events, 5M embeddings of dialogues with technical support and monthly aggregated purchases of four bank’s products. All entries are properly anonymized from real proprietary bank data. Using this data, we introduce a novel multimodal benchmark that incorporates two open-source finansical datasets. We provide numerical results demonstrating our multimodal baselines’ superiority over single-modal techniques for each task. As a result, the proposed dataset and benchmark can open new perspectives and facilitate the future development of practically important large-scale multimodal algorithms for event sequences. 

## Benchmark

This repository contains code to reproduce the benchmark results obtained in the paper.

## Structure of repo

- **scenario_mbd** - all experinemts for MBD dataset
- **scenario_datafusion** - all experinemts for Datafusion dataset

    - **conf** - all configuration files
    - **modules** - common code for the model pipelines
    - **models** - models are saved here.
    - **data** - store for datasets and embeddings from different methods
    - **notebooks** 
        - **Preprocessing** - data preprocessing for each modality for subsequent training and testing
        For MBD dataset:
            - **Geo**, **Trx**, **Dialogs** - unimodal experiments for various modalities are presented in Jupyter notebooks
            - **Fusion**
                - late fusion approaches: Concatenation and Blending.
                - Python and PySpark code for training Gradient Boosting models are provided. 
                - Multimodal matching task

    - **scripts**  
        - `*.sh` files with configured pipeline runs


## Hardware and software requirements

You should use a server with at least one GPU 32GB, 4 CPU cores, 96GB RAM.


Nvidia drivers should be installed.

Since benchmark are using spark, java must be installed.

We use the following frameworks, which are installed along with the python dependencies:
- [torch](https://pytorch.org/)
- [pytorch-lightning](https://lightning.ai/)
- [hydra](https://hydra.cc/docs/intro/)
- [pytorch-lifestream](https://github.com/dllllb/pytorch-lifestream)

## Environment installation

Use `python3.8` virtual environment. Install the required modules using the command:

```
pip install -r requirements.txt
```

## Benchmark run

Please note that all data and results will be stored in the root of the repository.

1. Load datasets
    - For MBD dataset follow the [temporal url](https://disk.yandex.ru/d/Pk9Mhx70VnUzbA) and download the data.
    - For Datafusion dataset use 'sh scenario_datafusion/scripts/get_data.sh'
2. Start with preprocessing data in 'notebooks/Preprocessing'. The preprocessed data will be stored as a separate parquet files.
4. To conduct experiments, you need to execute bash scripts. For instance, to run the CoLES method for the transactions modality, use the following command:
   ```
   sh scenarion_mbd/scripts/trx_coles.sh
   sh scenarion_datafusion/scripts/trx_coles.sh
   ```

To conduct all experiments use 
   ```
   sh scenarion_mbd/scripts/run_all.sh
   sh scenarion_datafusion/scripts/run_all.sh
   ```

   ```
   sh scenarion_mbd/scripts/run_matching_all.sh
   sh scenarion_datafusion/scripts/run_matching_all.sh
   ```

   ```
   sh scenarion_mbd/scripts/run_latefusion_all.sh
   sh scenarion_datafusion/scripts/run_latefusion_all.sh
   ```

   Alternatively, for MBD dataset you can use the corresponding Jupyter notebook in **notebooks**.

   Preprocessed data will be used. As a result, pretrained embeddings will be saved as a separate parquet files. Metrics will be saved into csv files.
   
5. For the late fusion approaches and matching task use 'notebooks/Fusion' or corresponding scripts.

   Pretrained embeddings will be used. Metrics will be saved into csv files.

For reproducability of experiments use PySpark ML.




