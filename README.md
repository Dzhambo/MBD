# Multimodal Banking Dataset: Understanding Client Needs through Event Sequences

## Abstract

Financial organizations collect a huge amount of data about clients that typically has a temporal (sequential) structure and is collected from various sources (modalities). Due to privacy issues, there are no large-scale open-source multimodal datasets of event sequences, which significantly limits the research in this area.

In this paper, we present the industrial-scale publicly available multimodal banking dataset, MBD, that contains more than 1M clients with several modalities: 215M bank transactions, 670M geo position events, 1.5M embeddings of dialogues with technical support and monthly aggregated purchases of four bank's products. All entries are properly anonymized from real proprietary bank data. Using this dataset, we introduce a novel benchmark with two business tasks: campaigning (purchase prediction in the next month) and matching of clients. We provide numerical results that demonstrate the superiority of our multi-modal baselines over single-modal techniques for each task. As a result, the proposed dataset can open new perspectives and facilitate the future development of practically important large-scale multimodal algorithms for event sequences.

## Benchmark

This repository contains code to reproduce the benchmark results obtained in the paper.

## Structure of repo

- **conf** - all configuration files
- **modules** - common code for the model pipelines
- **notebooks** 
    - **Preprocessing** - data preprocessing for each modality for subsequent training and testing
    - **Geo**, **Trx**, **Dialogs** - unimodal experiments for various modalities are presented in Jupyter notebooks
    - **Fusion**
        - late fusion approaches: Concatenation and Blending.
        - Python and PySpark code for training Gradient Boosting models are provided. 
        - Multimodal matching task

- **scripts**  
    - `*.py` files with different pipelines
    - `*.sh` files with configured pipeline runs

<!--   - we use a hydra-based experiment configuration structure.
    - run unimodal experiments for methods: Aggregation, CoLES, TabBert, TabGPT, for modalities geostream and transactions
    - run experiments for late fusion for pair modalities 
-->

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

1. Follow the [temporal url](https://disk.yandex.ru/d/Pk9Mhx70VnUzbA) and download the data.
2. Place the loaded data into the repository root. You will get the following folder structure:
    ```
    \dial_test.parquet\
    \dial_train.parquet\
    \geo_test.parquet\
    ...
    \trx_train.parquet\
    ```
3. Start with preprocessing data in 'notebooks/Preprocessing'. The preprocessed data will be stored as a separate parquet files.
4. To conduct experiments, you need to execute bash scripts. For instance, to run the CoLES method for the transactions modality, use the following command:
   ```
   bash trx_coles.sh
   ```

   Alternatively, you can use the corresponding Jupyter notebook in **notebooks**.

   Preprocessed data will be used. As a result, pretrained embeddings will be saved as a separate parquet files. Metrics will be saved into csv files.
   
5. For the late fusion approaches and matching task use 'notebooks/Fusion'.

   Pretrained embeddings will be used. Metrics will be saved into csv files.






