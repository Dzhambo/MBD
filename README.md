# Multimodal Banking Dataset: Understanding Client Needs through Event Sequences


## Abstract

Financial organizations collect a huge amount of data about clients that typically has a temporal (sequential) structure and is collected from various sources (modalities). Due to privacy issues, there are no large-scale open-source multimodal datasets of event sequences, which significantly limits the research in this area.

In this paper, we present the industrial-scale publicly available multimodal banking dataset, MBD, that contains more than 1M clients with several modalities: 215M bank transactions, 670M geo position events, 1.5M embeddings of dialogues with technical support and monthly aggregated purchases of four bank's products. All entries are properly anonymized from real proprietary bank data. Using this dataset, we introduce a novel benchmark with two business tasks: campaigning (purchase prediction in the next month) and matching of clients. We provide numerical results that demonstrate the superiority of our multi-modal baselines over single-modal techniques for each task. As a result, the proposed dataset can open new perspectives and facilitate the future development of practically important large-scale multimodal algorithms for event sequences.

## Structure of repo

- **notebooks** 
    - **Preprocessing** - data preprocessing for each modality for subsequent training and testing
    - **Geo**, **Trx**, **Dialogs** - unimodal experiments for various modalities are presented in Jupyter notebooks
    - **Fusion** - 
        - late fusion approaches: Concatenation and Blending.
        - Python and PySpark code for training Gradient Boosting models are provided. 
        - Multimodal matching task

- **scripts**  
    - we use a hydra-based experiment configuration structure.
    - run unimodal experiments for methods: Aggregation, CoLES, TabBert, TabGPT, for modalities geostream and transactions
    - run experiments for late fusion for pair modalities


## Usage

1. Start with preprocessing data in 'notebooks/Preprocessing'
2. To conduct experiments, you need to execute bash scripts. For instance, to run the CoLES method for the transactions modality, use the following command: **bash trx_coles.sh**. Alternatively, you can use the corresponding Jupyter notebook in **notebooks**.
3. For the late fusion approaches and matching task use 'notebooks/Fusion'
4. To obtain the results, execute the following command: **bash evaluate.sh**, specifying the desired file with scores.

## Environment

pip install -r requirements.txt



