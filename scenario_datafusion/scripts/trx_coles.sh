export FOLD=4
python ptls_train_inf_down_module.py \
        fold=${FOLD} \
        "dataset_unsupervised.train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'"\
        "inference.dataset_train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'" \
            --config-dir scenario_datafusion/conf --config-name trx_coles