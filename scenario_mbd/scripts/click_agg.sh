export FOLD=4
python ptls_train_inf_down_module.py \
        fold=${FOLD} \
        "dataset_unsupervised.train.data_files.${FOLD}='scenario_mbd/data/mm_dataset_fold/fold=5'"\
        "inference.dataset_train.data_files.${FOLD}='scenario_mbd/data/mm_dataset_fold/fold=5'" \
        --config-dir scenario_mbd/conf --config-name click_agg