for FOLD in 0 1 2 3 4
do
    echo ${FOLD} 
    python ptls_train_inf_down_module.py \
        fold=${FOLD} \
        "dataset_unsupervised.train.data_files.${FOLD}='scenario_mbd/data/mm_dataset_fold/fold=5'"\
            --config-dir scenario_mbd/conf --config-name matching_conf_trxgeo

    python ptls_train_inf_down_module.py \
        fold=${FOLD} \
        "dataset_unsupervised.train.data_files.${FOLD}='scenario_mbd/data/mm_dataset_fold/fold=5'"\
            --config-dir scenario_mbd/conf --config-name matching_conf_trxdial
    
    python ptls_train_inf_down_module.py \
        fold=${FOLD} \
        "dataset_unsupervised.train.data_files.${FOLD}='scenario_mbd/data/mm_dataset_fold/fold=5'"\
            --config-dir scenario_mbd/conf --config-name matching_conf_geodial

done