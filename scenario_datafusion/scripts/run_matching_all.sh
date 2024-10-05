for FOLD in 0 1 2 3 4
do
    echo ${FOLD} 
    python ptls_train_inf_down_module.py \
        fold=${FOLD} \
        "dataset_unsupervised.train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'"\
            --config-dir scenario_datafusion/conf --config-name matching_conf
done