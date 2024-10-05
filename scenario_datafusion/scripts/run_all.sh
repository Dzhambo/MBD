for FOLD in 0 1 2 3 4
do
    echo ${FOLD} 
   python ptls_train_inf_down_module.py \
       fold=${FOLD} \
       "dataset_unsupervised.train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'"\
       "inference.dataset_train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'" \
           --config-dir scenario_datafusion/conf --config-name trx_coles
    
   python ptls_train_inf_down_module.py \
       fold=${FOLD} \
       "dataset_unsupervised.train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'"\
           --config-dir scenario_datafusion/conf --config-name trx_supervised

    python ptls_train_inf_down_module.py \
        fold=${FOLD} \
        "dataset_unsupervised.train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'"\
        "inference.dataset_train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'" \
            --config-dir scenario_datafusion/conf --config-name trx_tabbert

    python ptls_train_inf_down_module.py \
        fold=${FOLD} \
        "dataset_unsupervised.train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'"\
        "inference.dataset_train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'" \
            --config-dir scenario_datafusion/conf --config-name trx_tabgpt

   python ptls_train_inf_down_module.py \
        fold=${FOLD} \
        "dataset_unsupervised.train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'"\
        "inference.dataset_train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'" \
          --config-dir scenario_datafusion/conf --config-name trx_agg


    python ptls_train_inf_down_module.py \
        fold=${FOLD} \
        "dataset_unsupervised.train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'"\
        "inference.dataset_train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'" \
            --config-dir scenario_datafusion/conf --config-name click_coles
    
    python ptls_train_inf_down_module.py \
        fold=${FOLD} \
        "dataset_unsupervised.train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'"\
            --config-dir scenario_datafusion/conf --config-name click_supervised

    python ptls_train_inf_down_module.py \
        fold=${FOLD} \
        "dataset_unsupervised.train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'"\
        "inference.dataset_train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'" \
            --config-dir scenario_datafusion/conf --config-name click_tabbert

    python ptls_train_inf_down_module.py \
        fold=${FOLD} \
        "dataset_unsupervised.train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'"\
        "inference.dataset_train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'" \
            --config-dir scenario_datafusion/conf --config-name click_tabgpt

    python ptls_train_inf_down_module.py \
         fold=${FOLD} \
         "dataset_unsupervised.train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'"\
         "inference.dataset_train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'" \
            --config-dir scenario_datafusion/conf --config-name click_agg

    python ptls_train_inf_down_module.py \
        fold=${FOLD} \
        "dataset_unsupervised.train.data_files.${FOLD}='/home/jovyan/mollaev/experiments/MBD/scenario_datafusion/data/mm_dataset_fold/fold=5'"\
            --config-dir scenario_datafusion/conf --config-name mm_supervised
done