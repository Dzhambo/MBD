for FOLD in 0 1 2 3 4
do
    echo ${FOLD}
    python run_late_fusion.py \
        logger_name=lf_coles_${FOLD} \
        "late_fusion.train1_path='scenario_mbd/data/embeddings/trx_coles_${FOLD}/train.parquet'" \
        "late_fusion.test1_path='scenario_mbd/data/embeddings/trx_coles_${FOLD}/test.parquet'" \
        "late_fusion.train2_path='scenario_mbd/data/embeddings/click_coles_${FOLD}/train.parquet'" \
        "late_fusion.test2_path='scenario_mbd/data/embeddings/click_coles_${FOLD}/test.parquet'" \
            --config-dir scenario_mbd/conf --config-name late_fusion

    python run_late_fusion.py \
        logger_name=lf_agg_${FOLD} \
        "late_fusion.train1_path='scenario_mbd/data/embeddings/trx_agg_${FOLD}/train.parquet'" \
        "late_fusion.test1_path='scenario_mbd/data/embeddings/trx_agg_${FOLD}/test.parquet'" \
        "late_fusion.train2_path='scenario_mbd/data/embeddings/click_agg_${FOLD}/train.parquet'" \
        "late_fusion.test2_path='scenario_mbd/data/embeddings/click_agg_${FOLD}/test.parquet'" \
            --config-dir scenario_mbd/conf --config-name late_fusion
    
    python run_late_fusion.py \
        logger_name=lf_tabgpt_${FOLD} \
        "late_fusion.train1_path='scenario_mbd/data/embeddings/trx_tabgpt_${FOLD}/train.parquet'" \
        "late_fusion.test1_path='scenario_mbd/data/embeddings/trx_tabgpt_${FOLD}/test.parquet'" \
        "late_fusion.train2_path='scenario_mbd/data/embeddings/click_tabgpt_${FOLD}/train.parquet'" \
        "late_fusion.test2_path='scenario_mbd/data/embeddings/click_tabgpt_${FOLD}/test.parquet'" \
            --config-dir scenario_mbd/conf --config-name late_fusion
    
    python run_late_fusion.py \
        logger_name=lf_tabbert_${FOLD} \
        "late_fusion.train1_path='scenario_mbd/data/embeddings/trx_tabbert_${FOLD}/train.parquet'" \
        "late_fusion.test1_path='scenario_mbd/data/embeddings/trx_tabbert_${FOLD}/test.parquet'" \
        "late_fusion.train2_path='scenario_mbd/data/embeddings/click_tabbert_${FOLD}/train.parquet'" \
        "late_fusion.test2_path='scenario_mbd/data/embeddings/click_tabbert_${FOLD}/test.parquet'" \
            --config-dir scenario_mbd/conf --config-name late_fusion
done