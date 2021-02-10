set -ex
export CUDA_VISIBLE_DEVICES=0
export DEFAULT_DATASET_DIR="$(pwd)/output/calamari/lines"
export DATASET_DIR="${1:-${DEFAULT_DATASET_DIR}}"
# export DEFAULT_OUTPUT_DIR="$(pwd)/output/models/calamari/calamari-n_folds_1-bs_8-lstm_650-lh_48"
export DEFAULT_OUTPUT_DIR="$(pwd)/output/models/calamari/calamari-n_folds_5-bs_8-lstm_200-lh_48"

export OUTPUT_DIR="${2:-${DEFAULT_OUTPUT_DIR}}"
calamari-cross-fold-train  \
    --best_models_dir "${OUTPUT_DIR}/" \
    --temporary_dir "${OUTPUT_DIR}__temp/" \
    --files "${DATASET_DIR}/train/*.png" \
    --n_folds 5 \
    --max_parallel_models 5 \
    --batch_size 8 \
    --checkpoint_frequency 10 
# --network 'cnn=8:3x3,cnn=16:3x3,cnn=32:3x3,cnn=64:3x3,pool=2x2,cnn=128:3x3,pool=2x2,lstm=650,dropout=0.5,learning_rate=0.001' \
#     --epoch 100 \
#    --output_dir "${OUTPUT_DIR}"
# --validation_dataset FILE \
# --validation "${DATASET_DIR}/test/\*.png" \
# --samples_per_epoch 1 \
# --single_fold 1 \
# --early_stopping_frequency 5 \
# --early_stopping_nbest 10 \
# --early_stopping_best_model_output_dir "${OUTPUT_DIR}" \
# --early_stopping_best_model_prefix best \

# --train_data_on_the_fly \
# --validation_data_on_the_fly \
	# --num_threads 8 \
