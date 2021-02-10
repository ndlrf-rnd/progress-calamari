set -ex

export CUDA_VISIBLE_DEVICES=0
export DATASET_DIR="$(pwd)/output/lines"
export OUTPUT_DIR="$(pwd)/output/models//calamari-lh_48-e_100-cnn_128-lstm_650-dropout_e1.1-lr_e4"

calamari-train  \
    --network 'cnn=8:3x3,cnn=16:3x3,cnn=32:3x3,cnn=64:3x3,pool=2x2,cnn=128:3x3,pool=2x2,lstm=650,dropout=0.1,learning_rate=0.0001' \
    --files "${DATASET_DIR}/train/*.png" \
    --validation "${DATASET_DIR}/test/*.png" \
    --epoch 100 \
    --validation_dataset FILE \
    --batch_size 8 \
    --checkpoint_frequency 5 \
    --num_threads 8 \
    --display 1 \
    --output_dir "${OUTPUT_DIR}/" \
    --early_stopping_frequency 5 \
    --early_stopping_nbest 10 \
    --early_stopping_best_model_output_dir "${OUTPUT_DIR}/" \
    --early_stopping_best_model_prefix 'best'

# --train_data_on_the_fly \
# --validation_data_on_the_fly \
