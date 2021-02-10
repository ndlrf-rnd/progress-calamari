set -ex

export CUDA_VISIBLE_DEVICES=0
export DATASET_DIR="$(pwd)/output/calamari/lines"
export OUTPUT_DIR="$(pwd)/output/models/calamari/calamari-lh_48-e_100-lstm_200-dropout_0.5-lr_0.001"

calamari-train  \
    --network 'cnn=40:3x3,pool=2x2,cnn=60:3x3,pool=2x2,lstm=200,dropout=0.5,learning_rate=0.001' \
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