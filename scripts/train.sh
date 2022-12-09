#Train JointIDSF
export lr=3e-5
export coef=0.5
export seed=42
echo "${lr}"
export MODEL_DIR=JointIDSF_XLM-Rencoder
export MODEL_DIR=$MODEL_DIR"/"$lr"/"$coef"/"$seed
echo "${MODEL_DIR}"
python3 train.py --task ner \
                  --token_level syllable-level \
                  --model_type xlmr \
                  --model_dir $MODEL_DIR \
                  --data_dir data/ \
                  --train_type augment_train \
                  --val_type public_test \
                  --test_type private_test \
                  --intent_embedding_path data/syllable-level/intent_embedding.pt \
                  --seed $seed \
                  --do_train \
                  --eval_train \
                  --eval_dev \
                  --eval_test \
                  --save_steps 69 \
                  --logging_steps 69 \
                  --num_train_epochs 500 \
                  --tuning_metric mean_intent_slot \
                  --use_crf \
                  --gpu_id 0 \
                  --intent_loss_coef $coef \
                  --learning_rate $lr \
                  --max_seq_len 32 \
                  --train_batch_size 64 \
                  --eval_batch_size 64 \
                  --dropout_rate 0.2 \
                  --early_stopping 15 \
