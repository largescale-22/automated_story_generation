python guideline.py \
  --model_name_or_path bert-base-uncased \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir tmp/swag/