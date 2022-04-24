DIVECE=$1
MODEL=$2

CUDA_VISIBLE_DEVICES=$DEVICE python guideline.py \
    --model_type $MODEL \
    --model_name_or_path "$MODEL-base-uncased" \
    --output_dir "outputs_guideline/$MODEL" \
    --cache_dir caches \
    --max_seq_length 1000 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --do_lower_case \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --warmup_steps 0 \
    --fp16 \
    --overwrite_output_dir \