export CUDA_VISIBLE_DEVICES = 0 1 2 3 4 5 6 7

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=8  \
  pretrain.py \
    --model "base" \
    --context_window 256 \
    --time_mask_ratio 0.40 \
    --sym_mask_ratio 0.15 \
    --patch_len 16 \
    --stride 16 \
    --data_path "./datasets/pretrain_data/" \
    --save_path "./checkpoints/SymTime_pretrain/" \
    --number 50 \
    --llm_name "DistilBert" \
    --num_epochs 256 \
    --warmup_epochs 32 \
    --batch_size 128 \
    --num_workers 1 \
    --optimizer "AdamW" \
    --criterion "MSE" \
    --warmup "LinearLR" \
    --scheduler "OneCycle" \
    --learning_rate 1e-5 \
