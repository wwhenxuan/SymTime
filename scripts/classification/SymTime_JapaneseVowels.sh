export CUDA_VISIBLE_DEVICES=3


python -u classification.py \
  --task_name "classification" \
  --dataset_name "JapaneseVowels" \
  --data "UEA" \
  --root_path "./datasets/JapaneseVowels" \
  --forward_layers 1 \
  --out_channels 64 \
  --stride 16 \
  --batch_size 12 \
  --learning_rate 0.0005 \
  --num_workers 0 \
  --lradj "cosine" \
  --patience 10 \
