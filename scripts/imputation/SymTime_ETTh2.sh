export CUDA_VISIBLE_DEVICES=0

forward_layers=2
stride=1
batch_size=4
lr=0.0001


python -u imputation.py \
  --task_name "imputation" \
  --dataset_name "ETTh2" \
  --data "ETTh2" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTh2.csv" \
  --mask_rate 0.125 \
  --forward_layers $forward_layers \
  --stride 8 \
  --batch_size $batch_size \
  --learning_rate $lr \
  --lradj "cosine" \


python -u imputation.py \
  --task_name "imputation" \
  --dataset_name "ETTh2" \
  --data "ETTh2" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTh2.csv" \
  --mask_rate 0.25 \
  --forward_layers $forward_layers \
  --stride $stride \
  --batch_size 64 \
  --learning_rate $lr \
  --lradj "cosine" \


python -u imputation.py \
  --task_name "imputation" \
  --dataset_name "ETTh2" \
  --data "ETTh2" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTh2.csv" \
  --mask_rate 0.375 \
  --forward_layers $forward_layers \
  --stride $stride \
  --batch_size $batch_size \
  --learning_rate $lr \
  --lradj "type1" \


python -u imputation.py \
  --task_name "imputation" \
  --dataset_name "ETTh2" \
  --data "ETTh2" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTh2.csv" \
  --mask_rate 0.50 \
  --forward_layers 3 \
  --stride $stride \
  --batch_size $batch_size \
  --learning_rate $lr \
  --lradj "type1" \
