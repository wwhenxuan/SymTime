python -u fine_tuning.py \
  --task_name "imputation" \
  --dataset_name "ETTm1" \
  --data "ETTm1" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTm1.csv" \
  --mask_rate 0.125 \
  --moving_avg True \
  --forward_layers 1 \
  --individual False \
  --stride 1 \
  --batch_size 64 \
  --learning_rate 0.0002 \
  --lradj "cosine" \

python -u fine_tuning.py \
  --task_name "imputation" \
  --dataset_name "ETTm1" \
  --data "ETTm1" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTm1.csv" \
  --mask_rate 0.25 \
  --moving_avg True \
  --forward_layers 2 \
  --individual False \
  --stride 1 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --lradj "cosine" \

python -u fine_tuning.py \
  --task_name "imputation" \
  --dataset_name "ETTm1" \
  --data "ETTm1" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTm1.csv" \
  --mask_rate 0.375 \
  --moving_avg True \
  --forward_layers 1 \
  --individual False \
  --stride 1 \
  --batch_size 64 \
  --learning_rate 0.0003 \
  --lradj "cosine" \

python -u fine_tuning.py \
  --task_name "imputation" \
  --dataset_name "ETTm1" \
  --data "ETTm1" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTm1.csv" \
  --mask_rate 0.50 \
  --moving_avg True \
  --forward_layers 1 \
  --individual False \
  --stride 1 \
  --batch_size 64 \
  --learning_rate 0.0002 \
  --lradj "cosine" \
