python -u fine_tuning.py \
  --task_name "imputation" \
  --dataset_name "ETTh2" \
  --data "ETTh2" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTh2.csv" \
  --mask_rate 0.125 \
  --moving_avg True \
  --forward_layers 2 \
  --individual False \
  --stride 8 \
  --batch_size 4 \
  --learning_rate 0.0001 \
  --lradj "cosine" \

python -u fine_tuning.py \
  --task_name "imputation" \
  --dataset_name "ETTh2" \
  --data "ETTh2" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTh2.csv" \
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
  --dataset_name "ETTh2" \
  --data "ETTh2" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTh2.csv" \
  --mask_rate 0.375 \
  --moving_avg True \
  --forward_layers 2 \
  --individual False \
  --stride 1 \
  --batch_size 4 \
  --learning_rate 0.0001 \
  --lradj "type1" \

python -u fine_tuning.py \
  --task_name "imputation" \
  --dataset_name "ETTh2" \
  --data "ETTh2" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTh2.csv" \
  --mask_rate 0.125 \
  --moving_avg True \
  --forward_layers 3 \
  --individual False \
  --stride 1 \
  --batch_size 4 \
  --learning_rate 0.0001 \
  --lradj "cosine" \
