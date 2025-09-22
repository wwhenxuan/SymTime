python -u fine_tuning.py \
  --task_name "imputation" \
  --dataset_name "ETTm2" \
  --data "ETTm2" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTm2.csv" \
  --mask_rate 0.125 \
  --moving_avg 1 \
  --forward_layers 4 \
  --individual 0 \
  --stride 1 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --lradj "type2" \

python -u fine_tuning.py \
  --task_name "imputation" \
  --dataset_name "ETTm2" \
  --data "ETTm2" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTm2.csv" \
  --mask_rate 0.25 \
  --moving_avg 1 \
  --forward_layers 2 \
  --individual 0 \
  --stride 1 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --lradj "type2" \

python -u fine_tuning.py \
  --task_name "imputation" \
  --dataset_name "ETTm2" \
  --data "ETTm2" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTm2.csv" \
  --mask_rate 0.375 \
  --moving_avg 1 \
  --forward_layers 2 \
  --individual 0 \
  --stride 1 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --lradj "type2" \

python -u fine_tuning.py \
  --task_name "imputation" \
  --dataset_name "ETTm2" \
  --data "ETTm2" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTm2.csv" \
  --mask_rate 0.50 \
  --moving_avg 1 \
  --forward_layers 2 \
  --individual 0 \
  --stride 1 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --lradj "type2" \
