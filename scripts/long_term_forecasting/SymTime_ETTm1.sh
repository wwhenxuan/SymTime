export CUDA_VISIBLE_DEVICES=0

seq_len=336

python -u long_term_forecast.py \
  --task_name "long_term_forecast" \
  --dataset_name "ETTm1" \
  --data "ETTm1" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTm1.csv" \
  --seq_len $seq_len \
  --pred_len 96 \
  --forward_layers 3 \
  --stride 8 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj "type1" \
  --num_workers 5 \
  --enc_in 7 \
  --train_epochs 16 \
  --patience 3 \


python -u long_term_forecast.py \
  --task_name "long_term_forecast" \
  --dataset_name "ETTm1" \
  --data "ETTm1" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTm1.csv" \
  --seq_len $seq_len \
  --pred_len 192 \
  --forward_layers 3 \
  --stride 4 \
  --batch_size 4 \
  --learning_rate 0.001 \
  --lradj "type1" \
  --num_workers 5 \
  --enc_in 7 \
  --train_epochs 16 \
  --patience 3 \


python -u long_term_forecast.py \
  --task_name "long_term_forecast" \
  --dataset_name "ETTm1" \
  --data "ETTm1" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTm1.csv" \
  --seq_len $seq_len \
  --pred_len 336 \
  --forward_layers 3 \
  --stride 8 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --lradj "type1" \
  --num_workers 5 \
  --enc_in 7 \
  --train_epochs 16 \
  --patience 3 \


python -u long_term_forecast.py \
  --task_name "long_term_forecast" \
  --dataset_name "ETTm1" \
  --data "ETTm1" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTm1.csv" \
  --seq_len $seq_len \
  --pred_len 720 \
  --forward_layers 3 \
  --stride 4 \
  --batch_size 8 \
  --learning_rate 0.00075 \
  --lradj "type1" \
  --num_workers 5 \
  --enc_in 7 \
  --train_epochs 16 \
  --patience 3 \
