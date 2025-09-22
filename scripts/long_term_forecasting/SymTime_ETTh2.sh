export CUDA_VISIBLE_DEVICES=0

seq_len = 336

python -u long_term_forecast.py \
  --task_name "long_term_forecast" \
  --dataset_name "ETTh2" \
  --data "ETTh2" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTh2.csv" \
  --seq_len $seq_len \
  --pred_len 96 \
  --forward_layers 3 \
  --stride $stride \
  --batch_size $batch_size \
  --learning_rate $lr \
  --lradj $lradj \
  --num_workers 5 \
  --enc_in 7 \
  --train_epochs 16 \
  --patience 3 \


python -u long_term_forecast.py \
  --task_name "long_term_forecast" \
  --dataset_name "ETTh2" \
  --data "ETTh2" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTh2.csv" \
  --seq_len $seq_len \
  --pred_len 192 \
  --forward_layers 3 \
  --stride $stride \
  --batch_size $batch_size \
  --learning_rate $lr \
  --lradj $lradj \
  --num_workers 5 \
  --enc_in 7 \
  --train_epochs 16 \
  --patience 3 \


python -u long_term_forecast.py \
  --task_name "long_term_forecast" \
  --dataset_name "ETTh2" \
  --data "ETTh2" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTh2.csv" \
  --seq_len $seq_len \
  --pred_len 336 \
  --forward_layers 3 \
  --stride $stride \
  --batch_size $batch_size \
  --learning_rate $lr \
  --lradj $lradj \
  --num_workers 5 \
  --enc_in 7 \
  --train_epochs 16 \
  --patience 3 \


python -u long_term_forecast.py \
  --task_name "long_term_forecast" \
  --dataset_name "ETTh2" \
  --data "ETTh2" \
  --root_path "./datasets/ETT/" \
  --data_path "ETTh2.csv" \
  --seq_len $seq_len \
  --pred_len 720 \
  --forward_layers 3 \
  --stride $stride \
  --batch_size $batch_size \
  --learning_rate $lr \
  --lradj $lradj \
  --num_workers 5 \
  --enc_in 7 \
  --train_epochs 16 \
  --patience 3 \
