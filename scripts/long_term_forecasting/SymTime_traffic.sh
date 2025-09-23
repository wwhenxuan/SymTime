export CUDA_VISIBLE_DEVICES=4


seq_len = 336
stride = 8
batch_size = 4
lr = 0.0001
lradj = "type1"


python -u long_term_forecast.py \
  --task_name "long_term_forecast" \
  --dataset_name "traffic" \
  --data "custom" \
  --root_path "./datasets/traffic/" \
  --data_path "traffic.csv" \
  --seq_len $seq_len \
  --pred_len 96 \
  --forward_layers 3 \
  --stride $stride \
  --batch_size $batch_size \
  --learning_rate $lr \
  --lradj $lradj \
  --num_workers 5 \
  --enc_in 862 \
  --train_epochs 16 \
  --patience 3 \


python -u long_term_forecast.py \
  --task_name "long_term_forecast" \
  --dataset_name "traffic" \
  --data "custom" \
  --root_path "./datasets/traffic/" \
  --data_path "traffic.csv" \
  --seq_len $seq_len \
  --pred_len 192 \
  --forward_layers 3 \
  --stride $stride \
  --batch_size $batch_size \
  --learning_rate $lr \
  --lradj $lradj \
  --num_workers 5 \
  --enc_in 862 \
  --train_epochs 16 \
  --patience 3 \


python -u long_term_forecast.py \
  --task_name "long_term_forecast" \
  --dataset_name "traffic" \
  --data "custom" \
  --root_path "./datasets/traffic/" \
  --data_path "traffic.csv" \
  --seq_len $seq_len \
  --pred_len 336 \
  --forward_layers 3 \
  --stride $stride \
  --batch_size $batch_size \
  --learning_rate $lr \
  --lradj $lradj \
  --num_workers 5 \
  --enc_in 862 \
  --train_epochs 16 \
  --patience 3 \


python -u long_term_forecast.py \
  --task_name "long_term_forecast" \
  --dataset_name "traffic" \
  --data "custom" \
  --root_path "./datasets/traffic/" \
  --data_path "traffic.csv" \
  --seq_len $seq_len \
  --pred_len 720 \
  --forward_layers 3 \
  --stride $stride \
  --batch_size $batch_size \
  --learning_rate $lr \
  --lradj $lradj \
  --num_workers 5 \
  --enc_in 862 \
  --train_epochs 16 \
  --patience 3 \