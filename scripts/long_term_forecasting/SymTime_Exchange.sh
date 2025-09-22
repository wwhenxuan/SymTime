python -u fine_tuning.py \
  --task_name "long_term_forecast" \
  --model_id "Exchange" \
  --data "custom" \
  --root_path "./datasets/exchange_rate/" \
  --data_path "exchange_rate.csv" \
  --seq_len 96 \
  --pred_len 96 \
  --moving_avg True \
  --forward_layers 3 \
  --stride 8 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj "type2" \
  --num_workers 5 \
  --train_epochs 16 \
  --patience 4 \


python -u fine_tuning.py \
  --task_name "long_term_forecast" \
  --model_id "Exchange" \
  --data "custom" \
  --root_path "./datasets/exchange_rate/" \
  --data_path "exchange_rate.csv" \
  --seq_len 96 \
  --pred_len 192 \
  --moving_avg True \
  --forward_layers 3 \
  --stride 4 \
  --batch_size 4 \
  --learning_rate 0.0001 \
  --lradj "type1" \
  --num_workers 5 \
  --train_epochs 16 \
  --patience 4 \

python -u fine_tuning.py \
  --task_name "long_term_forecast" \
  --model_id "Exchange" \
  --data "custom" \
  --root_path "./datasets/exchange_rate/" \
  --data_path "exchange_rate.csv" \
  --seq_len 96 \
  --pred_len 336 \
  --moving_avg True \
  --forward_layers 3 \
  --stride 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj "type1" \
  --num_workers 5 \
  --train_epochs 16 \
  --patience 4 \

python -u fine_tuning.py \
  --task_name "long_term_forecast" \
  --model_id "Exchange" \
  --data "custom" \
  --root_path "./datasets/exchange_rate/" \
  --data_path "exchange_rate.csv" \
  --seq_len 96 \
  --pred_len 720 \
  --moving_avg True \
  --forward_layers 3 \
  --stride 8 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --lradj "type1" \
  --num_workers 5 \
  --train_epochs 16 \
  --patience 4 \
