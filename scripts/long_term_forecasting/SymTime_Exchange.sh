export CUDA_VISIBLE_DEVICES=1


seq_len = 336


python -u long_term_forecast.py \ 
  --task_name "long_term_forecast" \ 
  --dataset_name "Exchange" \ 
  --data "custom" \ 
  --root_path "./datasets/exchange_rate/" \ 
  --data_path "exchange_rate.csv" \ 
  --seq_len $seq_len \ 
  --pred_len 96 \ 
  --forward_layers 3 \ 
  --stride 4 \ 
  --batch_size 4 \ 
  --learning_rate 0.001 \ 
  --lradj "cosine" \ 
  --num_workers 5 \ 
  --enc_in 8 \ 
  --train_epochs 16 \ 
  --patience 3 \ 


python -u long_term_forecast.py \ 
  --task_name "long_term_forecast" \ 
  --dataset_name "Exchange" \ 
  --data "custom" \ 
  --root_path "./datasets/exchange_rate/" \ 
  --data_path "exchange_rate.csv" \ 
  --seq_len $seq_len \ 
  --pred_len 96 \ 
  --forward_layers 3 \ 
  --stride 4 \ 
  --batch_size 4 \ 
  --learning_rate 0.001 \ 
  --lradj cosine \ 
  --num_workers 5 \ 
  --enc_in 8 \ 
  --train_epochs 16 \ 
  --patience 3 \ 


python -u long_term_forecast.py \ 
  --task_name "long_term_forecast" \ 
  --dataset_name "Exchange" \ 
  --data "custom" \ 
  --root_path "./datasets/exchange_rate/" \ 
  --data_path "exchange_rate.csv" \ 
  --seq_len $seq_len \ 
  --pred_len 96 \ 
  --forward_layers 3 \ 
  --stride 8 \ 
  --batch_size 8 \ 
  --learning_rate 0.00075 \ 
  --lradj "type1" \ 
  --num_workers 5 \ 
  --enc_in 8 \ 
  --train_epochs 16 \ 
  --patience 3 \ 


python -u long_term_forecast.py \ 
  --task_name "long_term_forecast" \ 
  --dataset_name "Exchange" \ 
  --data "custom" \ 
  --root_path "./datasets/exchange_rate/" \ 
  --data_path "exchange_rate.csv" \ 
  --seq_len $seq_len \ 
  --pred_len 96 \ 
  --forward_layers 3 \ 
  --stride 8 \ 
  --batch_size 4 \ 
  --learning_rate 0.001 \ 
  --lradj "cosine" \ 
  --num_workers 5 \ 
  --enc_in 8 \ 
  --train_epochs 16 \ 
  --patience 3 \ 