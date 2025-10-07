export CUDA_VISIBLE_DEVICES=0

forward_layers = 2
stride = 1
lr = 0.0001
lradj = "type1"


python -u imputation.py \ 
  --task_name "imputation" \ 
  --dataset_name "weather" \ 
  --data "weather" \ 
  --root_path "./datasets/weather/" \ 
  --data_path "weather.csv" \ 
  --mask_rate 0.125 \ 
  --forward_layers $forward_layers \ 
  --stride $stride \ 
  --batch_size 8 \ 
  --learning_rate $lr \ 
  --lradj $lradj \


python -u imputation.py \ 
  --task_name "imputation" \ 
  --dataset_name "weather" \ 
  --data "weather" \ 
  --root_path "./datasets/weather/" \ 
  --data_path "weather.csv" \ 
  --mask_rate 0.25 \ 
  --forward_layers $forward_layers \ 
  --stride $stride \ 
  --batch_size 32 \ 
  --learning_rate 0.00025 \ 
  --lradj $lradj \


python -u imputation.py \ 
  --task_name "imputation" \ 
  --dataset_name "weather" \ 
  --data "weather" \ 
  --root_path "./datasets/weather/" \ 
  --data_path "weather.csv" \ 
  --mask_rate 0.375 \ 
  --forward_layers $forward_layers \ 
  --stride $stride \ 
  --batch_size 4 \ 
  --learning_rate $lr \ 
  --lradj $lradj \


python -u imputation.py \ 
  --task_name "imputation" \ 
  --dataset_name "weather" \ 
  --data "weather" \ 
  --root_path "./datasets/weather/" \ 
  --data_path "weather.csv" \ 
  --mask_rate 0.50 \ 
  --forward_layers $forward_layers \ 
  --stride $stride \ 
  --batch_size 4 \ 
  --learning_rate $lr \ 
  --lradj "type2" \
