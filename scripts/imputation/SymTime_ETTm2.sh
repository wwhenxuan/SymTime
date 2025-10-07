export CUDA_VISIBLE_DEVICES=0

stride = 1
batch_size = 64
lr = 0.0001
lradj = "type2"


python -u imputation.py \ 
  --task_name "imputation" \ 
  --dataset_name "ETTm2" \ 
  --data "ETTm2" \ 
  --root_path "./datasets/ETT/" \ 
  --data_path "ETTm2.csv" \ 
  --mask_rate 0.125 \ 
  --forward_layers 4 \ 
  --stride $stride \ 
  --batch_size $batch_size \ 
  --learning_rate $lr \ 
  --lradj $lradj \


python -u imputation.py \ 
  --task_name "imputation" \ 
  --dataset_name "ETTm2" \ 
  --data "ETTm2" \ 
  --root_path "./datasets/ETT/" \ 
  --data_path "ETTm2.csv" \ 
  --mask_rate 0.25 \ 
  --forward_layers 2 \ 
  --stride $stride \ 
  --batch_size $batch_size \ 
  --learning_rate $lr \ 
  --lradj $lradj \


python -u imputation.py \ 
  --task_name "imputation" \ 
  --dataset_name "ETTm2" \ 
  --data "ETTm2" \ 
  --root_path "./datasets/ETT/" \ 
  --data_path "ETTm2.csv" \ 
  --mask_rate 0.375 \ 
  --forward_layers 2 \ 
  --stride $stride \ 
  --batch_size $batch_size \ 
  --learning_rate $lr \ 
  --lradj $lradj \


python -u imputation.py \ 
  --task_name "imputation" \ 
  --dataset_name "ETTm2" \ 
  --data "ETTm2" \ 
  --root_path "./datasets/ETT/" \ 
  --data_path "ETTm2.csv" \ 
  --mask_rate 0.50 \ 
  --forward_layers 2 \ 
  --stride $stride \ 
  --batch_size $batch_size \ 
  --learning_rate $lr \ 
  --lradj $lradj \
