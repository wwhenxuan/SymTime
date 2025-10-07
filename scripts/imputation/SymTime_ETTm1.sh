export CUDA_VISIBLE_DEVICES=0

stride = 1
batch_size = 64
lradj = "cosine"

python -u imputation.py \ 
  --task_name "imputation" \ 
  --dataset_name "ETTm1" \ 
  --data "ETTm1" \ 
  --root_path "./datasets/ETT/" \ 
  --data_path "ETTm1.csv" \ 
  --mask_rate 0.125 \ 
  --forward_layers 1 \ 
  --stride $stride \ 
  --batch_size $batch_size \ 
  --learning_rate 0.0002 \ 
  --lradj $lradj \


python -u imputation.py \ 
  --task_name "imputation" \ 
  --dataset_name "ETTm1" \ 
  --data "ETTm1" \ 
  --root_path "./datasets/ETT/" \ 
  --data_path "ETTm1.csv" \ 
  --mask_rate 0.25 \ 
  --forward_layers 2 \ 
  --stride $stride \ 
  --batch_size $batch_size \ 
  --learning_rate 0.0001 \ 
  --lradj $lradj \


python -u imputation.py \ 
  --task_name "imputation" \ 
  --dataset_name "ETTm1" \ 
  --data "ETTm1" \ 
  --root_path "./datasets/ETT/" \ 
  --data_path "ETTm1.csv" \ 
  --mask_rate 0.375 \ 
  --forward_layers 1 \ 
  --stride $stride \ 
  --batch_size 0.0003 \ 
  --learning_rate $lr \ 
  --lradj $lradj \


python -u imputation.py \ 
  --task_name "imputation" \ 
  --dataset_name "ETTm1" \ 
  --data "ETTm1" \ 
  --root_path "./datasets/ETT/" \ 
  --data_path "ETTm1.csv" \ 
  --mask_rate 0.50 \ 
  --forward_layers 1 \ 
  --stride $stride \ 
  --batch_size $batch_size \ 
  --learning_rate 0.0002 \ 
  --lradj $lradj \
