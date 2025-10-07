export CUDA_VISIBLE_DEVICES=0

stride = 1


python -u imputation.py \ 
  --task_name "imputation" \ 
  --dataset_name "ETTh1" \ 
  --data "ETTh1" \ 
  --root_path "./datasets/ETT/" \ 
  --data_path "ETTh1.csv" \ 
  --mask_rate 0.125 \ 
  --forward_layers 2 \ 
  --stride $stride \ 
  --batch_size 64 \ 
  --learning_rate 0.00025 \ 
  --lradj "cosine" \


python -u imputation.py \ 
  --task_name "imputation" \ 
  --dataset_name "ETTh1" \ 
  --data "ETTh1" \ 
  --root_path "./datasets/ETT/" \ 
  --data_path "ETTh1.csv" \ 
  --mask_rate 0.25 \ 
  --forward_layers 3 \ 
  --stride $stride \ 
  --batch_size 4 \ 
  --learning_rate 0.0001 \ 
  --lradj "type1" \


python -u imputation.py \ 
  --task_name "imputation" \ 
  --dataset_name "ETTh1" \ 
  --data "ETTh1" \ 
  --root_path "./datasets/ETT/" \ 
  --data_path "ETTh1.csv" \ 
  --mask_rate 0.375 \ 
  --forward_layers 2 \ 
  --stride $stride \ 
  --batch_size 8 \ 
  --learning_rate 0.0001 \ 
  --lradj "type2" \


python -u imputation.py \ 
  --task_name "imputation" \ 
  --dataset_name "ETTh1" \ 
  --data "ETTh1" \ 
  --root_path "./datasets/ETT/" \ 
  --data_path "ETTh1.csv" \ 
  --mask_rate 0.50 \ 
  --forward_layers 4 \ 
  --stride 8 \ 
  --batch_size 20 \ 
  --learning_rate 0.003 \ 
  --lradj "cosine" \
  