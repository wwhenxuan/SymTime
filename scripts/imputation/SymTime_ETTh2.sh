export CUDA_VISIBLE_DEVICES=0


python -u imputation.py \ 
  --task_name "imputation" \ 
  --dataset_name "ETTh2" \ 
  --data "ETTh2" \ 
  --root_path "./datasets/ETT/" \ 
  --data_path "ETTh2.csv" \ 
  --mask_rate 0.125 \ 
  --forward_layers $forward_layers \ 
  --stride $stride \ 
  --batch_size $batch_size \ 
  --learning_rate $lr \ 
  --lradj $lradj \


python -u imputation.py \ 
  --task_name "imputation" \ 
  --dataset_name "ETTh2" \ 
  --data "ETTh2" \ 
  --root_path "./datasets/ETT/" \ 
  --data_path "ETTh2.csv" \ 
  --mask_rate 0.25 \ 
  --forward_layers $forward_layers \ 
  --stride $stride \ 
  --batch_size $batch_size \ 
  --learning_rate $lr \ 
  --lradj $lradj \


python -u imputation.py \ 
  --task_name "imputation" \ 
  --dataset_name "ETTh2" \ 
  --data "ETTh2" \ 
  --root_path "./datasets/ETT/" \ 
  --data_path "ETTh2.csv" \ 
  --mask_rate 0.375 \ 
  --forward_layers $forward_layers \ 
  --stride $stride \ 
  --batch_size $batch_size \ 
  --learning_rate $lr \ 
  --lradj $lradj \


python -u imputation.py \ 
  --task_name "imputation" \ 
  --dataset_name "ETTh2" \ 
  --data "ETTh2" \ 
  --root_path "./datasets/ETT/" \ 
  --data_path "ETTh2.csv" \ 
  --mask_rate 0.50 \ 
  --forward_layers $forward_layers \ 
  --stride $stride \ 
  --batch_size $batch_size \ 
  --learning_rate $lr \ 
  --lradj $lradj \
