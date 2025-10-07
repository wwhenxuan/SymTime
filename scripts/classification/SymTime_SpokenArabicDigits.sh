export CUDA_VISIBLE_DEVICES=2


python -u classification.py \ 
  --task_name "classification" \ 
  --dataset_name "SpokenArabicDigits" \ 
  --data "UEA" \ 
  --root_path "./datasets/SpokenArabicDigits" \ 
  --forward_layers 2 \ 
  --out_channels 13 \ 
  --stride 4 \ 
  --batch_size 48 \ 
  --learning_rate 0.0002 \ 
  --num_workers 0 \ 
  --lradj "type1" \ 
  --patience 16 \ 
  