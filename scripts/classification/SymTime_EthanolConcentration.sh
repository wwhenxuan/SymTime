export CUDA_VISIBLE_DEVICES=5


python -u classification.py \ 
  --task_name "classification" \ 
  --dataset_name "EthanolConcentration" \ 
  --data "UEA" \ 
  --root_path "./datasets/EthanolConcentration" \ 
  --forward_layers 2 \ 
  --out_channels 16 \ 
  --stride 16 \ 
  --batch_size 12 \ 
  --learning_rate 0.0003 \ 
  --num_workers 0 \ 
  --lradj "type1" \  
