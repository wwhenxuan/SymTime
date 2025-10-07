export CUDA_VISIBLE_DEVICES=5


python -u classification.py \ 
  --task_name "classification" \ 
  --dataset_name "UWaveGestureLibrary" \ 
  --data "UEA" \ 
  --root_path "./datasets/UWaveGestureLibrary" \ 
  --forward_layers 3 \ 
  --out_channels 64 \ 
  --stride 8 \ 
  --batch_size 16 \ 
  --learning_rate 0.0003 \ 
  --num_workers 0 \ 
  --lradj "type1" \ 
