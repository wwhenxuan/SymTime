export CUDA_VISIBLE_DEVICES=5


python -u classification.py \ 
  --task_name "classification" \ 
  --dataset_name "FaceDetection" \ 
  --data "UEA" \ 
  --root_path "./datasets/FaceDetection" \ 
  --forward_layers 1 \ 
  --out_channels 32 \ 
  --stride 1 \ 
  --batch_size 32 \ 
  --learning_rate 0.0003 \ 
  --num_workers 0 \ 
  --lradj "type1" \   
