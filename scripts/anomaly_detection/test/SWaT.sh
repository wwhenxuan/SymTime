export CUDA_VISIBLE_DEVICES=7


python -u anomaly_detection.py \ 
  --task_name "anomaly_detection" \ 
  --dataset_name "SWaT" \ 
  --data "SWAT" \ 
  --root_path "./datasets/SWaT" \ 
  --anomaly_ratio 1 \ 
  --forward_layers 3 \ 
  --stride 8 \ 
  --batch_size 32 \ 
  --learning_rate 0.00025 \ 
  --lradj "cosine" \ 
  --enc_in 51 \ 
  --train_epochs 4 \ 
  --seq_len 100 \ 
