export CUDA_VISIBLE_DEVICES=5


seq_len = 100


python -u anomaly_detection.py \  
  --task_name "anomaly_detection" \ 
  --dataset_name "MSL" \ 
  --data "MSL" \ 
  --root_path "./datasets/MSL" \ 
  --anomaly_ratio 1 \ 
  --forward_layers 3 \ 
  --stride 8 \ 
  --batch_size 16 \ 
  --learning_rate 0.0005 \ 
  --lradj "type2" \ 
  --enc_in 55 \ 
  --train_epochs 5 \ 
  --seq_len $seq_len \ 