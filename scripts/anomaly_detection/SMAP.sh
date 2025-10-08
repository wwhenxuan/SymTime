export CUDA_VISIBLE_DEVICES=5

python -u anomaly_detection.py \
  --task_name "anomaly_detection" \
  --dataset_name "SMAP" \
  --data "SMAP" \
  --root_path "./datasets/SMAP" \
  --anomaly_ratio 1 \
  --forward_layers 3 \
  --stride 8 \
  --batch_size 16 \
  --learning_rate 0.00025 \
  --lradj "type1" \
  --enc_in 25 \
  --train_epochs 5 \
  --seq_len 100 \
