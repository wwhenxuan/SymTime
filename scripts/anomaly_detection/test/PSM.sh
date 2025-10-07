export CUDA_VISIBLE_DEVICES=15


seq_len = 100
forward_layers = 3
stride = 8
batch_size = 8
lr = 0.00025
lradj = "type1"


python -u anomaly_detection.py \
  --task_name "anomaly_detection" \
  --dataset_name "PSM" \
  --data "PSM" \
  --root_path "./datasets/PSM" \
  --anomaly_ratio 1 \
  --forward_layers $forward_layers \
  --stride $stride \
  --batch_size $batch_size \
  --learning_rate $lr \
  --lradj $lradj \
  --enc_in 4 \
  --train_epochs 12 \
  --seq_len $seq_len \
