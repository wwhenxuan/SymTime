export CUDA_VISIBLE_DEVICES=4


python -u classification.py \
  --task_name "classification" \
  --dataset_name "Heartbeat" \
  --data "UEA" \
  --root_path "./datasets/Heartbeat" \
  --forward_layers 2 \
  --out_channels 32 \
  --stride 16 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --num_workers 0 \
  --lradj "type1" \
  --patience 5 \
  