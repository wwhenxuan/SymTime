export CUDA_VISIBLE_DEVICES=5


python -u classification.py \
  --task_name "classification" \
  --dataset_name "Handwriting" \
  --data "UEA" \
  --root_path "./datasets/Handwriting" \
  --forward_layers 2 \
  --conv1d True \
  --out_channels 32 \
  --stride 16 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --num_workers 0 \
  --lradj "type1" \
